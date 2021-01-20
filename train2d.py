import os
import sys
import re
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import json
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid
from optimization import BertAdam

import segmentation_models_pytorch as smp
from networks.segtran2d import Segtran2d, set_segtran2d_config
from networks.segtran2d import CONFIG as config
import networks.deeplab as deeplab
from networks.nested_unet import UNet, NestedUNet
from networks.unet_3plus.unet_3plus import UNet_3Plus
from networks.pranet.PraNet_Res2Net import PraNet
from utils.losses import dice_loss_indiv, dice_loss_mix
from torchvision import transforms
import dataloaders.datasets2d
from dataloaders.datasets2d import refuge_map_mask, polyp_map_mask, reshape_mask, index_to_onehot
from train_util import AverageMeters, get_default, get_filename, get_seg_colormap
import imgaug.augmenters as iaa
import imgaug as ia

def print0(*print_args, **kwargs):
    if args.local_rank == 0:
        print(*print_args, **kwargs)

parser = argparse.ArgumentParser()
parser.add_argument('--task', dest='task_name', type=str, default='refuge', help='Name of the segmentation task.')
parser.add_argument('--ds', dest='ds_names', type=str, default=None, help='Dataset folders. Can specify multiple datasets (separated by ",")')
parser.add_argument('--split', dest='ds_split', type=str, default='train', 
                    choices=['train', 'all'], help='Split of the dataset')
                    
parser.add_argument('--maxiter', type=int,  default=10000, help='maximum training iterations')
parser.add_argument('--saveiter', type=int,  default=500, help='save model snapshot every N iterations')

parser.add_argument('--lrwarmup', dest='lr_warmup_steps', type=int,  default=500, help='Number of LR warmup steps')
parser.add_argument('--dicewarmup', dest='dice_warmup_steps', type=int,  default=0, help='Number of dice warmup steps (0: disabled)')
parser.add_argument('--bs', dest='batch_size', type=int, default=4, help='Total batch_size on all GPUs')
parser.add_argument('--opt', type=str,  default=None, help='optimization algorithm')
parser.add_argument('--lr', type=float,  default=-1, help='learning rate')
parser.add_argument('--decay', type=float,  default=-1, help='weight decay')
parser.add_argument('--gradclip', dest='grad_clip', type=float,  default=-1, help='gradient clip')
parser.add_argument('--attnclip', dest='attn_clip', type=int,  default=500, help='Segtran attention clip')
parser.add_argument('--cp', dest='checkpoint_path', type=str,  default=None, help='Load this checkpoint')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--locprob", dest='localization_prob', default=0, 
                    type=float, help='Probability of doing localization during training')
parser.add_argument("--tunebn", dest='tune_bn_only', action='store_true', 
                    help='Only tune batchnorms for domain adaptation, and keep model weights unchanged.')
parser.add_argument('--diceweight', dest='MAX_DICE_W', type=float, default=0.5, 
                    help='Weight of the dice loss.')

parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument("--debug", dest='debug', action='store_true', help='Debug program.')

parser.add_argument('--schedule', dest='lr_schedule', default='linear', type=str,
                    choices=['linear', 'constant', 'cosine'],
                    help='AdamW learning rate scheduler.')

parser.add_argument('--net', type=str,  default='segtran', help='Network architecture')

parser.add_argument('--bb', dest='backbone_type', type=str,  default='eff-b4', help='Backbone of Segtran / Encoder of other models')
parser.add_argument("--nopretrain", dest='use_pretrained', action='store_false', 
                    help='Do not use pretrained weights.')
# parser.add_argument('--ibn', dest='ibn_layers', type=str,  default=None, help='IBN layers')

parser.add_argument("--translayers", dest='num_translayers', default=1,
                    type=int, help='Number of Cross-Frame Fusion layers.')
parser.add_argument('--layercompress', dest='translayer_compress_ratios', type=str, default=None, 
                    help='Compression ratio of channel numbers of each transformer layer to save RAM.')
parser.add_argument("--baseinit", dest='base_initializer_range', default=0.02,
                    type=float, help='Base initializer range of transformer layers.')

parser.add_argument("--nosqueeze", dest='use_squeezed_transformer', action='store_false', 
                    help='Do not use attractor transformers (Default: use to increase scalability).')
parser.add_argument("--attractors", dest='num_attractors', default=256,
                    type=int, help='Number of attractors in the squeezed transformer.')
                    
parser.add_argument("--fusion", dest='apply_attn_stage', default='early',
                    choices=['early', 'late'],
                    type=str, help='Stage of attention-based feature fusion')
parser.add_argument("--poslayer1", dest='pos_embed_every_layer', action='store_false', 
                    help='Only add pos embedding to the first transformer layer input (Default: add to every layer).')
                    
parser.add_argument("--infpn", dest='in_fpn_layers', default='34',
                    choices=['234', '34', '4'],
                    help='Specs of input FPN layers')
parser.add_argument("--outfpn", dest='out_fpn_layers', default='1234',
                    choices=['1234', '234', '34'],
                    help='Specs of output FPN layers')

parser.add_argument("--outdrop", dest='out_fpn_do_dropout', action='store_true', 
                    help='Do dropout on out fpn features.')
parser.add_argument("--inbn", dest='in_fpn_use_bn', action='store_true', 
                    help='Use BatchNorm instead of GroupNorm in input FPN.')
parser.add_argument("--nofeatup", dest='bb_feat_upsize', action='store_false', 
                    help='Do not upsize backbone feature maps by 2.')

parser.add_argument('--insize', dest='orig_input_size', type=str, default=None, 
                    help='Use images of this size (among all cropping sizes) for training. Set to 0 to use all sizes.')
parser.add_argument('--patch', dest='patch_size', type=str, default=None, 
                    help='Resize input images to this size for training.')
                    
# Using random scaling as augmentation usually hurts performance. Not sure why.
parser.add_argument("--randscale", type=float, default=0.2, help='Do random scaling augmentation.')
parser.add_argument("--affine", dest='do_affine', action='store_true', help='Do random affine augmentation.')
parser.add_argument("--gray", dest='gray_alpha', type=float, default=0.5, 
                    help='Convert images to grayscale by so much degree.')
parser.add_argument("--reshape", dest='reshape_mask_type', type=str, default=None, 
                    choices=[None, 'rectangle', 'ellipse'],
                    help='Intentionally reshape the mask to test how well the model fits the mask bias.')

parser.add_argument('--dropout', type=float, dest='dropout_prob', default=-1, help='Dropout probability')
parser.add_argument('--modes', type=int, dest='num_modes', default=-1, help='Number of transformer modes')
parser.add_argument('--modedim', type=int, dest='attention_mode_dim', default=-1, help='Dimension of transformer modes')
parser.add_argument('--focus', dest='focus_class', type=int, default=-1, help='The class that is particularly predicted (with higher loss weight)')
parser.add_argument('--ablatepos', dest='ablate_pos_embed_type', type=str, default=None, 
                    choices=[None, 'zero', 'rand', 'sinu'],
                    help='Ablation to positional encoding schemes')
parser.add_argument('--multihead', dest='ablate_multihead', action='store_true', 
                    help='Ablation to multimode transformer (using multihead instead)')

args_dict = {  'trans_output_type': 'private',
               'mid_type': 'shared',
               'in_fpn_scheme':     'AN',
               'out_fpn_scheme':    'AN',
            }

args = parser.parse_args()
for arg, v in args_dict.items():
    args.__dict__[arg] = v

if args.ablate_multihead:
    args.use_squeezed_transformer = False
    
unet_settings    = { 'opt': 'adam', 
                     'lr': { 'sgd': 0.01, 'adam': 0.001 },
                     'decay': 0.0001, 'grad_clip': -1,
                   }
segtran_settings = { 'opt': 'adamw',
                     'lr': { 'adamw': 0.0002 },
                     'decay': 0.0001,  'grad_clip': 0.1,
                     'dropout_prob': { '234': 0.3, '34': 0.2, '4': 0.2 },
                     'num_modes':    { '234': 2,   '34': 4,   '4': 4 }
                   }

default_settings = { 'unet':            unet_settings,
                     'unet-scratch':    unet_settings,
                     'nestedunet':      unet_settings,
                     'unet3plus':       unet_settings,
                     'deeplabv3plus':   unet_settings,
                     'deeplab-smp':     unet_settings,
                     'pranet':          unet_settings,
                     'segtran':         segtran_settings,
                     'refuge': {
                                 'num_classes': 3,
                                 'bce_weight':  [0., 1, 2],
                                 'ds_class':    'SegCrop',
                                 'ds_names':    'train,valid,test,drishiti,rim',
                                 'orig_input_size': 576,
                                 # Each dim of the patch_size should be multiply of 32.
                                 'patch_size':      288,
                                 'uncropped_size': { 'train':    (2056, 2124), 
                                                     'test':     (1634, 1634),
                                                     'valid':    (1634, 1634),
                                                     'valid2':   (1940, 1940),
                                                     'test2':    -1,    # varying sizes
                                                     'drishiti': (2050, 1750),
                                                     'rim':      (2144, 1424)
                                                   },
                                 'has_mask':    { 'train': True,    'test': True, 
                                                  'valid': True,    'valid2': False,
                                                  'test2': False,
                                                  'drishiti': True, 'rim': True },
                                 'weight':      { 'train': 1,       'test': 1, 
                                                  'valid': 1,       'valid2': 1,
                                                  'test2': 1,
                                                  'drishiti': 1,    'rim': 1 },
                                 'orig_dir':    { 'test2': 'test2_orig' },
                                 'orig_ext':    { 'test2': '.jpg' },
                               },
                     'polyp':  {
                                 'num_classes': 2,
                                 'bce_weight':  [0., 1],
                                 'ds_class':    'SegWhole',
                                 'ds_names': 'CVC-ClinicDB-train,Kvasir-train',
                                 # Actual images are at various sizes. As the dataset is SegWhole, orig_input_size is ignored.
                                 # But output_upscale is computed as the ratio between orig_input_size and patch_size.
                                 # So set it to the same as patch_size to avoid output upscaling.
                                 # Setting orig_input_size to -1 also leads to output_upscale = 1.
                                 # All images of different sizes are resized to 320*320.
                                 'orig_input_size': 320,    
                                 'patch_size':      320,
                                 'has_mask':    { 'CVC-ClinicDB-train': True,   'Kvasir-train': True, 
                                                  'CVC-ClinicDB-test': True,    'Kvasir-test': True, 
                                                  'CVC-300': True,              'CVC-ColonDB': False,
                                                  'ETIS-LaribPolypDB': True },
                                 'weight':      { 'CVC-ClinicDB-train': 1,      'Kvasir-train': 1, 
                                                  'CVC-ClinicDB-test': 1,       'Kvasir-test': 1, 
                                                  'CVC-300': 1,                 'CVC-ColonDB': 1,
                                                  'ETIS-LaribPolypDB': 1  }
                               },
                     'oct':  {
                                 'num_classes': 10,
                                 'bce_weight':  [0., 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 'ds_class':    'SegWhole',
                                 'ds_names':    'duke',
                                 # Actual images are at various sizes. As the dataset is SegWhole, orig_input_size is ignored.
                                 # But output_upscale is computed as the ratio between orig_input_size and patch_size.
                                 # If you want to avoid avoid output upscaling, set orig_input_size to the same as patch_size.
                                 # The actual resolution of duke is (296, 500~542). 
                                 # Set to (288, 512) will crop the central areas.
                                 # The actual resolution of pcv is (633, 720). Removing 9 pixels doesn't matter.
                                 'orig_input_size': { 'duke': (288, 512), 'seed': (1024, 512), 'pcv': (624, 720) } ,
                                 'patch_size':      { 'duke': (288, 512), 'seed': (512,  256), 'pcv': (312, 360) }, 
                                 'has_mask':        { 'duke': True,       'seed': False,       'pcv': False },
                                 'weight':          { 'duke': 1,          'seed': 1,           'pcv': 1 }
                               },
                   }

get_default(args, 'orig_input_size',    default_settings, None,   [args.task_name, 'orig_input_size'])
get_default(args, 'patch_size',         default_settings, None,   [args.task_name, 'patch_size'])
if type(args.patch_size) == str:
    args.patch_size = [ int(length) for length in args.patch_size.split(",") ]
    if len(args.patch_size) == 1:
        args.patch_size = (args.patch_size[0], args.patch_size[0])    
if type(args.patch_size) == int:
    args.patch_size = (args.patch_size, args.patch_size)
if type(args.orig_input_size) == str:
    args.orig_input_size = [ int(length) for length in args.orig_input_size.split(",") ]
if type(args.orig_input_size) == int:
    args.orig_input_size = (args.orig_input_size, args.orig_input_size)
if args.orig_input_size[0] > 0:
    args.output_upscale = args.orig_input_size[0] / args.patch_size[0]
else:
    args.output_upscale = 1

get_default(args, 'ds_names',           default_settings, None, [args.task_name, 'ds_names'])

ds_stats_map = { 'refuge': 'refuge-cropped-gray{:.1f}-stats.json',
                 'polyp':  'polyp-whole-gray{:.1f}-stats.json',
                 'oct':    'oct-whole-gray{:.1f}-stats.json' }

stats_file_tmpl = ds_stats_map[args.task_name]
stats_filename = stats_file_tmpl.format(args.gray_alpha)
ds_stats = json.load(open(stats_filename))
default_settings[args.task_name].update(ds_stats)
print0("'{}' mean/std loaded from '{}'".format(args.task_name, stats_filename))
    
args.ds_names = args.ds_names.split(",")
train_data_paths = []

for ds_name in args.ds_names:
    train_data_path = os.path.join("../data/", args.task_name, ds_name)
    train_data_paths.append(train_data_path)

args.job_name = '{}-{}'.format(args.task_name, ','.join(args.ds_names))

timestamp = datetime.now().strftime("%m%d%H%M")
checkpoint_dir = "../model/%s-%s-%s" %(args.net, args.job_name, timestamp)
print0("Model checkpoints will be saved to '%s'" %checkpoint_dir)
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def init_optimizer(net, max_epoch, batch_per_epoch):
    # Prepare optimizer
    # Each param is a tuple ( param name, Parameter(tensor(...)) )
    optimized_params = list( param for param in net.named_parameters() if param[1].requires_grad )
    low_decay = ['backbone'] #['bias', 'LayerNorm.weight']
    no_decay = []
    high_lr = ['alphas']
    
    high_lr_params = []
    high_lr_names = []
    no_decay_params = []
    no_decay_names = []
    low_decay_params = []
    low_decay_names = []
    normal_params = []
    normal_names = []

    for n, p in optimized_params:
        if any(nd in n for nd in no_decay):
            no_decay_params.append(p)
            no_decay_names.append(n)
        elif any(nd in n for nd in low_decay):
            low_decay_params.append(p)
            low_decay_names.append(n)
        elif any(nd in n for nd in high_lr):
            high_lr_params.append(p)
            high_lr_names.append(n)
        else:
            normal_params.append(p)
            normal_names.append(n)

    optimizer_grouped_parameters = [
        { 'params': normal_params,       'weight_decay': args.decay,        'lr': args.lr },
        { 'params': low_decay_params,    'weight_decay': args.decay * 0.1,  'lr': args.lr },
        { 'params': no_decay_params,     'weight_decay': 0.0,               'lr': args.lr },
        { 'params': high_lr_params,      'weight_decay': 0.0,               'lr': args.lr * 100 },
    ]

    for group_name, param_group in zip( ('normal', 'low_decay', 'no_decay', 'high_lr'), 
                                        (normal_params, low_decay_params, no_decay_params, high_lr_params) ):
        print0("{}: {} weights".format(group_name, len(param_group)))
        
    args.t_total = int(batch_per_epoch * max_epoch)
    print0("Batch per epoch: %d" % batch_per_epoch)
    print0("Total Iters: %d" % args.t_total)
    print0("LR: %f" %args.lr)

    args.lr_warmup_ratio = args.lr_warmup_steps / args.t_total
    print0("LR Warm up: %.3f=%d iters" % (args.lr_warmup_ratio, args.lr_warmup_steps))

    optimizer = BertAdam(optimizer_grouped_parameters,
                         warmup=args.lr_warmup_ratio, t_total=args.t_total,
                         weight_decay=args.decay)

    return optimizer

def load_model(net, optimizer, args, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    params = net.state_dict()
    if 'model' in state_dict:
        model_state_dict = state_dict['model']
        optim_state_dict = state_dict['optim_state']
        cp_args          = state_dict['args']
        cp_iter_num      = state_dict['iter_num']
    else:
        model_state_dict = state_dict
        optim_state_dict = None
        cp_args          = None
        cp_iter_num      = 0
        
    ignored_keys = [ 'maxiter', 'checkpoint_path', 'model_input_size', 't_total', 'num_workers',
                     'lr_warmup_ratio', 'lr_warmup_steps', 'local_rank', 'distributed', 'world_size',
                     'saveiter', 'dice_warmup_steps', 'opt', 'lr', 'decay',
                     'initializer_range', 'base_initializer_range',
                     'grad_clip', 'localization_prob', 'tune_bn_only', 'MAX_DICE_W', 'deterministic',
                     'lr_schedule', 'out_fpn_do_dropout', 'randscale', 'do_affine', 'focus_class',
                     'bce_weight', 'translayer_compress_ratios',
                     'seed', 'debug', 'ds_name', 'batch_size', 'dropout_prob',
                     'patch_size', 'orig_input_size', 'output_upscale',
                     'checkpoint_dir', 'iters', 'out_origsize', 'out_softscores', 'verbose_output',
                     'gpu', 'test_interp', 'do_remove_frag', 'reload_mask', 'ds_split', 'ds_names',
                     'job_name', 'mean', 'std', 'mask_thres', ]
    warn_keys = [ 'num_recurrences' ]
                        
    if args.net == 'segtran' and cp_args is not None:
        for k in cp_args:
            if (k in warn_keys) and (args.__dict__[k] != cp_args[k]):
                print("args[{}]={}, checkpoint args[{}]={}, inconsistent!".format(k, args.__dict__[k], k, cp_args[k]))
                continue

            if (k not in ignored_keys) and (args.__dict__[k] != cp_args[k]):
                print("args[{}]={}, checkpoint args[{}]={}, inconsistent!".format(k, args.__dict__[k], k, cp_args[k]))
                exit(0)
                    
    params.update(model_state_dict)
    net.load_state_dict(params)
    if optim_state_dict is not None:
        optimizer.load_state_dict(optim_state_dict)
                
    logging.info("Model loaded from '{}'".format(checkpoint_path))
    # warmup info is mainly in optim_state_dict. So after loading the checkpoint, 
    # the optimizer won't do warmup already.
    args.lr_warmup_steps = 0
    print0("LR Warm up reset to 0 iters.")
    
    return cp_iter_num
    
def save_model(net, optimizer, args, checkpoint_dir, iter_num):
    if args.local_rank == 0:
        save_model_path = os.path.join(checkpoint_dir, 'iter_'+str(iter_num)+'.pth')
        torch.save( { 'iter_num': iter_num, 'model': net.state_dict(),
                      'optim_state': optimizer.state_dict(),
                      'args': vars(args) },  
                    save_model_path)
                        
        logging.info("save model to '{}'".format(save_model_path))

def warmup_constant(x, warmup=500):
    if x < warmup:
        return x/warmup
    return 1
         
if __name__ == "__main__":
    logFormatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    rootLogger = logging.getLogger()
    while rootLogger.handlers:
         rootLogger.handlers.pop()
    fileHandler = logging.FileHandler(checkpoint_dir+"/log.txt")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)
    rootLogger.propagate = False

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        print0("Set this session to deterministic mode")

    if args.tune_bn_only:
        if args.checkpoint_path is None:
            print0("Tuning BN requires to specify a checkpoint to load")
            exit(0)
        args.lr_warmup_steps = 0
        
    is_master = (args.local_rank == 0)
    n_gpu = torch.cuda.device_count()
    args.device = 'cuda'
    args.distributed = False
    # 'WORLD_SIZE' is set by 'torch.distributed.launch'. Do not set manually
    # Its value is specified by "--nproc_per_node=k"
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = args.world_size > 1
    else:
        args.world_size = 1

    args.batch_size //= args.world_size
    print0("n_gpu: {}, world size: {}, rank: {}, batch size: {}, seed: {}".format(
                n_gpu, args.world_size, args.local_rank, args.batch_size,
                args.seed))

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://',
                                             world_size=args.world_size,
                                             rank=args.local_rank)

    get_default(args, 'opt',            default_settings, None,   [args.net, 'opt'])
    get_default(args, 'lr',             default_settings, -1,     [args.net, 'lr', args.opt])
    get_default(args, 'decay',          default_settings, -1,     [args.net, 'decay'])
    get_default(args, 'grad_clip',      default_settings, -1,     [args.net, 'grad_clip'])
    get_default(args, 'num_classes',    default_settings, None,   [args.task_name, 'num_classes'])
    args.binarize = (args.num_classes == 2)

    get_default(args, 'bce_weight',     default_settings, None,   [args.task_name, 'bce_weight'])
    get_default(args, 'ds_class',       default_settings, None,   [args.task_name, 'ds_class'])
    
    args.bce_weight = torch.tensor(args.bce_weight).cuda()
    args.bce_weight = args.bce_weight * (args.num_classes - 1) / args.bce_weight.sum()
            
    if args.net == 'segtran':
        get_default(args, 'dropout_prob',   default_settings, -1, [args.net, 'dropout_prob', args.in_fpn_layers])
        get_default(args, 'num_modes',      default_settings, -1, [args.net, 'num_modes', args.in_fpn_layers])
    
    if args.randscale > 0:
        crop_percents = (-args.randscale, args.randscale)
    else:
        crop_percents = (0, 0)
    
    # Images after augmentation/transformation should keep their original size orig_input_size.  
    # Will be resized before fed into the model.  
    tgt_width, tgt_height = args.orig_input_size
    
    if args.do_affine:
        affine_prob = 0.3
    else:
        affine_prob = 0
        
    common_aug_func =   iaa.Sequential(
                            [
                                # resize the image to the shape of orig_input_size
                                iaa.Resize({'height': tgt_height, 'width': tgt_width}),   
                                iaa.Sometimes(0.5, iaa.CropAndPad(
                                    percent=crop_percents,
                                    pad_mode='constant', # ia.ALL,
                                    pad_cval=0
                                )),
                                # apply the following augmenters to most images
                                iaa.Fliplr(0.2),  # Horizontally flip 20% of all images
                                iaa.Flipud(0.2),  # Vertically flip 20% of all images
                                iaa.Sometimes(0.3, iaa.Rot90((1,3))), # Randomly rotate 90, 180, 270 degrees 30% of the time
                                # Affine transformation reduces dice by ~1%. So disable it by setting affine_prob=0.
                                iaa.Sometimes(affine_prob, iaa.Affine(
                                        rotate=(-45, 45), # rotate by -45 to +45 degrees
                                        shear=(-16, 16), # shear by -16 to +16 degrees
                                        order=1,
                                        cval=(0,255),
                                        mode='reflect'
                                )),
                                # iaa.Sometimes(0.3, iaa.GammaContrast((0.7, 1.7))),    # Gamma contrast degrades.
                                # When tgt_width==tgt_height, PadToFixedSize and CropToFixedSize are unnecessary.
                                # Otherwise, we have to take care if the longer edge is rotated to the shorter edge.
                                iaa.PadToFixedSize(width=tgt_width,  height=tgt_height),    
                                iaa.CropToFixedSize(width=tgt_width, height=tgt_height),
                                iaa.Grayscale(alpha=args.gray_alpha)
                            ])
                          
    DataSetClass = dataloaders.datasets2d.__dict__[args.ds_class]
    
    db_trains = []
    ds_settings = default_settings[args.task_name]
    
    for i, train_data_path in enumerate(train_data_paths):
        ds_name         = args.ds_names[i]
        ds_weight       = ds_settings['weight'][ds_name]
        if 'uncropped_size' in ds_settings:
            uncropped_size  = ds_settings['uncropped_size'][ds_name]
        else:
            uncropped_size  = -1

        if uncropped_size == -1 and 'orig_dir' in ds_settings:
            orig_dir  = ds_settings['orig_dir'][ds_name]
            orig_ext  = ds_settings['orig_ext'][ds_name]
        else:
            orig_dir = orig_ext = None
                        
        has_mask      = ds_settings['has_mask'][ds_name]
        mean          = ds_settings['mean'][ds_name]
        std           = ds_settings['std'][ds_name]

        image_trans_func =  transforms.Compose([   
                                transforms.RandomChoice([
                                    transforms.ColorJitter(brightness=0.2),
                                    transforms.ColorJitter(contrast=0.2), 
                                    transforms.ColorJitter(saturation=0.2),
                                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0), 
                                ]),     
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                            ])
                                
        db_train = DataSetClass(base_dir=train_data_path,
                                split=args.ds_split,
                                mode='train',
                                mask_num_classes=args.num_classes,
                                has_mask=has_mask,
                                ds_weight=ds_weight,
                                common_aug_func=common_aug_func,
                                image_trans_func=image_trans_func,
                                segmap_trans_func=None,
                                binarize=args.binarize,
                                train_loc_prob=args.localization_prob,
                                chosen_size=args.orig_input_size[0],   # ignored in SegWhole instances.
                                uncropped_size=uncropped_size,
                                min_output_size=args.patch_size,
                                orig_dir=orig_dir,
                                orig_ext=orig_ext)
                                
        db_trains.append(db_train)
        print0("{}: {} images, uncropped_size: {}, has_mask: {}".format(
                args.ds_names[i], len(db_train), uncropped_size, has_mask))
        
    db_train_combo = ConcatDataset(db_trains)
    print0("Combined dataset: {} images".format(len(db_train_combo)))

    # num_modalities is used in segtran.
    # num_modalities = 0 means there's not the modality dimension 
    # (but still a single modality) in the images loaded from db_train.
    args.num_modalities = 0
    if args.translayer_compress_ratios is not None:
        args.translayer_compress_ratios = [ float(r) for r in args.translayer_compress_ratios.split(",") ]
    else:
        args.translayer_compress_ratios = [ 1 for layer in range(args.num_translayers + 1) ]
    
    if args.distributed:
        train_sampler = DistributedSampler(db_train_combo, shuffle=True, 
                                           num_replicas=args.world_size, 
                                           rank=args.local_rank)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)

    args.num_workers = 0 if args.debug else 4
    trainloader = DataLoader(db_train_combo, batch_size=args.batch_size, sampler=train_sampler,
                             num_workers=args.num_workers, pin_memory=True, shuffle=shuffle,
                             worker_init_fn=worker_init_fn)
    max_epoch = math.ceil(args.maxiter / len(trainloader))
    
    logging.info(str(args))
    base_lr = args.lr

    if args.net == 'unet':
        # timm-efficientnet performs slightly worse.
        backbone_type = re.sub("^eff", "efficientnet", args.backbone_type)
        net = smp.Unet(backbone_type, classes=args.num_classes, encoder_weights='imagenet')
    elif args.net == 'unet-scratch':
        net = UNet(num_classes=args.num_classes)
    elif args.net == 'nestedunet':
        net = NestedUNet(num_classes=args.num_classes)
    elif args.net == 'unet3plus':
        net = UNet_3Plus(n_classes=args.num_classes)
    elif args.net == 'pranet':
        net = PraNet(num_classes=args.num_classes - 1)
    elif args.net.startswith('deeplab'):
        use_smp_deeplab = args.net.endswith('smp')
        if use_smp_deeplab:
            backbone_type = re.sub("^eff", "efficientnet", args.backbone_type)
            net = smp.DeepLabV3Plus(backbone_type, classes=args.num_classes, encoder_weights='imagenet')
        else:
            model_name = args.net + "_" + args.backbone_type
            model_map = {
                'deeplabv3_resnet50':       deeplab.deeplabv3_resnet50,
                'deeplabv3plus_resnet50':   deeplab.deeplabv3plus_resnet50,
                'deeplabv3_resnet101':      deeplab.deeplabv3_resnet101,
                'deeplabv3plus_resnet101':  deeplab.deeplabv3plus_resnet101,
                'deeplabv3_mobilenet':      deeplab.deeplabv3_mobilenet,
                'deeplabv3plus_mobilenet':  deeplab.deeplabv3plus_mobilenet
            }
            net = model_map[model_name](num_classes=args.num_classes, output_stride=8)
    elif args.net == 'segtran':
        set_segtran2d_config(args)
        net = Segtran2d(config)
    else:
        breakpoint()
        
    net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0001)
    elif args.opt == 'adamw':
        optimizer = init_optimizer(net, max_epoch, len(trainloader))
    
    if args.checkpoint_path is not None:
        iter_num = load_model(net, optimizer, args, args.checkpoint_path)
        start_epoch = math.ceil(iter_num / len(trainloader))
        logging.info("Start epoch/iter: {}/{}".format(start_epoch, iter_num))
    else:
        iter_num = 0
        start_epoch = 0
    
    if args.tune_bn_only:
        net.eval()
        if args.backbone_type.startswith('eff'):
            for idx, block in enumerate(net.backbone._blocks):
                # Stops at layer 3. Layers 0, 1, 2 are set to training mode (BN tunable).
                if idx == net.backbone.endpoint_blk_indices[3]:
                    print0("Tuning stops at block {} in backbone '{}'".format(idx, args.backbone_type))
                    break
                block.train()
        else:
            print0("Backbone '{}' not supported.".format(args.backbone_type))
            exit(0)
    else:
        net.train()
        
    real_net = net

    if args.distributed:
        sync_bn_net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        sync_bn_net.cuda()
        net = DistributedDataParallel(sync_bn_net, device_ids=[args.local_rank],
                                      output_device=args.local_rank,
                                      find_unused_parameters=True)

    scheduler = None
    lr_ = base_lr

    if is_master:
        writer = SummaryWriter(checkpoint_dir + '/log')
    logging.info("{} epochs, {} itertations each.".format(max_epoch, len(trainloader)))

    dice_loss_func = dice_loss_indiv
    class_weights = torch.ones(args.num_classes).cuda()
    class_weights[0] = 0
    if args.focus_class != -1 and args.num_classes > 2:
        class_weights[args.focus_class] = 2
    class_weights /= class_weights.sum()
    # img_stats = AverageMeters()
    # The weight of first class, i.e., the background is set to 0. Because it's negative
    # of WT. Optimizing w.r.t. WT is enough.
    # Positive voxels (ET, WT, TC) receive higher weights as they are fewer than negative voxels.
    bce_loss_func = nn.BCEWithLogitsLoss( # weight=weights_batch.view(-1,1,1,1),
                                          pos_weight=args.bce_weight)
    
    for epoch_num in tqdm(range(start_epoch, max_epoch), ncols=70, disable=(args.local_rank != 0)):
        print0()
        time1 = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch_num)

        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            image_batch, mask_batch = sampled_batch['image'].cuda(), sampled_batch['mask'].cuda()
            weights_batch = sampled_batch['weight'].cuda()
            if args.task_name == 'refuge':
                # after mapping, mask_batch is already float.
                mask_batch = refuge_map_mask(mask_batch)
            elif args.task_name == 'polyp':
                mask_batch = polyp_map_mask(mask_batch)
            elif args.task_name == 'oct':
                mask_batch = index_to_onehot(mask_batch, args.num_classes)
                
            # args.patch_size is typically set as 1/2 of args.orig_input_size.
            # Scale down the input images to save RAM. 
            # The mask tensor is kept the original size. The output segmap will 
            # be scaled up by a factor of args.output_upscale.
            image_batch = F.interpolate(image_batch, size=args.patch_size,
                                        mode='bilinear', align_corners=False)
                                      
            iter_num = iter_num + 1
            # image_batch: [4, 1, 112, 112, 80]
            # mask_batch: [4, 112, 112, 80]
            # outputs:      [4, 4, 112, 112, 80]
            # If args.tune_bn_only, only tune backbone BNs. 
            # Transformer group norms do not have running stats.
            if args.tune_bn_only:
                with torch.no_grad():
                    outputs = net(image_batch)

                if iter_num % 50 == 0:
                    save_model(real_net, optimizer, args, checkpoint_dir, iter_num)                    
                continue
                
            outputs = net(image_batch)
            if args.net == 'pranet':
                # Use lateral_map_2 for single-loss training.
                # Outputs is missing one channel (background). 
                # As the background doesn't incur any loss, its value doesn't matter. 
                # So add an all-zero channel to it.
                outputs0 = outputs[3]
                background = torch.zeros_like(outputs0[:, [0]])
                outputs = torch.cat([background, outputs0], dim=1)

            outputs = F.interpolate(outputs, size=mask_batch.shape[2:], 
                                    mode='bilinear', align_corners=False)
            dice_losses = []
            DICE_W = args.MAX_DICE_W # * warmup_constant(iter_num, args.dice_warmup_steps)
            
            # BCEWithLogitsLoss uses raw scores, so use outputs here instead of outputs_soft.
            # Permute the class dimension to the last dimension (required by BCEWithLogitsLoss).
            total_ce_loss   = bce_loss_func(outputs.permute([0, 2, 3, 1]), 
                                            mask_batch.permute([0, 2, 3, 1]))
            total_dice_loss = 0
            outputs_soft    = torch.sigmoid(outputs)
            
            for cls in range(1, args.num_classes):
                # bce_loss_func is actually nn.BCEWithLogitsLoss(), so use raw scores as input.
                # dice loss always uses sigmoid/softmax transformed probs as input.
                dice_loss = dice_loss_func(outputs_soft[:, cls], mask_batch[:, cls])
                dice_losses.append(dice_loss)
                total_dice_loss = total_dice_loss + dice_loss * class_weights[cls]
                
            loss = (1 - DICE_W) * total_ce_loss + DICE_W * total_dice_loss

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            if args.distributed:
                total_ce_loss   = reduce_tensor(total_ce_loss.data)
                total_dice_loss = reduce_tensor(total_dice_loss.data)
                dice_losses     = [ reduce_tensor(dice_loss.data) for dice_loss in dice_losses ]
                loss            = reduce_tensor(loss.data)
                
            if is_master:
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/total_ce_loss', total_ce_loss.item(), iter_num)
                writer.add_scalar('loss/total_dice_loss', total_dice_loss.item(), iter_num)
                writer.add_scalar('loss/loss', loss.item(), iter_num)
                if len(dice_losses) > 1:
                    dice_loss_str = ",".join( [ "%.4f" %dice_loss for dice_loss in dice_losses ] )
                    logging.info('%d loss: %.4f, ce: %.4f, dice: %.4f (%s)' % \
                                    (iter_num, loss.item(), total_ce_loss.item(),
                                     total_dice_loss.item(), dice_loss_str))
                else:
                    logging.info('%d loss: %.4f, ce: %.4f, dice: %.4f' % \
                                    (iter_num, loss.item(), total_ce_loss.item(),
                                     total_dice_loss.item()))
                                     
            if iter_num % 50 == 0 and is_master:
                grid_image = make_grid(image_batch, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)
                
                # make_grid can only handle <= 3 classes
                grid_image = make_grid(outputs_soft[:, :3], 5, normalize=False)
                writer.add_image('train/Predicted_mask', grid_image, iter_num)

                grid_image = make_grid(mask_batch[:, :3], 5, normalize=False)
                writer.add_image('train/Groundtruth_mask', grid_image, iter_num)

            ## change lr
            if args.opt == 'sgd' and iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % args.saveiter == 0:
                save_model(real_net, optimizer, args, checkpoint_dir, iter_num)
            if iter_num >= args.maxiter:
                break
            time1 = time.time()
        if iter_num >= args.maxiter:
            break

    if args.maxiter % args.saveiter != 0:
        save_model(real_net, optimizer, args, checkpoint_dir, iter_num)

    if is_master:
        writer.close()
