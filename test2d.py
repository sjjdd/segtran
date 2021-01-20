import os
import time
import re
from datetime import datetime
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from networks.segtran2d import Segtran2d, set_segtran2d_config
import networks.segtran_shared as segtran_shared
from networks.segtran2d import CONFIG as config
import networks.deeplab as deeplab
from networks.nested_unet import UNet, NestedUNet
from networks.unet_3plus.unet_3plus import UNet_3Plus
from networks.unet2d.unet_model import UNet as VanillaUNet
from networks.pranet.PraNet_Res2Net import PraNet
from test_util2d import test_all_cases, remove_fragmentary_segs
import dataloaders.datasets2d
from dataloaders.datasets2d import refuge_map_mask, refuge_inv_map_mask, polyp_map_mask, \
                                   polyp_inv_map_mask, reshape_mask, index_to_onehot, onehot_inv_map
import imgaug.augmenters as iaa
from train_util import get_default, get_filename, get_seg_colormap, visualize_model, eval_robustness
from functools import partial
import subprocess
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--task', dest='task_name', type=str, default='refuge', help='Name of the segmentation task.')
parser.add_argument('--ds', dest='ds_name', type=str, default='valid2', help='Dataset name for test')
parser.add_argument('--split', dest='ds_split', type=str, default='all',
                    choices=['train', 'test', 'all'], help='Split of the dataset')
parser.add_argument('--cpdir', dest='checkpoint_dir', type=str, default=None,
                    help='Load checkpoint(s) from this directory')
parser.add_argument('--iters', type=str,  default='8000,7000', help='checkpoint iteration(s)')
parser.add_argument('--bs', dest='batch_size', type=int, default=8, help='batch size')

parser.add_argument('--insize', dest='orig_input_size', type=str, default=None,
                    help='Use images of this size (among all cropping sizes) for training. Set to 0 to use all sizes.')
parser.add_argument('--patch', dest='patch_size', type=str, default=None,
                    help='Do test on such input image patches.')

parser.add_argument('--outorigsize', dest='out_origsize', action='store_true',
                    help='Output seg maps in the same size of original uncropped images')
parser.add_argument("--debug", dest='debug', action='store_true', help='Debug program.')
parser.add_argument("--verbose", dest='verbose_output', action='store_true', 
                    help='Output individual scores of each image.')

parser.add_argument('--gpu', type=str,  default='0', help='ID of GPU to use')
parser.add_argument('--net', type=str,  default='segtran', help='Network architecture')
parser.add_argument('--bb', dest='backbone_type', type=str,  default='eff-b4', help='Segtran backbone')

parser.add_argument("--nosqueeze", dest='use_squeezed_transformer', action='store_false',
                    help='Do not use attractor transformers (Default: use to increase scalability).')
parser.add_argument("--attractors", dest='num_attractors', default=256,
                    type=int, help='Number of attractors in the squeezed transformer.')

parser.add_argument("--translayers", dest='num_translayers', default=1,
                    type=int, help='Number of Cross-Frame Fusion layers.')
parser.add_argument('--layercompress', dest='translayer_compress_ratios', type=str, default=None, 
                    help='Compression ratio of channel numbers of each transformer layer to save RAM.')
parser.add_argument("--baseinit", dest='base_initializer_range', default=0.02,
                    type=float, help='Initializer range of transformer layers.')

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

parser.add_argument("--inbn", dest='in_fpn_use_bn', action='store_true',
                    help='Use BatchNorm instead of GroupNorm in input FPN.')
parser.add_argument('--attnclip', dest='attn_clip', type=int,  default=500, help='Segtran attention clip')

parser.add_argument('--modes', type=int, dest='num_modes', default=-1, help='Number of transformer modes')
parser.add_argument('--modedim', type=int, dest='attention_mode_dim', default=-1, help='Dimension of transformer modes')
parser.add_argument("--nofeatup", dest='bb_feat_upsize', action='store_false', 
                    help='Do not upsize backbone feature maps by 2.')
parser.add_argument("--testinterp", dest='test_interp', type=str, default=None,
                    help='Test how much error simple interpolation would cause. (Specify scaling ratio here)')
parser.add_argument("--gray", dest='gray_alpha', type=float, default=0.5,
                    help='Convert images to grayscale by so much degree.')
parser.add_argument("--removefrag", dest='do_remove_frag', action='store_true',
                    help='As a postprocessing step, remove fragmentary segments; only keep the biggest segment.')
parser.add_argument("--reloadmask", dest='reload_mask', action='store_true',
                    help='Reload mask directly from the gt file (ignoring the dataloader input). Used when input images are in varying sizes.')
parser.add_argument("--reshape", dest='reshape_mask_type', type=str, default=None,
                    choices=[None, 'rectangle', 'ellipse'],
                    help='Intentionally reshape the mask to test how well the model fits the mask bias.')
parser.add_argument("--t", dest='mask_thres', type=float, default=0.5,
                    help='The threshold of converting soft mask scores to 0/1.')
parser.add_argument('--ablatepos', dest='ablate_pos_embed_type', type=str, default=None,
                    choices=[None, 'zero', 'rand', 'sinu'],
                    help='Ablation to positional encoding schemes')
parser.add_argument('--multihead', dest='ablate_multihead', action='store_true',
                    help='Ablation to multimode transformer (using multihead instead)')
parser.add_argument('--vis', dest='vis_mode', type=str, default=None,
                    choices=[None, 'rf'],
                    help='Do visualization')
parser.add_argument('--robust', dest='eval_robustness', action='store_true',
                    help='Evaluate feature map robustness against augmentation.')
parser.add_argument('--augdeg', dest='aug_degree', type=float, default=0.5,
                    help='Augmentation degree when doing robustness evaluation.')

args_dict = {   'trans_output_type': 'private',
                'mid_type': 'shared',
                'in_fpn_scheme': 'AN',
                'out_fpn_scheme': 'AN',
                'use_pretrained': True, # Doesn't matter if we load a trained checkpoint.                    
            }

args = parser.parse_args()
for arg, v in args_dict.items():
    args.__dict__[arg] = v
    
if args.ablate_multihead:
    args.use_squeezed_transformer = False
    
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
test_data_parent = os.path.join("../data/", args.task_name)
test_data_path = os.path.join("../data/", args.task_name, args.ds_name)
if args.checkpoint_dir is not None:
    timestamp = args.checkpoint_dir.split("-")[-1]
    timestamp = timestamp.replace("/", "")
else:
    timestamp = ""
args.job_name = '{}-{}'.format(args.task_name, args.ds_name)

segtran_settings = {
                     'num_modes':  { '234': 2,   '34': 4,   '4': 4 }
                   }

default_settings = { 'unet':            {},
                     'unet-scratch':    {},
                     'nestedunet':      {},
                     'unet3plus':       {},
                     'deeplabv3plus':   {},
                     'deeplab-smp':     {},
                     'pranet':          {},
                     'segtran':         segtran_settings,
                     'refuge': {
                                 'num_classes': 3,
                                 'ds_class':    'SegCrop',
                                 # 'ds_names': 'train,valid,test',
                                 'orig_input_size': 576,
                                 # Each dim of the patch_size should always be multiply of 8.
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
                                 # 'bce_weight':  [0., 1],
                                 'ds_class':    'SegWhole',
                                 # 'ds_names': 'CVC-ClinicDB-train,Kvasir-train',
                                 'orig_input_size': 320,    # actual images are at various sizes. All resize to 320*320.
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
                                 # 'bce_weight':  [0., 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 'ds_class':    'SegWhole',
                                 'ds_names':    'duke',
                                 # Actual images are at various sizes. As the dataset is SegWhole, orig_input_size is ignored.
                                 # But output_upscale is computed as the ratio between orig_input_size and patch_size.
                                 # If you want to avoid output upscaling, set orig_input_size to the same as patch_size.
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

ds_stats_map = { 'refuge': 'refuge-cropped-gray{:.1f}-stats.json',
                 'polyp':  'polyp-whole-gray{:.1f}-stats.json',
                 'oct':    'oct-whole-gray{:.1f}-stats.json' }

stats_file_tmpl = ds_stats_map[args.task_name]
stats_filename = stats_file_tmpl.format(args.gray_alpha)
ds_stats = json.load(open(stats_filename))
default_settings[args.task_name].update(ds_stats)
print("'{}' mean/std loaded from '{}'".format(args.task_name, stats_filename))

get_default(args, 'mean',           default_settings, None,   [args.task_name, 'mean', args.ds_name])
get_default(args, 'std',            default_settings, None,   [args.task_name, 'std',  args.ds_name])
get_default(args, 'num_classes',    default_settings, None,   [args.task_name, 'num_classes'])
args.binarize = (args.num_classes == 2)
get_default(args, 'ds_class',       default_settings, None,   [args.task_name, 'ds_class'])

DataSetClass = dataloaders.datasets2d.__dict__[args.ds_class]

# Images after augmentation/transformation should keep their original size model_input_size.
# Will be resized before fed into the model.
tgt_width, tgt_height = args.orig_input_size

common_aug_func     = iaa.Sequential([
                            iaa.Resize({'height': tgt_height, 'width': tgt_width}),
                            iaa.Grayscale(alpha=args.gray_alpha)
                      ])
image_trans_func    = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(args.mean, args.std)
                      ])
segmap_trans_func   = None
'''
 transforms.Compose([
                          transforms.Lambda(lambda mask: reshape_mask(mask, 0, 255, shape=args.reshape_mask_type)),
                          transforms.ToTensor()
                      ])
'''

ds_settings     = default_settings[args.task_name]
if 'uncropped_size' in ds_settings:
    uncropped_size  = ds_settings['uncropped_size'][args.ds_name]
else:
    uncropped_size  = -1

if uncropped_size == -1 and 'orig_dir' in ds_settings:
    orig_dir  = ds_settings['orig_dir'][args.ds_name]
    orig_dir  = os.path.join(test_data_parent, orig_dir)
    orig_ext  = ds_settings['orig_ext'][args.ds_name]
else:
    orig_dir = orig_ext = None

has_mask        = ds_settings['has_mask'][args.ds_name]

db_test = DataSetClass(base_dir=test_data_path,
                       split=args.ds_split,
                       mode='test',
                       mask_num_classes=args.num_classes,
                       has_mask=has_mask,
                       common_aug_func=common_aug_func,
                       image_trans_func=image_trans_func,
                       segmap_trans_func=segmap_trans_func,
                       binarize=args.binarize,
                       train_loc_prob=0,
                       chosen_size=args.orig_input_size[0],
                       uncropped_size=uncropped_size,
                       orig_dir=orig_dir,
                       orig_ext=orig_ext)

args.num_workers = 0 if args.debug else 4
testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)
# num_modalities is used in segtran.
# num_modalities = 0 means there's not the modality dimension
# (but still a single modality) in the images loaded from db_train.
args.num_modalities = 0
if args.translayer_compress_ratios is not None:
    args.translayer_compress_ratios = [ float(r) for r in args.translayer_compress_ratios.split(",") ]
else:
    args.translayer_compress_ratios = [ 1 for layer in range(args.num_translayers+1) ]

def load_model(net, args, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=torch.device(args.device))
    params = net.state_dict()
    if 'model' in state_dict:
        model_state_dict = state_dict['model']
        cp_args          = state_dict['args']
        cp_iter_num      = state_dict['iter_num']
    else:
        model_state_dict = state_dict
        cp_args          = None
        cp_iter_num      = 0

    ignored_keys = [ 'maxiter', 'checkpoint_path', 'model_input_size', 't_total', 'num_workers',
                     'lr_warmup_ratio', 'lr_warmup_steps', 'local_rank', 'distributed', 'world_size',
                     'saveiter', 'dice_warmup_steps', 'opt', 'lr', 'decay',
                     'initializer_range', 'base_initializer_range',
                     'grad_clip', 'localization_prob', 'tune_bn_only', 'MAX_DICE_W', 'deterministic',
                     'lr_schedule', 'out_fpn_do_dropout', 'randscale', 'do_affine', 'focus_class',
                     'bce_weight', 
                     'seed', 'debug', 'ds_name', 'batch_size', 'dropout_prob',
                     'patch_size', 'orig_input_size', 'output_upscale',
                     'checkpoint_dir', 'iters', 'out_origsize', 'out_softscores', 'verbose_output',
                     'gpu', 'test_interp', 'do_remove_frag', 'reload_mask', 'ds_split', 'ds_names',
                     'job_name', 'mean', 'std', 'mask_thres', ]

    warn_keys = [ 'num_recurrences' ]

    # Some old models don't have these keys in args. But they use the values specified here.
    old_default_keys = { 'num_recurrences': 1 }
    args2 = copy.copy(args)

    if args.net == 'segtran' and cp_args is not None:
        for k in old_default_keys:
            if k not in args:
                args2.__dict__[k] = old_default_keys[k]

        for k in cp_args:
            if (k in warn_keys) and (args2.__dict__[k] != cp_args[k]):
                print("args[{}]={}, checkpoint args[{}]={}, inconsistent!".format(k, args2.__dict__[k], k, cp_args[k]))
                continue

            if (k not in ignored_keys) and (args2.__dict__[k] != cp_args[k]):
                print("args[{}]={}, checkpoint args[{}]={}, inconsistent!".format(k, args2.__dict__[k], k, cp_args[k]))
                exit(0)

    params.update(model_state_dict)
    net.load_state_dict(params)

    print("Model loaded from '{}'".format(checkpoint_path))

def test_calculate_metric(iter_nums):
    if args.net == 'unet':
        # timm-efficientnet performs slightly worse.
        if not args.vis_mode:
            backbone_type = re.sub("^eff", "efficientnet", args.backbone_type)
            net = smp.Unet(backbone_type, classes=args.num_classes, encoder_weights='imagenet')
        else:
            net = VanillaUNet(n_channels=3, n_classes=args.num_classes)
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
        get_default(args, 'num_modes',  default_settings, -1,   [args.net, 'num_modes', args.in_fpn_layers])
        set_segtran2d_config(args)
        print(args)
        net = Segtran2d(config)
    else:
        breakpoint()

    net.cuda()
    net.eval()

    # Currently colormap is used only for OCT task.
    colormap = get_seg_colormap(args.num_classes, return_torch=True).cuda()

    # prepred: pre-prediction. postpred: post-prediction.
    task2mask_prepred   = { 'refuge': refuge_map_mask,      'polyp': polyp_map_mask,
                            'oct': partial(index_to_onehot, num_classes=args.num_classes) }
    task2mask_postpred  = { 'refuge': refuge_inv_map_mask,  'polyp': polyp_inv_map_mask,
                            'oct': partial(onehot_inv_map, colormap=colormap) }
    mask_prepred_mapping_func   =   task2mask_prepred[args.task_name]
    mask_postpred_mapping_funcs = [ task2mask_postpred[args.task_name] ]
    if args.do_remove_frag:
        remove_frag = lambda segmap: remove_fragmentary_segs(segmap, 255)
        mask_postpred_mapping_funcs.append(remove_frag)

    if not args.checkpoint_dir:
        if args.vis_mode is not None:
            visualize_model(net, args.vis_mode, db_test)
            return

        if args.eval_robustness:
            eval_robustness(net, testloader, args.aug_degree)
            return

    allcls_avg_metric = None
    for iter_num in iter_nums:
        if args.checkpoint_dir:
            checkpoint_path = os.path.join(args.checkpoint_dir, 'iter_' + str(iter_num) + '.pth')
            load_model(net, args, checkpoint_path)

            if args.vis_mode is not None:
                visualize_model(net, args.vis_mode)
                continue

            if args.eval_robustness:
                eval_robustness(net, testloader, args.aug_degree)
                continue

        save_result = not args.test_interp
        if save_result:
            test_save_paths = []
            test_save_dirs  = []
            test_save_dir_tmpl  = "%s-%s-%s-%d" %(args.net, args.job_name, timestamp, iter_num)
            for suffix in ("-soft", "-%.1f" %args.mask_thres):
                test_save_dir = test_save_dir_tmpl + suffix
                test_save_path = "../prediction/%s" %(test_save_dir)
                if not os.path.exists(test_save_path):
                    os.makedirs(test_save_path)
                test_save_dirs.append(test_save_dir)
                test_save_paths.append(test_save_path)
        else:
            test_save_paths = None
            test_save_dirs  = None

        allcls_avg_metric, allcls_metric_count = \
                test_all_cases(net, testloader, task_name=args.task_name,
                               num_classes=args.num_classes,
                               mask_thres=args.mask_thres,
                               model_type=args.net,
                               orig_input_size=args.orig_input_size,
                               patch_size=args.patch_size,
                               stride=(args.orig_input_size[0] // 2, args.orig_input_size[1] // 2),
                               test_save_paths=test_save_paths,
                               out_origsize=args.out_origsize,
                               mask_prepred_mapping_func=mask_prepred_mapping_func,
                               mask_postpred_mapping_funcs=mask_postpred_mapping_funcs,
                               reload_mask=args.reload_mask,
                               test_interp=args.test_interp,
                               verbose=args.verbose_output)

        print("Iter-%d scores on %d images:" %(iter_num, allcls_metric_count[0]))
        dice_sum = 0
        for cls in range(1, args.num_classes):
            dice = allcls_avg_metric[cls-1]
            print('class %d: dice = %.3f' %(cls, dice))
            dice_sum += dice
        avg_dice = dice_sum / (args.num_classes - 1)
        print("Average dice: %.3f" %avg_dice)

        if args.net == 'segtran':
            max_attn, avg_attn, clamp_count, call_count = \
                [ segtran_shared.__dict__[v] for v in ('max_attn', 'avg_attn', 'clamp_count', 'call_count') ]
            print("max_attn={:.2f}, avg_attn={:.2f}, clamp_count={}, call_count={}".format(
                  max_attn, avg_attn, clamp_count, call_count))

        if save_result:
            FNULL = open(os.devnull, 'w')
            for pred_type, test_save_dir, test_save_path in zip(('soft', 'hard'), test_save_dirs, test_save_paths):
                do_tar = subprocess.run(["tar", "cvf", "%s.tar" %test_save_dir, test_save_dir], cwd="../prediction",
                                        stdout=FNULL, stderr=subprocess.STDOUT)
                # print(do_tar)
                print("{} tarball:\n{}.tar".format(pred_type, os.path.abspath(test_save_path)))

    return allcls_avg_metric


if __name__ == '__main__':
    iter_nums = [ int(i) for i in args.iters.split(",") ]
    if args.test_interp is not None:
        args.test_interp = [ int(i) for i in args.test_interp.split(",") ]

    args.device = 'cuda'
    if args.vis_mode is not None:
        # Gradients are required for visualization.
        allcls_avg_metric = test_calculate_metric(iter_nums)
    else:
        with torch.no_grad():
            allcls_avg_metric = test_calculate_metric(iter_nums)
