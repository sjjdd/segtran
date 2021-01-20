import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import os
from matplotlib import cm
from receptivefield.pytorch import PytorchReceptiveField
from receptivefield.image import get_default_image
import receptivefield
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

def get_seg_colormap(num_classes, return_torch):
    N = num_classes
    jet_colormap = cm.get_cmap('jet')
    seg_colormap = [ jet_colormap(i) for i in range(0, int(256/(N-1))*N, int(256/(N-1))) ]
    for i, color in enumerate(seg_colormap):
        seg_colormap[i] = [ int(256*frac) for frac in color[:3] ]

    if return_torch:
        seg_colormap = torch.tensor(seg_colormap, dtype=torch.uint8)
    else:
        seg_colormap = np.array(seg_colormap, dtype=np.uint8)
        
    return seg_colormap
    
def get_default(args, var_name, default_settings, default_value, key_list):
    # var_name has been specified a custom value in command line. So not to use default value instead.
    if (var_name in args) and (args.__dict__[var_name] != default_value):
        return
    v = default_settings
    for k in key_list:
        v = v[k]
    args.__dict__[var_name] = v

def get_filename(file_path):
    filename = os.path.normpath(file_path).lstrip(os.path.sep).split(os.path.sep)[-1]
    return filename
    
class AverageMeters(object):
    """Computes and stores the average and current values of given keys"""
    def __init__(self):
        self.total_reset()

    def total_reset(self):
        self.val   = {'total': {}, 'disp': {}}
        self.avg   = {'total': {}, 'disp': {}}
        self.sum   = {'total': {}, 'disp': {}}
        self.count = {'total': {}, 'disp': {}}

    def disp_reset(self):
        self.val['disp']   = {}
        self.avg['disp']   = {}
        self.sum['disp']   = {}
        self.count['disp'] = {}

    def update(self, key, val, n=1, is_val_avg=True):
        if type(val) == torch.Tensor:
            val = val.item()
        if type(n) == torch.Tensor:
            n = n.item()
        
        if np.isnan(val):
            pdb.set_trace()
            
        for sig in ('total', 'disp'):
            self.val[sig][key]    = val
            self.sum[sig].setdefault(key, 0)
            self.count[sig].setdefault(key, 0.0001)
            self.avg[sig].setdefault(key, 0)
            if is_val_avg:
                self.sum[sig][key] += val * n
            else:
                self.sum[sig][key] += val
                
            self.count[sig][key] += n
            self.avg[sig][key]    = self.sum[sig][key] / self.count[sig][key]


# Replace BatchNorm with GroupNorm
# https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/104686
def bn2gn(model, old_layer_type, new_layer_type, num_groups, convert_weights):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = bn2gn(module, old_layer_type, new_layer_type, num_groups, convert_weights)

        # single module
        if type(module) == old_layer_type:
            old_layer = module
            new_layer = new_layer_type(num_groups, module.num_features, module.eps, module.affine) 
            if convert_weights:
                new_layer.weight = old_layer.weight
                new_layer.bias = old_layer.bias

            model._modules[name] = new_layer

    return model

def remove_norms(model, name):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = remove_norms(module, name)

        if type(module) == nn.LayerNorm or \
           type(module) == nn.BatchNorm2d or \
           type(module) == nn.GroupNorm:
            do_nothing = nn.Identity()
            model._modules[name] = do_nothing

    return model

def visualize_model(net, vis_mode, dataset=None):
    remove_norms(net, 'net')
    input_shape = [1152, 1152, 3]
    rf = PytorchReceptiveField(lambda: net)
    if dataset is None:
        probe_image = get_default_image(input_shape, name='cat')
        probe_mask  = None
    else:
        probe_image_dict = dataset[24]
        probe_image = probe_image_dict['image']
        probe_mask  = probe_image_dict['mask']
        test_patch = F.interpolate(test_patch, size=patch_size,
                                   mode='bilinear', align_corners=False)
                                       
    rf_params = rf.compute(input_shape = input_shape)
    
    '''
    # plot receptive fields
    rf.plot_rf_grids(
        custom_image=cat_image, 
        figsize=(20, 12), 
        layout=(1, 3))
    '''
    for i in range(net.num_vis_layers):
        rf.plot_gradient_at(fm_id=i, point=(16, 16), image=probe_image, figsize=(7, 7))
    plt.show()

def pearson(t1, t2, dim=-1):
    assert t1.shape == t2.shape
    if dim == -1:
        t1flat = t1.flatten()
        t2flat = t2.flatten()
        t1flatz = t1flat - t1flat.mean()
        t2flatz = t2flat - t2flat.mean()
        norm1 = (t1flatz**2).float().sum().sqrt()
        norm2 = (t2flatz**2).float().sum().sqrt()
        norm1[norm1 < 1e-5] = 1
        norm2[norm2 < 1e-5] = 1

        corr = (t1flatz * t2flatz).float().sum() / (norm1 * norm2)
        return corr.item()

def lr_pearson(t1):
    left, right = t1.chunk(2, dim=-1)
    return pearson(left, right)

def batch_norm(t4d, debug=False):
    chan_num = t4d.shape[1]
    t4d_chanfirst = t4d.transpose(0, 1)
    t4d_flat = t4d_chanfirst.reshape(chan_num, -1)
    stds  = t4d_flat.std(dim=1)
    means = t4d_flat.mean(dim=1)
    t4d_normed = (t4d_flat - means.view(chan_num, 1)) / stds.view(chan_num, 1)
    t4d_normed = t4d_normed.reshape(t4d_chanfirst.shape).transpose(0, 1)
    return t4d_normed
        
def eval_robustness(net, dataloader, aug_degree=0.5):
    AUG_DEG = (aug_degree, aug_degree) # The bigger, the higher degree of aug is applied.
    augs = [
        transforms.ColorJitter(brightness=AUG_DEG),
        transforms.ColorJitter(contrast=AUG_DEG), 
        transforms.ColorJitter(saturation=AUG_DEG),
        transforms.Resize((192, 192)),
        transforms.Resize((432, 432)),
        transforms.Pad(0)   # Placeholder. Replace input images with random noises.
    ]
    is_resize = [ False, False, False, True, True, False ]
    
    num_augs = len(augs)
    num_iters = 24
    # on_pearsons: pearsons between old and new feature maps
    on_pearsons = np.zeros((num_augs, net.num_vis_layers))
    # lr_old_pearsons/lr_new_pearsons: pearsons between left-half and right-half of the feature maps
    lr_old_pearsons = np.zeros((net.num_vis_layers))
    old_stds        = np.zeros((net.num_vis_layers))
    lr_new_pearsons = np.zeros((num_augs, net.num_vis_layers))
    new_stds        = np.zeros((num_augs, net.num_vis_layers))
    aug_counts      = np.zeros(num_augs)
    print("Evaluating %d augs on %d layers of feature maps" %(num_augs, net.num_vis_layers))
    do_BN = True
    
    for it in tqdm(range(num_iters)):
        aug_idx = it % num_augs
        aug_counts[aug_idx] += 1
        aug = augs[aug_idx]
        dataloader.dataset.image_trans_func2 = transforms.Compose( [ aug ] + \
                                                                   dataloader.dataset.image_trans_func.transforms )

        batch = next(iter(dataloader))
        image_batch, image2_batch, mask_batch = batch['image'].cuda(), batch['image2'].cuda(), batch['mask'].cuda()
        if aug_idx == 5:
            image2_batch.normal_()
            
        with torch.no_grad():
            _ = net(image_batch)
            orig_features = copy.copy(net.feature_maps)
            orig_bn_features = list(orig_features)
            net.feature_maps = []
            _ = net(image2_batch)
            aug_features  = copy.copy(net.feature_maps)
            aug_bn_features  = list(aug_features)
            net.feature_maps = []
            for layer_idx in range(net.num_vis_layers):
                if is_resize[aug_idx] and orig_features[layer_idx].shape != aug_features[layer_idx].shape:
                    try:
                        aug_features[layer_idx] = F.interpolate(aug_features[layer_idx], size=orig_features[layer_idx].shape[-2:],
                                                                mode='bilinear', align_corners=False)
                    except:
                        breakpoint()
                
                if do_BN and orig_features[layer_idx].dim() == 4:
                    orig_bn_features[layer_idx] = batch_norm(orig_features[layer_idx])
                    aug_bn_features[layer_idx]  = batch_norm(aug_features[layer_idx])

                pear = pearson(orig_bn_features[layer_idx], aug_bn_features[layer_idx])
                on_pearsons[aug_idx, layer_idx]     += pear
                lr_old_pearsons[layer_idx] += lr_pearson(orig_bn_features[layer_idx])
                lr_new_pearsons[aug_idx, layer_idx] += lr_pearson(aug_bn_features[layer_idx])
                
                # 4D feature maps. Assume dim 1 is the channel dim.
                if orig_features[layer_idx].dim() == 4:
                    chan_num = orig_features[layer_idx].shape[1]
                    old_std  = orig_features[layer_idx].transpose(0, 1).reshape(chan_num, -1).std(dim=1).mean()
                    new_std  = aug_features[layer_idx].transpose(0, 1).reshape(chan_num, -1).std(dim=1).mean()
                else:
                    old_std  = orig_features[layer_idx].std()
                    new_std  = aug_features[layer_idx].std()
                old_stds[layer_idx] += old_std
                new_stds[aug_idx, layer_idx] += new_std
                    
    on_pearsons /= np.expand_dims(aug_counts, 1)
    lr_old_pearsons /= num_iters
    lr_new_pearsons /= np.expand_dims(aug_counts, 1)
    old_stds /= num_iters
    new_stds /= np.expand_dims(aug_counts, 1)
    
    for layer_idx in range(net.num_vis_layers):
        print("%d: Orig LR P %.3f, Std %.3f" %(layer_idx, lr_old_pearsons[layer_idx], old_stds[layer_idx]))

    for aug_idx in range(num_augs):
        print(augs[aug_idx])
        for layer_idx in range(net.num_vis_layers):
            print("%d: ON P %.3f, LR P %.3f, Std %.3f" %(layer_idx, 
                            on_pearsons[aug_idx, layer_idx], 
                            lr_new_pearsons[aug_idx, layer_idx], 
                            new_stds[aug_idx, layer_idx]))
            