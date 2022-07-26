from einops import repeat, rearrange, reduce
import wandb
from collections import defaultdict
from typing import Union
import torch
import torch.nn as nn
import numpy as np
from utils.config import FLAGS
#from config import FLAGS

def make_divisible(v, divisor=8, min_value=1):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2)//divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 us=[True, True], ratio=[1, 1]):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
        self.ratio = ratio
        self.us = us

    def forward(self, input):
        if self.us[0]:
            self.in_channels = make_divisible(
                self.in_channels_max
                * self.width_mult
                / self.ratio[0]) * self.ratio[0]

        if self.us[1]:
            self.out_channels = make_divisible(
                self.out_channels_max
                * self.width_mult
                / self.ratio[1]) * self.ratio[1]
        
        y = nn.functional.conv2d(
            input, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)

        return y


class DynamicGroupConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 us=[True, True], ratio=[1, 1]):
        super(DynamicGroupConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
        self.ratio = ratio
        self.us = us

        self.density = None
        self.mask = None
        self.BS_R = int(FLAGS.BS_R)
        self.BS_C = int(FLAGS.BS_C)

        # HB-level
        #self.dense_in_channels = int(in_channels * 0.25)
        #self.dense_out_channels = int(out_channels * 0.25)
        #self.dense_mask = torch.zeros_like(self.weight).cuda()
        #self.dense_mask[:self.dense_out_channels, :self.dense_in_channels, :, :] = 1.

    def change_mask(self):
        if self.us[0]:
            self.in_channels = make_divisible(
                self.in_channels_max
                * self.width_mult
                / self.ratio[0]) * self.ratio[0]
        if self.us[1]:
            self.out_channels = make_divisible(
                self.out_channels_max
                * self.width_mult
                / self.ratio[1]) * self.ratio[1]

        if self.density == 1.0 or self.density == 0.0:  # Only Channel-level
            self.mask = None
        else:
            # HB-level
            block_weight = self.weight.data
            #block_weight[:self.dense_out_channels_max, :self.dense_in_channels_max:, :, :] = 0.
            block_weight_l2 = block_weight.reshape(self.out_channels_max//self.BS_R, self.BS_R, (self.in_channels_max * block_weight.size(2) * block_weight.size(3))//self.BS_C, self.BS_C, -1).pow(2).mean(dim=(1, 3, 4))


            #block_weight_l2 = self.weight.reshape(self.out_channels//self.BS_R, self.BS_R, self.in_channels//self.BS_C, self.BS_C, -1).pow(2).mean(dim=(1, 3, 4))
            q = 100 * (1-self.density)
            thresh_val = percentile(block_weight_l2, q)

            block_mask = (block_weight_l2 > thresh_val).type(torch.float)
            block_mask = torch.repeat_interleave(block_mask, self.BS_R, dim=0)
            block_mask = torch.repeat_interleave(block_mask, self.BS_C, dim=1)

            # HB-level
            #block_mask[:self.dense_out_channels_max, :self.dense_in_channels_max] = 1.

            ## 1x1 convolution
            #self.mask = block_mask.reshape(self.out_channels_max, self.in_channels_max, 1, 1)

            self.mask = block_mask.reshape(self.out_channels_max, self.in_channels_max, block_weight.size(2), block_weight.size(3))

    def forward(self, input):
        if self.mask is not None:
            weight = self.weight.cuda() * self.mask.cuda()
        else:
            weight = self.weight
        #print(f"{self}: {torch.sum((weight == 0.0).int()).item()/weight.numel()}")
        y = nn.functional.conv2d(
            input, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class DynamicGroupBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1):
        super(DynamicGroupBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=False)
        
        self.ratio = ratio
        self.width_mult = FLAGS.width_mult
        self.density = None
        self.ignore_model_profiling = True
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(self.num_features, affine=False) for i in FLAGS.density_list])

    def forward(self, input):

        if self.density in FLAGS.density_list:
            idx = FLAGS.density_list.index(self.density)
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean,
                self.bn[idx].running_var,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps)
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps)
        return y

def log_sparsity(m):
    if isinstance(m, DynamicGroupConv2d):
        if m.mask is not None:
            weight = m.weight.data * m.mask.data
            print(f"{m}={torch.sum((weight == 0.0).int()).item() / weight.numel()}")

def log_nnz(m):
    print("nnz log")
    if isinstance(m, DynamicGroupConv2d):
        if m.mask is not None:
            weight = m.weight.data * m.mask.data
            print(f"{m}: {torch.sum((weight == 0.0).int()).item()} ")

def log_weight_dist(model):
    param_list = []
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, DynamicGroupConv2d):
            param_list.append(layer.weight.view(-1).cpu().detach().numpy())

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.boxplot(param_list)
    plt.show()
    plt.savefig('local_pre-train.png')

def recored_sparsity(model, density, epoch):
    if density == 1.0:
        return

    log = defaultdict(float)
    idx = 1
    for layer in model.modules():
        if isinstance(layer, DynamicGroupConv2d):
            k_size = layer.weight.size(3)
            mask = layer.mask.data
            ratio = 100 * torch.sum((mask == 0.0).int()).item() / mask.numel()
            log[f"{density}-conv-{k_size}-{idx}"] = ratio
            idx += 1
    
    wandb.log(log, step=epoch)        

def change_in_mask(model, sparsity, epoch):
    if sparsity == 1.0:
        return

    if not hasattr(change_in_mask, "prev_mask"):
        change_in_mask.prev_mask = defaultdict(lambda : defaultdict(torch.tensor))
    
    idx = 1
    if not f"{sparsity}" in change_in_mask.prev_mask:
        for layer in model.modules():
            if isinstance(layer, DynamicGroupConv2d):
                k_size = layer.weight.size(3)
                mask = layer.mask.data
                change_in_mask.prev_mask[f"{sparsity}"][f"conv-{k_size}-{idx}"] = mask
                idx += 1
        return

    log = {}
    for layer in model.modules():
        if isinstance(layer, DynamicGroupConv2d):
            k_size = layer.weight.size(3)
            layer_name = f"conv-{k_size}-{idx}"
            mask = change_in_mask.prev_mask[f"{sparsity}"][layer_name]
            new_mask = layer.mask.data
            change_ratio = 100 * torch.sum((mask != new_mask).int()).item() / mask.numel()
            change_in_mask.prev_mask[f"{sparsity}"][layer_name] = new_mask
            log[f"change in mask at {sparsity}-{layer_name}"] = change_ratio
            idx += 1
    
    wandb.log(log, step=epoch)           
                
        
def conv_change_mask(m):
    if isinstance(m, DynamicGroupConv2d):
        m.change_mask()

def global_pruning_update(model, density):
    global_weight_l2 = torch.tensor([]).cuda()
    layer_info = []
    for layer in model.modules():
        if isinstance(layer, DynamicGroupConv2d):
            if density == 1.0:
                layer.mask = None
                continue

            block_weight = layer.weight.clone()
            block_weight_l2 = block_weight.reshape(layer.out_channels_max//layer.BS_R, layer.BS_R, (layer.in_channels_max * block_weight.size(2) * block_weight.size(3))//layer.BS_C, layer.BS_C, -1).pow(2).mean(dim=(1, 3, 4)).view(-1).cuda()
            global_weight_l2 = torch.cat([global_weight_l2, block_weight_l2])
            o, i, h, w = layer.weight.size()
            weight_shape = (o//layer.BS_R, i//layer.BS_C, h, w)
            layer_info.append([weight_shape, np.prod(block_weight_l2.size())])

    if density == 1.0: return

    total_size = 0

    for layer in layer_info:
        total_size += layer[1]

    global_threshold = percentile(global_weight_l2, 100 * (1 - density))
    global_block_mask = (global_weight_l2 > global_threshold).type(torch.float)
    layer_id = 0
    
    for layer in model.modules():
        if isinstance(layer, DynamicGroupConv2d):
            weight_shape, size = layer_info[layer_id]
            total_size -= size
            layer_block_mask, global_block_mask = torch.split(global_block_mask, [size, total_size])

            layer_block_mask = torch.reshape(layer_block_mask, weight_shape)
            layer_block_mask = torch.repeat_interleave(layer_block_mask, layer.BS_R, dim=0)
            layer_block_mask = torch.repeat_interleave(layer_block_mask, layer.BS_C, dim=1)
            layer.mask = layer_block_mask.reshape(layer.out_channels_max, layer.in_channels_max, weight_shape[2], weight_shape[3])
            layer_id += 1



def global_normal_pruning_update(model, density):
    global_normal_weight_l2 = torch.tensor([]).cuda()
    layer_info = []
    for layer in model.modules():
        if isinstance(layer, DynamicGroupConv2d):
            if density == 1.0:
                layer.mask = None
                continue

            block_weight = layer.weight.clone()
            block_weight_l2 = block_weight.reshape(layer.out_channels_max//layer.BS_R, layer.BS_R, layer.in_channels_max * block_weight.size(2) * block_weight.size(3)//layer.BS_C, layer.BS_C, -1).pow(2).mean(dim=(1, 3, 4)).view(-1).cuda()
            sorted_scores, sorted_idx = block_weight_l2.sort(descending=True)
            cumsum = torch.cumsum(sorted_scores, dim=0).cuda()
            block_normal_weight_l2 = torch.zeros(sorted_scores.shape).cuda()
            block_normal_weight_l2[sorted_idx] = sorted_scores / cumsum
            global_normal_weight_l2 = torch.cat([global_normal_weight_l2, block_normal_weight_l2])
            o, i, h, w = layer.weight.size()
            weight_shape = (o//layer.BS_R, i//layer.BS_C, h, w)
            layer_info.append([weight_shape, np.prod(block_normal_weight_l2.size())])

    if density == 1.0: return

    total_size = 0

    for layer in layer_info:
        total_size += layer[1]

    global_threshold = percentile(global_normal_weight_l2, 100 * (1 - density))
    global_block_mask = (global_normal_weight_l2 > global_threshold).type(torch.float)
    layer_id = 0
    
    for layer in model.modules():
        if isinstance(layer, DynamicGroupConv2d):
            weight_shape, size = layer_info[layer_id]
            total_size -= size
            layer_block_mask, global_block_mask = torch.split(global_block_mask, [size, total_size])

            layer_block_mask = torch.reshape(layer_block_mask, weight_shape)
            layer_block_mask = torch.repeat_interleave(layer_block_mask, layer.BS_R, dim=0)
            layer_block_mask = torch.repeat_interleave(layer_block_mask, layer.BS_C, dim=1)
            layer.mask = layer_block_mask.reshape(layer.out_channels_max, layer.in_channels_max, weight_shape[2], weight_shape[3])
            layer_id += 1

def bn_calibration_init(m):
    """ calculating post-statistics of batch normalization """
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        if getattr(FLAGS, 'cumulative_bn_stats', False):
            m.momentum = None
