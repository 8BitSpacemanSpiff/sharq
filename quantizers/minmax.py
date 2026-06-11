import torch
import torch.nn as nn
from utils.quant_utils import fake_quantize, grid_search


def get_min_max_range(weight, sym):
    tmp = torch.zeros((*weight.shape[:-1], 1), device=weight.device)
    w_min = torch.minimum(torch.amin(weight, dim=-1, keepdim=True), tmp)
    w_max = torch.maximum(torch.amax(weight, dim=-1, keepdim=True), tmp)

    if sym:
        w_max = torch.maximum(torch.abs(w_min), w_max)
        tmp = w_min < 0
        if torch.any(tmp):
            w_min[tmp] = -w_max[tmp]
    tmp = (w_min == 0) & (w_max == 0)
    w_min[tmp] = -1
    w_max[tmp] = +1

    return w_min, w_max


class MinMaxQuantizer(nn.Module):
    def __init__(self, shape=1):
        super(MinMaxQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
            self, n_bits, per_channel=False, group_size=-1, sym=True, 
            mse=False, norm=2.4, grid=100, maxshrink=1.0,
        ):
        self.n_bits = n_bits
        self.group_size = group_size
        self.maxq = torch.tensor(2 ** n_bits - 1)
        self.per_channel = per_channel
        if not self.per_channel:
            group_size = -1
        self.sym = sym
        
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink


    def find_params_H(self, weight, H_col, search):
        w_min, w_max = get_min_max_range(weight, self.sym)
        scale, zero = grid_search(weight, w_max, w_min, self.maxq, self.sym, self.norm, self.grid, H_col, search)
            
        return scale, zero
    
    
    def find_params(self, weight):
        dev = weight.device
        self.maxq = self.maxq.to(dev)

        shape = weight.shape
        if self.per_channel:
            weight = weight.flatten(1)
            if self.group_size != -1:
                assert weight.shape[-1] % self.group_size == 0
                n_groups = weight.shape[-1] // self.group_size
                weight = weight.view(-1, n_groups, self.group_size)
        else:
            weight = weight.flatten().unsqueeze(0)

        w_min, w_max = get_min_max_range(weight, self.sym)

        scale, zero = grid_search(weight, w_max, w_min, self.maxq, self.sym, self.norm, self.grid, None, self.mse)

        if not self.per_channel:
            tmp = shape[0]
            scale = scale.repeat(tmp, 1)
            zero = zero.repeat(tmp, 1)

        self.scale = scale
        self.zero = zero


    def quantize(self, x):
        assert self.ready(), "Quantization parameters (i.e., scale and zero) should be determined first!"
        
        return fake_quantize(x, self.scale, self.zero, self.maxq)


    def ready(self):
        return torch.all(self.scale != 0)