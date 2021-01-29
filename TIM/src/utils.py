# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import ModuleList
import torch.nn.functional as F

import copy

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu  
    elif activation == "prelu" :
        a = 0.01
        def f(x):
            # return torch.relu(x) + a * (x - abs(x))*0.5
            return F.prelu(input = x, weight = torch.tensor(a))
        return f
    elif activation == "elu" :
        a = 1.
        def f(x):
            return F.elu(input = x, alpha = torch.tensor(a))
        return f

    raise RuntimeError("activation should be relu/gelu/prelu/elu, not {}".format(activation))



def rand(shape : tuple, a = 0, b = 1., random_seed = 0) :
    torch.manual_seed(random_seed)
    return (b - a) * torch.rand(shape) + b 