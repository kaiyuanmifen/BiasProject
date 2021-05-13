# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

#eps = torch.finfo(torch.float32).eps # 1.1920928955078125e-07
#eps = 1e-20 # TODO : search for the smallest `eps` number under pytorch such as `torch.log(eps) != -inf`

def bias_classification_loss(q_c: Tensor, p_c: Tensor, weight = None, 
                                reduction : str = "mean", softmax = True, sigmoid = False) -> Tensor:
    assert reduction in ["mean", "sum", "none"]
    #assert torch.equal(torch.sum(p_c, dim = 1), torch.ones(bach_size, dtype=p_c.dtype))
    #assert torch.equal(torch.sum(q_c, dim = 1), torch.ones(bach_size, dtype=q_c.dtype))
    assert not (softmax and sigmoid)
    
    if weight is None :
        weight = torch.ones_like(p_c)
    else :
        if weight.dim() == 1 :
            assert list(weight.shape) == [p_c.size(1)]
            weight = weight.expand_as(p_c) 
        elif weight.dim() == 2 :
            assert weight.shape == p_c.shape
        else :
            raise RuntimeError("weight.shape incorrect")
    
    if softmax :
        # Multi-class approach
        CE = torch.sum(- weight * p_c * F.log_softmax(q_c, dim = 1), dim = 1) # batch_size
    elif sigmoid :
        # Multi-label approach
        #CE = torch.sum(- weight * p_c * F.logsigmoid(q_c), dim = 1) # batch_size
        CE = torch.sum(- weight * (p_c * F.logsigmoid(q_c) + (1-p_c) * torch.log(1 - torch.sigmoid(q_c))), dim = 1) # batch_size
    else :
        CE = torch.sum(- weight * p_c * torch.log(q_c + torch.finfo(q_c.dtype).eps), dim = 1) # batch_size
    if reduction == "none" :
        return CE
    elif reduction == "mean" :
        return torch.mean(CE) # or CE.mean()
    elif reduction == "sum" :
        return torch.sum(CE) # or CE.sum()

def bce_bias_classification_loss(q_c: Tensor, p_c: Tensor, weight = None, 
                                    reduction : str = "mean") -> Tensor :
    return bias_classification_loss(q_c, p_c, weight, reduction, softmax = False, sigmoid = True)

class BiasClassificationLoss(nn.Module):
    def __init__(self, weight = None, reduction: str = 'mean', softmax = False) -> None:
        super(BiasClassificationLoss, self).__init__()
        assert reduction in ["mean", "sum", "none"]
        self.weight = weight
        self.reduction = reduction
        self.softmax = softmax
    
    def forward(self, q_c: Tensor, p_c: Tensor) -> Tensor:
        """assume p_c, q_c is (batch_size, num_of_classes)"""
        return bias_classification_loss(q_c, p_c, self.weight, self.reduction, self.softmax)
    
def kl_divergence_loss(logits, target, weight=None, softmax = False) :
    # https://discuss.pytorch.org/t/kl-divergence-loss/65393/4?u=pascal_notsawo
    kl_loss = F.kl_div(F.log_softmax(logits, dim = 1), F.softmax(target, dim = 1) if softmax else target, reduction="none").mean()
    l1_loss = 0 # F.l1_loss(F.softmax(logits, dim = 1), target)
    l2_loss = 0 # F.mse_loss(F.softmax(logits, dim = 1), target)
    return kl_loss + l1_loss + l2_loss

def nll_loss(logits, target, weight=None):
    return F.nll_loss(F.log_softmax(logits), target, weight=weight)

def gaussian_nll_loss(logits, target, weight=None) :
    # https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
    bs, n_class = target.shape
    #var = torch.ones(bs, n_class, requires_grad=True).to(logits.device) #heteroscedastic
    var = torch.ones(bs, 1, requires_grad=True).to(logits.device) #homoscedastic
    return F.gaussian_nll_loss(input = F.softmax(logits, dim=1), target = target, var = var, full=False, eps=1e-06, reduction='mean')

def bp_mll_loss(c: Tensor, y: Tensor, bias=(1, 1), weight=None) -> Tensor:
    r"""compute the loss, which has the form:
        L = \sum_{i=1}^{m} \frac{1}{|Y_i| \cdot |\bar{Y}_i|} \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
    :param c: prediction tensor, size: batch_size * n_labels
    :param y: target tensor, size: batch_size * n_labels
    :return: size: scalar tensor
    
    Multi-Label Neural Networks with Applications to Functional Genomics and Text Categorization
    Min-Ling Zhang and Zhi-Hua Zhou, Senior Member, IEEE, 2016
    
    https://github.com/idocx/BP_MLL_Pytorch/blob/master/bp_mll.py
    """
    assert len(bias) == 2 and all(map(lambda x: isinstance(x, int) and x > 0, bias)), "bias must be positive integers"
    
    y = y.float()
    y_bar = -y + 1
    y_norm = torch.pow(y.sum(dim=(1,)), bias[0])
    y_bar_norm = torch.pow(y_bar.sum(dim=(1,)), bias[1])
    assert torch.all(y_norm != 0) or torch.all(y_bar_norm != 0), "an instance cannot have none or all the labels"
    return torch.mean(1 / torch.mul(y_norm, y_bar_norm) * pairwise_sub_exp(y, y_bar, c))

def pairwise_sub_exp(y: Tensor, y_bar: Tensor, c: Tensor) -> Tensor:
    r"""compute \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
    
    https://github.com/idocx/BP_MLL_Pytorch/blob/master/bp_mll.py"""
    truth_matrix = y.unsqueeze(2).float() @ y_bar.unsqueeze(1).float()
    exp_matrix = torch.exp(c.unsqueeze(1) - c.unsqueeze(2))
    return (torch.mul(truth_matrix, exp_matrix)).sum(dim=(1, 2))

def hamming_loss(c: Tensor, y: Tensor, threshold=0.8) -> Tensor:
    """compute the hamming loss (refer to the origin paper)
    :param c: size: batch_size * n_labels, output of NN
    :param y: size: batch_size * n_labels, target
    :return: Scalar
    
    https://github.com/idocx/BP_MLL_Pytorch/blob/master/bp_mll.py
    """
    assert 0 <= threshold <= 1, "threshold should be between 0 and 1"
    p, q = c.size()
    return 1.0 / (p * q) * (((c > threshold).int() - y) != 0).float().sum()

def one_errors(c: Tensor, y: Tensor) -> Tensor:
    """compute the one-error function
    https://github.com/idocx/BP_MLL_Pytorch/blob/master/bp_mll.py"""
    p, _ = c.size()
    return (y[0, torch.argmax(c, dim=1)] != 1).float().sum() / p