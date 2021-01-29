# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
from typing import Optional, Any, Tuple # Callable, List, Optional
import warnings

import math

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, MultiheadAttention, ModuleList, Dropout, Linear, LayerNorm, Sequential
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence

from .utils import _get_clones, _get_activation_fn


def Attention(query, key, value, dropout_p=0.1, training : bool = True, key_padding_mask = None, attn_mask = None, 
    need_weights : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Scale dot product attention
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        training: apply dropout if is ``True``.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        Inputs:
        - query: :math:`(target_sequence_length, batch_size, d_q)`
        - key: :math:`(source_sequence_length, batch_size, d_k)`
        - value: :math:`(source_sequence_length, batch_size, d_v)` 
        - key_padding_mask: :math:`(batch_size, source_sequence_length)`
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(target_sequence_length, source_sequence_length)` 
          3D mask :math:`(batch_size*num_heads, target_sequence_length, source_sequence_length)`. 
          attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        
        Outputs:
        - attn_output: :math:`(target_sequence_length, batch_size, d_v)`
        - attn_output_weights: :math:`(batch_size, target_sequence_length, source_sequence_length)`
    """
    num_heads = 1
    tgt_len, bsz, d_q = query.size()
    src_len, _, d_k = key.size()
    assert d_k == d_q
    assert src_len == value.size(0) 
    assert key.size(1) == value.size(1) == bsz
    d_v = value.size(2)

    scaling = float(d_k) ** -0.5
    attn_weights = torch.bmm(query*scaling, key.transpose(1, 2))
    assert list(attn_weights.size()) == [tgt_len, bsz, src_len]

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, tgt_len, src_len]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, tgt_len, src_len]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
            #raise NotImplementedError("3D")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

        if attn_mask.dtype == torch.bool:
            attn_weights.masked_fill_(attn_mask, float("-inf"))
        elif mask == 0 :
            attn_weights.masked_fill_(attn_mask == 0, float("-inf"))
        else:
            #attn_weights += attn_mask
            raise NotImplementedError("")

    if key_padding_mask is not None:
        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

        attn_weights = attn_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_weights = attn_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_weights = F.softmax(attn_weights, dim=-1) 
    attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_weights, value)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, d_v]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, d_v)
    
    if need_weights:
        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


def GroupLinear(inputs : Tensor, layers : ModuleList, activation = None) -> Tensor :
    """
    Args:
        inputs  : (ns, d, batch_size, d_in) 
        layers : (ns, d_in, d_out)
        activation : ...
    """
    assert len(layers) == len(inputs)
    if activation is not None:
        return [activation(layer(x)) for layer, x in zip(layers, inputs)]
    else :
        return [layer(x) for layer, x in zip(layers, inputs)]

class MechanismCompetitionLayer(Module):
    """Compute Mechanism Competition"""
    def __init__(self, d_mech, n_s, bias = True) :
        """
        d_mech : single mechanism dimension
        n_s : number of mechanisms
        """
        super(MechanismCompetitionLayer, self).__init__()
        W_c = nn.Linear(in_features = d_mech, out_features = 1, bias = bias)
        self.W_c = _get_clones(module = W_c, N = n_s)
        self.activation = lambda x : F.softmax(input = x, dim = 1)

    def forward(self, h : Tensor) -> Tensor:
        #return torch.stack(GroupLinear(inputs = h, layers = self.W_c, activation=self.activation)) # (ns, seq_len, batch_size, 1)
        return GroupLinear(inputs = h, layers = self.W_c, activation = self.activation) # (ns, seq_len, batch_size, 1)

class MHA(MultiheadAttention):
    """TODO"""
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MHA, self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
        # in lieu of : self.q_proj_weight = nn.parameter.Parameter (torch.Tensor(embed_dim, embed_dim))
        self.q_proj_weight = nn.parameter.Parameter(torch.Tensor(embed_dim, self.kdim))
       
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None) :
        super(MHA, self).forward( query, key, value, key_padding_mask, need_weights, attn_mask)

class MechanismWiseSelfAttentionSubLayer(Module):
    def __init__(self, d_mech, H, n_s, d_k, d_v, dropout=0., bias = True, add_bias_kv=False, add_zero_attn=False, debug = True) :
        super(MechanismWiseSelfAttentionSubLayer, self).__init__()
        self.debug = debug
        if debug :
            W2_Q = nn.Linear(in_features = d_mech, out_features = H*d_k, bias = bias)
            self.W2_Q = _get_clones(module = W2_Q, N = n_s)
            self.W2_K = _get_clones(module = W2_Q, N = n_s)
            W2_V = nn.Linear(in_features = d_mech, out_features = H*d_v, bias = bias)
            self.W2_V = _get_clones(module = W2_V, N = n_s)
            W2_0 = nn.Linear(in_features = H*d_v, out_features = d_mech, bias = bias)
            self.W2_0 = _get_clones(module = W2_0, N = n_s)
        else :
            attn = MultiheadAttention(
            #attn = MHA(
                  embed_dim=d_mech, num_heads=H, dropout=dropout, bias=bias, 
                                      add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, 
                                      kdim=H*d_k, vdim=H*d_v
                       )
            self.attn = _get_clones(module = attn, N = n_s)

        self.norm = LayerNorm(d_mech)
    
    def forward(self, h : Tensor, c : Tensor, key_padding_mask=None, need_weights=True, attn_mask=None) -> Tensor:
        """
        h : (n_s, seq_len, batch_size, d_mech)
        c : (n_s, seq_len, batch_size, 1)
        """
        if self.debug :
            Q = GroupLinear(h, self.W2_Q) # (n_s, seq_len, batch_size, H*d_k)
            K = GroupLinear(h, self.W2_K) # (n_s, seq_len, batch_size, H*d_k)
            V = GroupLinear(h, self.W2_V) # (n_s, seq_len, batch_size, H*d_v)
            M = [Attention(query = q, key = k, value = v, key_padding_mask =  key_padding_mask, attn_mask = attn_mask) 
                for q, k, v in zip(Q, K, V)] # (n_s, seq_len, batch_size, H*d_v)
            M = GroupLinear(M, self.W2_0) # (n_s, seq_len, batch_size, d_mech)
        else :
            #M = GroupLinear(inputs = h, layers = self.attn) # (n_s, seq_len, batch_size, H*d_v)
            M = [
                layer(query = x, key = x, value = x, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)[0] 
                for layer, x in zip(self.attn, h) ] # (n_s, seq_len, batch_size, H*d_v)

        h = [self.norm(h_i + c_i * M_i) for h_i, c_i, M_i in zip(h, c, M)] # (n_s, seq_len, batch_size, d_mech)
        #h = torch.stack(h)
        return h
        

class InterMechanismAttentionSubLayer(Module):
    """"""
    def __init__(self, d_mech, H_c, n_s, d_k, d_v, dropout=0., bias = True, add_bias_kv=False, add_zero_attn=False, debug = True) :
        """"""
        super(InterMechanismAttentionSubLayer, self).__init__()
        self.debug = debug
        if debug :
            W3_Q = nn.Linear(in_features = d_mech, out_features = H_c*d_k, bias = bias)
            self.W3_Q = _get_clones(module = W3_Q, N = n_s)
            self.W3_K = _get_clones(module = W3_Q, N = n_s)
            W3_V = nn.Linear(in_features = d_mech, out_features = H_c*d_v, bias = bias)
            self.W3_V = _get_clones(module = W3_V, N = n_s)
            W3_0 = nn.Linear(in_features = H_c*d_v, out_features = d_mech, bias = bias)
            self.W3_0 = _get_clones(module = W3_0, N = n_s)
        else :
            attn = MultiheadAttention(embed_dim=d_mech, num_heads=H_c, dropout=dropout, bias=bias, 
                                      add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, 
                                      kdim=H_c*d_k, vdim=H_c*d_v)
            self.attn = _get_clones(module = attn, N = n_s)

        self.norm = LayerNorm(d_mech)

    def forward(self, h : Tensor, key_padding_mask=None, need_weights=True, attn_mask=None) -> Tensor:
        """
        """
        if self.debug :
            Q = GroupLinear(h, self.W3_Q) # (n_s, seq_len, batch_size, H_c*d_k)
            K = GroupLinear(h, self.W3_K) # (n_s, seq_len, batch_size, H_c*d_k)
            V = GroupLinear(h, self.W3_V) # (n_s, seq_len, batch_size, H_c*d_v)
            #Q = torch.stack(Q).view((self.n_s, seq_len*batch_size, self.H_c*self.d_k)) # (n_s, seq_len * batch_size, H_c*d_k)
            #K = torch.stack(K).view((self.n_s, seq_len*batch_size, self.H_c*self.d_k)) # (n_s, seq_len * batch_size, H_c*d_k)
            #V = torch.stack(V).view((self.n_s, seq_len*batch_size, self.H_c*self.d_v)) # (n_s, seq_len * batch_size, H_c*d_v)
            M = [Attention(query = q, key = k, value = v, key_padding_mask =  key_padding_mask, attn_mask) 
                for q, k, v in zip(Q, K, V)] # (n_s, seq_len * batch_size, H_c*d_v)
            M = GroupLinear(M, self.W3_0) # (n_s, seq_len, batch_size, d_mech)
        else :
            #M = GroupLinear(inputs = h, layers = self.attn) # (n_s, seq_len, batch_size, H_c*d_v)
            M = [
                layer(query = x, key = x, value = x, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)[0] 
                for layer, x in zip(self.attn, h) ] # (n_s, seq_len, batch_size, H_c*d_v)

        h = [self.norm(h_i + M_i) for h_i, M_i in zip(h, M)] # (n_s, seq_len, batch_size, d_mech)
        #h = torch.stack(h)
        return h

class MechanismWisePositionWiseFFNSubLayer(Module):
    def __init__(self, d_mech, n_s, dffn_m, bias = True) :
      
        super(MechanismWisePositionWiseFFNSubLayer, self).__init__()
        W_1 = nn.Linear(in_features = d_mech, out_features = dffn_m, bias = bias)
        self.W_1 = _get_clones(module = W_1, N = n_s)

        W_2 = nn.Linear(in_features = dffn_m, out_features = d_mech, bias = bias)
        self.W_2 = _get_clones(module = W_2, N = n_s)

        self.norm = LayerNorm(d_mech)

        self.activation = torch.sigmoid
    
    def forward(self, h):
        out = GroupLinear(h, self.W_1, activation=self.activation) # (n_s, seq_len, batch_size, dffn_m)
        out = GroupLinear(out, self.W_2) # (n_s, seq_len, batch_size, d_mech)
        h = [self.norm(h_i + F_i) for h_i, F_i in zip(h, out)] # (n_s, seq_len, batch_size, d_mech)
        #h = torch.stack(h)
        return h

class TIM_Encoder_Layer(Module):
    def __init__(self, d_model, d_ffn, n_s, d_k, d_v, H, H_c, dropout=0.1, activation="relu", bias = True,
                 add_bias_kv=False, add_zero_attn=False) :
        """
        n_s : number of mechanisms
        d_k : key size
        d_v : value size
        H : number of heads for self-attention 
        H_c : number of heads for inter-mechanism attention
        """
        super(TIM_Encoder_Layer, self).__init__()

        assert d_model % n_s == 0
        assert d_ffn % n_s == 0

        self.n_s = n_s
        self.d_mech = d_model // n_s
        dffn_m = d_ffn // n_s 
        
        # Mechanism Competition sub-layer
        self.competitionSubLayer = MechanismCompetitionLayer(self.d_mech, n_s, bias = bias) 
        self.dropout1 = Dropout(dropout)
        # Mechanism-wise self-attention sub-layer
        self.attn1 = MechanismWiseSelfAttentionSubLayer(self.d_mech, H, n_s, d_k, d_v, dropout, bias, add_bias_kv, add_zero_attn) 
        self.dropout2 = Dropout(dropout)
        # Inter-mechanism Attention Sub-Layer
        self.attn2 = InterMechanismAttentionSubLayer(self.d_mech, H_c, n_s, d_k, d_v, dropout, bias, add_bias_kv, add_zero_attn)
        self.dropout3 = Dropout(dropout)
        # Mechanism-wise, Position-Wise, FFN SubLayer
        self.ffn = MechanismWisePositionWiseFFNSubLayer(self.d_mech, n_s, dffn_m, bias)
        self.dropout4 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, h : Tensor, src_mask : Optional[Tensor] = None, src_key_padding_mask : Optional[Tensor] = None) :
        """
        h : (seq_len, batch_size, d_model), d_model = d_mech*n_s
        """
        assert self.d_mech * self.n_s == h.size(2)
        seq_len = h.size(0)
        batch_size = h.size(1)

        h = h.reshape(seq_len, batch_size, self.d_mech, self.n_s) # (seq_len, batch_size,  d_mech, n_s)
        # todo : transpose instead
        #h = h.transpose(-1,)
        h = h.view((self.n_s, seq_len, batch_size, self.d_mech)) # (n_s, seq_len, batch_size, d_mech)

        # todo : reshape key_padding_mask and attn_mask if not None

        # Step 1: Compute Mechanism Competition
        c = self.competitionSubLayer(h) # (ns, seq_len, batch_size, 1)

        # Step 2: Mechanism-wise self-attention sub-layer
        h = self.attn1(h = h, c = c, key_padding_mask = key_padding_mask, attn_mask=attn_mask) # (n_s, seq_len, batch_size, d_mech)

        # Step 3: Inter-mechanism Attention Sub-Layer
        h = self.attn2(h = h, key_padding_mask = key_padding_mask, attn_mask=attn_mask) # (n_s, seq_len, batch_size, d_mech)

        # Step 4: Mechanism-wise, Position-Wise, FFN SubLayer
        h = self.ffn(h) # (n_s, seq_len, batch_size, d_mech)

        h = torch.stack(h)
        h = h.reshape(seq_len, batch_size, self.d_mech, self.n_s) # (seq_len, batch_size,  d_mech, n_s)
        h = h.view((seq_len, batch_size, self.d_mech * self.n_s)) # (seq_len, batch_size, d_model)
        return h