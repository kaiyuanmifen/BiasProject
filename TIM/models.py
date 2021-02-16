# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple
import copy
from typing import Callable, List, Optional, Tuple, Any
import math
import numpy as np
import warnings

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Dropout, Linear, LayerNorm, Sequential
# The disadvantage of their method (MultiheadAttention) is that they impose d_embed=dk=dv.
#from torch.nn import MultiheadAttention
from torch.nn.init import xavier_uniform_
from torch.autograd import Variable

class ModelConfig(NamedTuple):
    "Configuration for BERT model"
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers 

    d_model:int = 512 # Dimension of Hidden Layer in Transformer Encoder
    num_heads: int = 8 # Numher of Heads in Multi-Headed Attention Layers
    d_k: int = 512
    d_v: int = 512
    num_encoder_layers: int = 6 # Numher of Hidden Layers
    dim_feedforward: int = 2048 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    dropout_rate: int = 0.1
    vocab_size: int = None # Size of Vocabulary
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments


def custom_rand(shape : tuple, a = 0, b = 1., random_seed = 0, requires_grad = False) :
    """generate a random tensor of shape `shape` fill with number in range (a, b)"""
    torch.manual_seed(random_seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    return (b - a) * torch.rand(shape).requires_grad_(requires_grad) + b 

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        #return lambda x : x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
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


## Position-wise Feed-Forward Networks
class Dense(nn.Module):
    """tf.keras.layers.Dense"""
    def __init__(self, in_features, out_features, bias = True, activation = None):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias = bias)
        self.activation = activation if activation else lambda x : x

        # nn.init.normal_(self.linear.weight, mean=0, std=1)
        # nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.constant_(self.linear.bias, 0.)

    def forward(self, x):
        return self.activation(self.linear(x))

class FFN(Module):
    """Position-wise Feed-Forward Networks : FFN(x) = max(0, xW1 + b1)W2 + b2"""
    def __init__(self, d_model, dff, dropout_rate = 0.) :
        super(FFN, self).__init__()
        self.net = Sequential(
            Dense(in_features = d_model, out_features = dff, activation = _get_activation_fn("relu")),  # (seq_len, batch_size, dff)
            Dense(in_features = dff, out_features = d_model),  # (seq_len, batch_size, d_model)
            # We delegate this to the encoder and decoder layer, just like the norm layer.
            #Dropout(p = dropout_rate),
            )
    def forward(self, x) :
        return self.net(x)

## Scaled dot product attention
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, 
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        """
        query : (batch_size, len_q, d_k)
        key : (batch_size, len_k, d_k)
        value : (batch_size, len_k, d_v)
        attn_mask : (len_q, len_k) or (batch_size, len_q, len_k)
        key_padding_mask : (batch_size, len_q) or (batch_size, len_q, 1)

        return : # (batch_size, len_q, d_v), (batch_size, num_heads, len_q, len_k)

        key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, where padding elements are indicated by 1s.
    
        attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """
        batch_size, len_q, _ = q.shape
        len_k = k.size(1)
        
        #neg_inf = float("-inf") 
        #neg_inf = -np.inf

        # https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py#L122
        neg_inf = -1e8
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        attn = torch.bmm(q, k.transpose(1, 2)) # (batch_size, len_q, len_k)
        attn = attn / self.temperature # (batch_size, len_q, len_k)

        if attn_mask is not None:
            # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L4715
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
                if list(attn_mask.size()) != [1, len_q, len_k]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [batch_size, len_q, len_k]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
                    #raise NotImplementedError("3D")
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))


            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(attn_mask.bool(), neg_inf) # (batch_size, len_q, len_k)
                #attn_weights.masked_fill_(attn_mask == 0, neg_inf)
            else: # float
                attn += attn_mask
                #raise NotImplementedError("")

        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L4739
        # Masking to ignore padding (query side)
        if key_padding_mask is not None:
            # convert ByteTensor key_padding_mask to bool
            if key_padding_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
                )
                key_padding_mask = key_padding_mask.to(torch.bool)
        
            assert key_padding_mask.size(0) == batch_size
            assert key_padding_mask.size(1) == len_q

            if key_padding_mask.dim() == 2: # (batch_size, len_q) 
                key_padding_mask = key_padding_mask.unsqueeze(-1) # (batch_size, len_q, 1)

            if key_padding_mask.dtype == torch.bool:
                attn = attn.masked_fill(key_padding_mask, neg_inf) # (batch_size, len_q, len_k)
                #attn_weights.masked_fill_(key_padding_mask == 0, neg_inf)
            else: # float
                attn += key_padding_mask
                #raise NotImplementedError("")

        attn = self.softmax(attn) # (batch_size, len_q, len_k)
        attn = self.dropout(attn) # (batch_size, len_q, len_k)
        output = torch.bmm(attn, v) # (batch_size, len_q, d_v)

        return output, attn

## Multi-head attention
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, num_heads, d_k, d_v, dropout=0.1, debug_num = 0):
        super(MultiHeadAttention, self).__init__()
        self.debug_num = debug_num # Just a difference in implementation details
        self.num_heads = num_heads
        if debug_num == 0 :
            self.d_k = d_k
            self.d_v = d_v

            self.W_Q = nn.Linear(d_model, num_heads * d_k)
            self.W_K = nn.Linear(d_model, num_heads * d_k)
            self.W_V = nn.Linear(d_model, num_heads * d_v)
        
        elif debug_num == 1 :
            self.W_Q = _get_clones(nn.Linear(d_model, d_k), num_heads)
            self.W_K = _get_clones(nn.Linear(d_model, d_k), num_heads)
            self.W_V = _get_clones(nn.Linear(d_model, d_v), num_heads)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                      attn_dropout=dropout)
        self.W_0 = nn.Linear(num_heads * d_v, d_model)

        # We delegate this to the encoder and decoder layer, just like the norm layer.
        #self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, 
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        """
        query : (batch_size, len_q, d_model)
        key : (batch_size, len_k, d_model)
        value : (batch_size, len_v, d_model)
        attn_mask : (len_q, len_k) or (batch_size, len_q, len_k)
        key_padding_mask : (batch_size, len_q) or (batch_size, len_q, 1)

        return : # (batch_size, len_q, d_model), (batch_size, num_heads, len_q, len_k)

        key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, where padding elements are indicated by 1s.
    
        attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """
        if self.debug_num == 0 :
            d_k, d_v, num_heads = self.d_k, self.d_v, self.num_heads

            batch_size, len_q, _ = q.size()
            batch_size, len_k, _ = k.size()
            batch_size, len_v, _ = v.size()

            q = self.W_Q(q).view(batch_size, len_q, num_heads, d_k)
            k = self.W_K(k).view(batch_size, len_k, num_heads, d_k)
            v = self.W_V(v).view(batch_size, len_v, num_heads, d_v)

            q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (num_heads*batch_size) x len_q x d_k
            k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (num_heads*batch_size) x len_k x d_k
            v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (num_heads*batch_size) x len_v x d_v

            # Repeat masks num_heads times
            if attn_mask is not None:
                attn_mask = attn_mask.repeat(num_heads, 1, 1)  # (num_heads*batch_size) x len_q x len_k
            if key_padding_mask is not None:
                # todo ?
                key_padding_mask = key_padding_mask.unsqueeze(-1) #.repeat(1, 1, len_q)
                key_padding_mask = key_padding_mask.repeat(num_heads, 1, 1)

            output, attn = self.attention(q, k, v, key_padding_mask, attn_mask)

            output = output.view(num_heads, batch_size, len_q, d_v) # num_heads x batch_size x len_q x d_v
            output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1)  # batch_size x len_q x (num_heads*d_v)

            output = self.W_0(output) # batch_size x len_q x d_model
            #output = self.dropout(output) # batch_size x len_q x d_model
            
            # attn : # (batch_size * num_heads) x len_q x len_k
            attn = attn.contiguous().view(batch_size, num_heads, len_q, len_k) # batch_size x num_heads x len_q x len_k

            return output, attn

        elif self.debug_num == 1 :
            # Same mask applied to all heads.
            if attn_mask is not None:
                #attn_mask = attn_mask.unsqueeze(1)
                pass
            if key_padding_mask is not None :
                key_padding_mask = key_padding_mask.unsqueeze(-1) 
            
            heads = [ 
                    self.attention(W_Q_i(q), W_K_i(k), W_V_i(v), key_padding_mask, attn_mask) # (batch_size, len_q, d_v), (batch_size, len_q, len_k)
                    for W_Q_i, W_K_i, W_V_i in zip(self.W_Q, self.W_K, self.W_V)
                    ]
            output, attn = zip(*heads) # (batch_size, len_q, d_v)*num_heads, (batch_size, len_q, len_k)*num_heads
            
            output = torch.cat(output, dim=-1) # batch_size x len_q x (num_heads*d_v)
            output = self.W_0(output) # batch_size x len_q x d_model
            #output = self.dropout(output) # batch_size x len_q x d_model
            
            attn = torch.stack(attn, dim=1) # batch_size x num_heads x len_q x len_k

            return output, attn

## Encoder layer
class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        num_heads: the number of heads in the multiheadattention models (required).
        d_k : key dim
        d_v : value dim
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, num_heads, d_k, d_v, dim_feedforward=2048, dropout=0.1, activation="relu", debug_num = 0):
        super(TransformerEncoderLayer, self).__init__()

        assert d_model % num_heads == 0
        
        self.d_model = d_model 

        self.self_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout=dropout, debug_num = debug_num)
        # Delegates from the MultiheadAttention layer
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

        self.ffn = FFN(d_model = d_model, dff = dim_feedforward, dropout_rate = dropout)
        # Delegates from the Position-wise Feed-Forward Networks
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, 
                src_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
        
            scr : (batch_size, input_dim, d_model)
            attn_mask : (input_dim, input_dim) or (batch_size, input_dim, input_dim)
            key_padding_mask : (batch_size, input_dim)

            return : # (batch_size, input_dim, d_model), (batch_size, num_heads, input_dim, target_dim)
        """
        src2, enc_slf_attn = self.self_attn(q = src, k = src, v = src,  
                                            attn_mask=src_mask, 
                                            key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        
        src2 = self.ffn(src)
        src = self.norm2(src + self.dropout2(src2))
        
        return src, enc_slf_attn


## **TIM (as Encoder Layer)**

### Group Linear Layer
def GroupLinear(inputs : Tensor, layers : ModuleList, activation = None) -> Tensor :
    """
    Args:
        inputs  : (ns, batch_size, _, d_in) 
        layers : (ns, d_in, d_out)
        activation : callable

    return (ns, batch_size, _, d_out)
    """
    assert len(layers) == len(inputs)
    if activation is not None:
        return [activation(layer(x)) for layer, x in zip(layers, inputs)]
    else :
        return [layer(x) for layer, x in zip(layers, inputs)]

### Mechanism Competition
class MechanismCompetitionLayer(Module):
    """Compute Mechanism Competition"""
    def __init__(self, d_mech, n_s, bias = True) :
        """
        d_mech : single mechanism dimension
        n_s : number of mechanisms
        """
        super(MechanismCompetitionLayer, self).__init__()
        self.W_c = _get_clones(module = nn.Linear(d_mech, 1, bias = bias), N = n_s)
        
    def forward(self, h : Tensor) -> Tensor:
        """
        h : (ns, batch_size, seq_len, d_mech)
        """
        c = GroupLinear(inputs = h, layers = self.W_c) # (batch_size, seq_len, 1)*ns
        c = torch.stack(c) # (ns, batch_size, seq_len, 1)
        #print(c)
        c = F.softmax(input = c, dim = 0) # (ns, batch_size, seq_len, 1)
        #print(c)

        return c

### Mechanism-wise self-attention sub-layer
class MechanismWiseSelfAttentionSubLayer(Module):
    def __init__(self, d_mech, H, n_s, d_k, d_v, dropout=0.1, bias = True) :
        super(MechanismWiseSelfAttentionSubLayer, self).__init__()
        self.attn = _get_clones(module = MultiHeadAttention(d_mech, H, d_k, d_v, dropout), N = n_s)
        self.norm = LayerNorm(d_mech)
    
    def forward(self, h : Tensor, c : Tensor, 
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
            h : input hidden representation  
            c : Mechanism Competition
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            h : (n_s, batch_size, seq_len, d_mech)
            c : (n_s, batch_size, seq_len, 1)
            src_mask : (seq_len, seq_len) or (batch_size, seq_len, seq_len)
            src_key_padding_mask : (batch_size, seq_len)

            return : # (n_s, batch_size, seq_len, d_mech), todo
        """

        M = [
             layer(q = x, k = x, v = x, 
                   key_padding_mask = key_padding_mask, attn_mask = attn_mask) 
                  # (batch_size, seq_len, d_mech), (batch_size, H, seq_len, seq_len)
             for layer, x in zip(self.attn, h) ] 
        
        M, attn = zip(*M) # (batch_size, seq_len, d_mech)*n_s, (batch_size, H, seq_len, seq_len)*n_s

        h = [self.norm(h_i + c_i * M_i) for h_i, c_i, M_i in zip(h, c, M)] # (batch_size, seq_len, d_mech)*n_s
        h = torch.stack(h) # (n_s, batch_size, seq_len, d_mech)

        attn = torch.stack(attn) # (n_s, batch_size, H, seq_len, seq_len)

        return h, attn

### Inter-mechanism Attention Sub-Layer
class InterMechanismAttentionSubLayer(Module):
    """"""
    def __init__(self, d_mech, H_c, n_s, d_k, d_v, dropout=0.1, bias = True) :
        """"""
        super(InterMechanismAttentionSubLayer, self).__init__()
        self.attn = _get_clones(module = MultiHeadAttention(d_mech, H_c, d_k, d_v, dropout), N = n_s)
        self.norm = LayerNorm(d_mech)
 
    def forward(self, h : Tensor, 
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
            h : input hidden representation  
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
 
        Shape:
        
            h : (n_s, batch_size, seq_len, d_mech)
            src_mask : (seq_len, seq_len) or (batch_size, seq_len, seq_len)
            src_key_padding_mask : (batch_size, seq_len)
 
            return : # (n_s, batch_size, seq_len, d_mech), todo
        """
        mask, query_mask = None, None # todo
 
        M = [
             layer(q = x, k = x, v = x, 
                   key_padding_mask = key_padding_mask, attn_mask = attn_mask) 
              # (batch_size, seq_len, d_mech), (batch_size, H_c, seq_len, seq_len)
             for layer, x in zip(self.attn, h) ] 
        
        M, enc_dec_attn = zip(*M) # (batch_size, seq_len, d_mech)*n_s, (batch_size, H_c, seq_len, seq_len)*n_s
        
        h = [self.norm(h_i + M_i) for h_i, M_i in zip(h,  M)] # (batch_size, seq_len, d_mech)*n_s
        h = torch.stack(h) # (n_s, batch_size, seq_len, d_mech)
 
        enc_dec_attn = torch.stack(enc_dec_attn) # (n_s, batch_size, H_c, seq_len, seq_len)
 
        return h, enc_dec_attn

### Mechanism-wise, Position-Wise, FFN SubLayer
class MechanismWisePositionWiseFFNSubLayer(Module):
    def __init__(self, d_mech, n_s, dffn_m, bias = True) :
      
        super(MechanismWisePositionWiseFFNSubLayer, self).__init__()
        self.W_1 = _get_clones(module = nn.Linear(d_mech, dffn_m, bias = bias), N = n_s)
        self.W_2 = _get_clones(module = nn.Linear(dffn_m, d_mech, bias = bias), N = n_s)
        self.norm = LayerNorm(d_mech)
        self.activation = _get_activation_fn("relu")
    
    def forward(self, h):
        """
        h : (n_s, batch_size, seq_len, d_mech)
        """
        out = GroupLinear(h, self.W_1, activation=self.activation) # (n_s, batch_size, seq_len, dffn_m)
        out = GroupLinear(out, self.W_2) # (n_s,  batch_size, seq_len, d_mech)
        h = [self.norm(h_i + F_i) for h_i, F_i in zip(h, out)] # (batch_size, seq_len, d_mech)*n_s
        h = torch.stack(h) # (n_s, batch_size, seq_len, d_mech)
        return h

### TIM Encoder-Layer
class TIM_EncoderLayer(Module):
    def __init__(self, d_model, d_ffn, n_s, d_k, d_v, H, H_c, dropout=0.1, activation="relu", bias = True) :
        """
        n_s : number of mechanisms
        d_k : key size
        d_v : value size
        H : number of heads for self-attention 
        H_c : number of heads for inter-mechanism attention
        """
        super(TIM_EncoderLayer, self).__init__()

        assert d_model % n_s == 0
        assert d_ffn % n_s == 0

        self.n_s = n_s
        self.d_mech = d_model // n_s
        dffn_m = d_ffn // n_s 
        
        # Mechanism Competition sub-layer
        self.competitionSubLayer = MechanismCompetitionLayer(self.d_mech, n_s, bias = bias) 
        
        # Mechanism-wise self-attention sub-layer
        self.attn1 = MechanismWiseSelfAttentionSubLayer(self.d_mech, H, n_s, d_k, d_v, dropout, bias) 
        self.dropout1 = Dropout(dropout)

        # Inter-mechanism Attention Sub-Layer
        self.attn2 = InterMechanismAttentionSubLayer(self.d_mech, H_c, n_s, d_k, d_v, dropout, bias)
        self.dropout2 = Dropout(dropout)

        # Mechanism-wise, Position-Wise, FFN SubLayer
        self.ffn = MechanismWisePositionWiseFFNSubLayer(self.d_mech, n_s, dffn_m, bias)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, h : Tensor, 
                src_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor] :
        """
            h : input hidden representation  
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
        
            h : (batch_size, seq_len, d_model), d_model = d_mech*n_s
            src_mask : (seq_len, seq_len) or (batch_size, seq_len, seq_len)
            src_key_padding_mask : (batch_size, seq_len)

            return : # (batch_size, seq_len, d_model), (batch_size, num_heads, input_dim, target_dim), -*-
        """
        assert self.d_mech * self.n_s == h.size(2)
        batch_size = h.size(0)
        seq_len = h.size(1)
         
        h = h.contiguous().view(batch_size, seq_len, self.d_mech, self.n_s) # (batch_size, seq_len, d_mech, n_s)
        h = h.permute(3, 0, 1, 2) # (n_s, batch_size, seq_len, d_mech)

        # Step 1: Compute Mechanism Competition
        c = self.competitionSubLayer(h) # (n_s, batch_size, seq_len, 1)

        # Step 2: Mechanism-wise self-attention sub-layer
        h, attn1 = self.attn1(h, c, src_key_padding_mask, src_mask) # (n_s, batch_size, seq_len, d_mech), (n_s, batch_size, H, seq_len, seq_len)
        h = self.dropout1(h)

        # Step 3: Inter-mechanism Attention Sub-Layer
        h, attn2 = self.attn2(h, src_key_padding_mask, src_mask) # (n_s, batch_size, seq_len, d_mech), (n_s, batch_size, H_c, seq_len, seq_len)
        h = self.dropout2(h)

        # Step 4: Mechanism-wise, Position-Wise, FFN SubLayer
        h = self.ffn(h) # (n_s, batch_size, seq_len, d_mech)
        h = self.dropout3(h)

        h = h.permute(1, 2, 3, 0) # (batch_size, seq_len, d_mech, n_s)
        h = h.contiguous().view(batch_size, seq_len, self.d_mech * self.n_s) # (seq_len, batch_size, d_model)
        
        return h, attn1, attn2

## Encoder
class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).

        tim_encoder_layer : an instance of the TIM_EncoderLayer() class (required).
        tim_layers_pos : in the encoder layer list, which positions tim_encoder_layer occupies.

        if there are only tim_encoder_layer, it is better to set it to encoder_layer.
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, dropout = 0.1,  
                 tim_encoder_layer = None, tim_layers_pos : list = None
                 ):
        super(TransformerEncoder, self).__init__()
        
        if tim_encoder_layer is None :
            self.layers = _get_clones(encoder_layer, num_layers)
        else :
            assert  tim_layers_pos is not None
            self.layers = []
            for i in range(num_layers) :
                if i in  tim_layers_pos :
                    self.layers.append(copy.deepcopy(tim_encoder_layer))  
                else :
                    self.layers.append(copy.deepcopy(encoder_layer))
            self.layers = ModuleList(self.layers)                    
            
    def forward(self, src: Tensor, mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src : (batch_size, seq_len, d_model)
            mask: (seq_len, seq_len) or (batch_size, seq_len, seq_len)
            src_key_padding_mask: (batch_size, seq_len) 

            return : # (batch_size, seq_len, d_model), num_layers * (batch_size, num_heads, seq_len, seq_len)
        """
        output = src 
        attns = []
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attns.append(output[-1])
            output = output[0]

        return output, attns 

## token, position and segment embedding 
class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, vocab_size, d_model, max_len, n_segments, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model) # token embedding
        self.pos_embed = nn.Embedding(max_len, d_model) # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model) # segment(token type) embedding
 
        self.norm = LayerNorm(d_model)
        self.drop = Dropout(dropout)
 
    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)
 
        emb = self.tok_embed(x)
        # Attention Is All You Need, section 3.4, Embeddings and Softmax
        # In the embedding layers, we multiply those weights by sqrt(d_model)
        #emb = emb*torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        emb = emb + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(emb))

# Transformer
class Transformer(Module):
    r"""A Transformer model. 

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        num_heads: the number of heads in the multiheadattention models (default=8).
        d_k:(default=d_model)
        d_v:(default=d_model)
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        vocab_size : 
        custom_embedding : None
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
  
        tim_encoder_layer : (default=None)
        tim_layers_pos : (default=None)
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_k = None, d_v = None, 
                 num_encoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 vocab_size = None, 
                 max_len = None, n_segments = None, 
                 custom_embedding = None, 
                 activation: str = "relu", 
                 custom_encoder: Optional[Any] = None,
                 tim_encoder_layer = None, tim_layers_pos : list = None) -> None:
        super(Transformer, self).__init__()

        d_k = d_k if d_k is not None else d_model 
        d_v = d_v if d_v is not None else d_model
        self.d_model = d_model 
                
        assert vocab_size is not None or custom_embedding is not None
        if custom_embedding is not None :
            self.embedding = custom_embedding
        else :
            assert max_len is not None and n_segments is not None
            self.embedding = Embeddings(vocab_size, d_model, max_len, n_segments, dropout)

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_k, d_v, dim_feedforward, dropout, activation)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                              dropout = dropout,
                                              tim_encoder_layer = tim_encoder_layer, 
                                              tim_layers_pos = tim_layers_pos)

        self._reset_parameters()

    def forward(self, src: Tensor, seg, src_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None,
                ):
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            seg : segment(token type) for embedding
            src_mask: the additive mask for the src sequence (optional). 
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
           
        Shape:
            S is the source sequence length and N is the batch size

            - src: (N, S) 
            - seg : (N, S)
            - src_mask: (S, S) / (N, S, S)
            - src_key_padding_mask: (N, S) 
           
            Note: src_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight. 
            src_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: (batch_size, seq_len, d_model), *
        """
        output = src # (batch_size, seq_len)

        # adding embedding and position encoding.
        output = self.embedding(output, seg)  # (batch_size, seq_len, d_model)
        output, enc_attn = self.encoder(output, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output, enc_attn

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

## BertModel for Pretrain
class BertModel4Pretrain(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, transformer : Transformer):
        super().__init__()
        self.transformer = transformer
        d_model = transformer.d_model
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = _get_activation_fn("gelu")
        self.norm = LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.transformer.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        """
        # todo : check
        input_ids : (batch_size, input_seq_len)
        segment_ids : (batch_size, input_seq_len)
        input_mask : (batch_size, input_seq_len) 
        masked_pos : (batch_size, input_seq_len)  
        """
        src_mask = None
        src_key_padding_mask = input_mask
        h, _ = self.transformer(input_ids, segment_ids, src_mask, src_key_padding_mask)
        # h : (batch_size, input_seq_len, d_model)
        # [CLS] : The final hidden state corresponding to this token is used as the aggregate 
        # sequence representation for classification
        # BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (page : todo)
        C = h[:, 0] #or h[:,0,:] # (batch_size, d_model)
        pooled_h = self.activ1(self.fc(C)) # (batch_size, d_model)
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1)) # (batch_size, input_seq_len, d_model)
        h_masked = torch.gather(h, 1, masked_pos) # (batch_size, input_seq_len, vocab_size)
        h_masked = self.norm(self.activ2(self.linear(h_masked))) # (batch_size, input_seq_len, vocab_size)
        logits_lm = self.decoder(h_masked) + self.decoder_bias # (batch_size, input_seq_len, vocab_size)
        logits_clsf = self.classifier(pooled_h) # (batch_size, 2)

        return logits_lm, logits_clsf

## Bert for classification
class BertClassifier(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, transformer : Transformer, n_labels, dropout=0.1):
        super().__init__()
        self.transformer = transformer
        d_model = transformer.d_model
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        src_mask = None
        src_key_padding_mask = input_mask
        h, _ = self.transformer(input_ids, segment_ids, src_mask, src_key_padding_mask)
        # h : (batch_size, input_seq_len, d_model)
        # [CLS] : The final hidden state corresponding to this token is used as the aggregate 
        # sequence representation for classification
        # BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (page : todo)
        C = h[:, 0] #or h[:,0,:] # (batch_size, d_model)
        pooled_h = self.activ(self.fc(C)) # (batch_size, d_model)
        logits = self.classifier(self.drop(pooled_h)) # (batch_size, n_labels)
        return logits