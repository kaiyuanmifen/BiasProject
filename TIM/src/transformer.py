# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
from typing import Optional, Any

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

class PositionalEncoding():

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def __call__(self, seq_len):
    	#seq_len = x.size(0)
        return self.pe[:seq_len, :]

class Dense(nn.Module):
    """tf.keras.layers.Dense"""
    def __init__(self, in_features, out_features, bias = True, activation = None):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias = bias)
        self.activation = activation if activation else lambda x : x
    def forward(self, x):
        return self.activation(self.linear(x))

class FFN(Module):
    def __init__(self, d_model, dff, dropout_rate = 0.) :
        super(FFN, self).__init__()
        self.net = Sequential(
            Dense(in_features = d_model, out_features = dff, activation=torch.relu),  # (batch_size, seq_len, dff)
            Dropout(p = dropout_rate),
            Dense(in_features = dff, out_features = d_model)  # (batch_size, seq_len, d_model)
            )
    def forward(self, x) :
        return self.net(x)


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()

        self.d_model = d_model 

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

        self.ffn = FFN(d_model = d_model, dff = dim_feedforward, dropout_rate = dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        #src2 = self.self_attn(query = src, key = src, value = src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2, _ = self.self_attn(query = src, key = src, value = src, 
                                 attn_mask=src_mask, 
                                 key_padding_mask=src_key_padding_mask) # (input_seq_len, batch_size, d_model)
        src = self.norm1(src + self.dropout1(src2))
        
        src2 = self.ffn(src)
        src = self.norm2(src + self.dropout2(src2))
        
        return src

class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()

        self.d_model = d_model 

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        
        self.ffn = FFN(d_model = d_model, dff = dim_feedforward, dropout_rate = dropout)
        self.norm3 = LayerNorm(d_model)      
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        #tgt2 = self.self_attn(query = tgt, key = tgt, value = tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt2, _ = self.self_attn(query = tgt, key = tgt, value = tgt, 
                                 attn_mask=tgt_mask, 
                                 key_padding_mask=tgt_key_padding_mask) # (target_seq_len, batch_size, d_model)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        #tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        
        tgt2, _ = self.multihead_attn(query = tgt, key = memory, value = memory, 
                                      attn_mask=memory_mask, 
                                      key_padding_mask=memory_key_padding_mask) # (target_seq_len, batch_size, d_model)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
  
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(tgt2))

        return tgt

class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, maximum_position_encoding = 10000, dropout = 0.1,  
                 input_vocab_size = None, custom_embedding = None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.dropout = Dropout(p=dropout)
        
        self.d_model = encoder_layer.d_model 
        assert input_vocab_size or custom_embedding
        if custom_embedding is not None :
            self.embedding = custom_embedding
        else :
            self.embedding = nn.Embedding(input_vocab_size, self.d_model)

        self.pos_encoding = PositionalEncoding(d_model = self.d_model, dropout=0.1, max_len=maximum_position_encoding)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src # (input_seq_len, batch_size)
        seq_len = src.size(0)

        # adding embedding and position encoding.
        output = self.embedding(output)  # (input_seq_len, batch_size, d_model)
        output = output*torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        output = output + self.pos_encoding(seq_len)
        output = self.dropout(output)

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None, maximum_position_encoding = 10000, dropout = 0.1,
                 target_vocab_size = None, custom_embedding = None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.dropout = Dropout(p=dropout)
        
        self.d_model = decoder_layer.d_model 
        assert target_vocab_size or custom_embedding
        if custom_embedding is not None :
            self.embedding = custom_embedding
        else :
            self.embedding = nn.Embedding(target_vocab_size, self.d_model)

        self.pos_encoding = PositionalEncoding(d_model = self.d_model, max_len=maximum_position_encoding)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt # (target_seq_len, batch_size)
        seq_len = tgt.size(0)

        # adding embedding and position encoding.
        output = self.embedding(output)  # (target_seq_len, batch_size, d_model)
        output = output*torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        output = output + self.pos_encoding(seq_len)
        output = self.dropout(output)

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Transformer(Module):
    r"""A transformer model. 

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        input_vocab_size : 
        target_vocab_size :
        pe_input : 10000
        pe_target : 10000,
        custom_input_embedding : None
        custom_target_embedding : None,
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 input_vocab_size = None, target_vocab_size=None, pe_input=10000, pe_target=10000,
                 custom_input_embedding = None, custom_target_embedding = None,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None) -> None:
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm,
                                              maximum_position_encoding = pe_input,  
                                              input_vocab_size = input_vocab_size, 
                                              custom_embedding = custom_input_embedding)


        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              maximum_position_encoding = pe_target,  
                                              target_vocab_size = target_vocab_size, 
                                              custom_embedding = custom_target_embedding)
            
        #self.final_layer = nn.Linear(in_features = target_vocab_size, out_features = target_vocab_size, bias = True)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight. 
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, N, E)`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        """

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        #if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
        #    raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

