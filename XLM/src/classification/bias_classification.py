# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch.utils.data import Dataset
from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import os
import json
from tqdm import tqdm
import itertools
import gc
import time
import copy
from collections import OrderedDict
import numpy as np
import pandas as pd
from pandas.io.parsers import ParserError
import matplotlib.pyplot as plt
import random

from logging import getLogger

#import fastBPE

#from transformers import BertModel, BertTokenizer
from .utils import to_bpe_py, to_bpe_cli, get_data_path, path_leaf 
from .metrics import top_k
from ..utils import truncate, AttrDict
from ..optim import get_optimizer
from ..data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from ..model.transformer import TransformerModel
from ..model.embedder import SentenceEmbedder, get_layers_positions
from ..trainer import Trainer as MainTrainer

#git clone https://github.com/NVIDIA/apex
#pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
#import apex

# bias corpus
special_tokens = ["<url>", "<email>", "<phone>", "<number>", "<digit>", "<cur>"]
dico_rest=4+len(special_tokens)

logger = getLogger()

def init_linear(linear, in_features, is_first):
    return
    with torch.no_grad():
        if True :
            nn.init.normal_(linear.weight, mean=0, std=1) 
            #nn.init.xavier_uniform_(proj.weight)
            nn.init.constant_(linear.bias, 0.)
        else :
            # Inspired from Sitzmann et al. (2020), Implicit Neural Representations with Periodic Activation Functions. arXiv: 2006.09661 [cs.CV]
            if is_first :
                a  = 1 / in_features   
            else :
                a = np.sqrt(6 / in_features)
                
            linear.weight.uniform_(-a, a) 
            nn.init.constant_(linear.bias, 0.)
        
def init_rnn(rnn):
    # https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605/2?u=pascal_notsawo
    return
    with torch.no_grad():
        for name, param in rnn.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

class GRU(nn.Module):
    def __init__(self, d_model, params):
        super().__init__()
        self.drop = nn.Dropout(params.dropout if not params.freeze_transformer else 0)
        self.rnn = nn.GRU(d_model, params.hidden_dim, num_layers = params.n_layers, 
                            bidirectional = params.bidirectional, batch_first = True,
                            dropout = 0 if params.n_layers < 2 else params.dropout)
    def forward(self, x):
        # x : (input_seq_len, batch_size, d_model)
        x = x.transpose(0, 1) # (batch_size, input_seq_len, d_model)
        _, x = self.rnn(self.drop(x)) # [n_layers * n_directions, batch_size, d_model]
        if self.rnn.bidirectional:
            return torch.cat((x[-2,:,:], x[-1,:,:]), dim = 1)
        else:
            return x[-1,:,:]   
        
class PredLayer4Classification(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    BERT model for token-level classification
    debug_num = 0 : Linear/AdaptiveLogSoftmaxWithLoss
    debug_num = 1 : Linear + Tanh + Dropout + Linear/AdaptiveLogSoftmaxWithLoss
    debug_num = 2 : GRU + Dropout + Linear/AdaptiveLogSoftmaxWithLoss
    """
    def __init__(self, d_model, n_labels, params, criterion = None, dropout = None):
        super().__init__()
        self.asm = params.asm
        self.n_labels = n_labels
        self.debug_num = params.debug_num
        
        if dropout is None :
            dropout = params.dropout if not params.freeze_transformer else 0
        
        if dropout != 0 :
            net = [nn.Dropout(dropout)]
        else :
            net = [nn.Identity()]
            
        if self.debug_num == 0 :
            if params.asm is False:
                net.append(nn.Linear(d_model, n_labels))
                init_linear(net[-1], d_model, is_first = True)
            else :
                in_features=d_model
        elif self.debug_num == 1 :
            net.extend([
                nn.Linear(d_model, params.hidden_dim), 
                #nn.BatchNorm1d(params.hidden_dim), 
                nn.Tanh(),
                #nn.ReLU(),
                nn.Dropout(params.dropout)
            ])
            init_linear(net[1], d_model, is_first = True)
            if params.asm is False:
                net.append(nn.Linear(params.hidden_dim, n_labels))
                init_linear(net[-1], params.hidden_dim, is_first = False)
            else :
                in_features=params.hidden_dim
        elif self.debug_num == 2 :
            net.append(GRU(d_model, params))
            init_rnn(net[1])
            if params.n_layers < 2 :
                net.append(nn.Dropout(params.dropout))            
            in_features = params.hidden_dim * 2 if params.bidirectional else params.hidden_dim
            if params.asm is False:
                net.append(nn.Linear(in_features, n_labels))
                init_linear(net[-1], in_features, is_first = False)
        else :
            raise NotImplementedError("debug_num = %d not found"%(self.debug_num))
        
        self.proj = nn.Sequential(*net)
        
        if params.asm is True:
            self.classifier = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=in_features,
                n_classes=n_labels,
                cutoffs=params.asm_cutoffs,
                div_value=params.asm_div_value,
                head_bias=True,  # default is False
            )
        
        else :
            if criterion is not None :
                self.criterion = criterion
                self.bce = False
                self.kl_div = False
            else :
                self.bce = True
                self.kl_div = False
                assert not (self.bce and self.kl_div)
                if params.version in [2, 4] :
                    if self.bce :
                        #self.criterion = nn.BCEWithLogitsLoss().to(params.device)
                        self.criterion = F.binary_cross_entropy_with_logits
                        #self.criterion = bce_bias_classification_loss
                        #self.criterion = bp_mll_loss
                        #self.criterion = gaussian_nll_loss
                    elif self.kl_div : 
                        self.criterion = kl_divergence_loss
                    else :
                        #self.criterion = BiasClassificationLoss(softmax = params.log_softmax).to(params.device)
                        self.criterion = bias_classification_loss  
                else :
                    #self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean').to(params.device)
                    self.criterion = F.cross_entropy

    def forward(self, x, y, weights = None, get_scores=True):
        """
        Compute the loss, and optionally the scores.
        """
        #x = F.normalize(input = x, p=2, dim=1, eps=1e-12, out=None)
        x = self.proj(x)
        if self.asm is False:
            if self.bce : # For binary_cross_entropy_with_logits, binarize the target
                w = (y != 0)
                y = w.to(y.dtype)
                #if weights is None :
                #    weights = w.to(float) # mask for loss
                pass
                
            scores = x.view(-1, self.n_labels)
            loss = self.criterion(scores, y, weight=weights)
        else:
            _, loss = self.classifier(x, y)
            scores = self.classifier.log_prob(x) if get_scores else None

        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """    
        x = self.proj(x)
        assert x.dim() == 2
        return self.classifier.log_prob(x) if self.asm else x
    
class PredLayer4Regression(nn.Module):
    def __init__(self, d_model, params):
        super().__init__()
        params.n_labels = 2
        net = [
            nn.Dropout(params.dropout if not params.freeze_transformer else 0),
            nn.Linear(d_model, params.hidden_dim), 
            nn.ReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.hidden_dim, 1) 
        ]
        self.proj1 = nn.Sequential(*net)
        self.proj2 = nn.Sequential(*net)
        
        self.criterion = nn.MSELoss()
        
        self.threshold = params.threshold
        
    def forward(self, x, y, weights = None, get_scores=True):
        """
        Compute the loss and scores 
        """
        y1 = self.proj1(x).view(-1)
        y2 = self.proj2(x).view(-1)
        scores = y1 >= self.threshold
        
        b, c = y[:,0], y[:,1]
        loss = self.criterion(y1, b) + self.criterion(y2, c)
        
        return scores, loss

class PredLayer4Workers(nn.Module):
    def __init__(self, d_model, params):
        super().__init__()
        
        num_workers = 3
        self.num_workers = num_workers
        
        if False :
            hidden_dim = params.hidden_dim
            self.proj = PredLayer4Classification(d_model, n_labels = hidden_dim, params = params).proj
            dropout = params.dropout
        else :
            hidden_dim = d_model
            self.proj = nn.Identity()
            dropout = params.dropout if not params.freeze_transformer else 0
        
        criterion = F.cross_entropy
        self.scores_proj = nn.ModuleList([
            PredLayer4Classification(hidden_dim, n_labels = 6, params = params, criterion = criterion,
                                    dropout = dropout)
            for _ in range(num_workers)
        ])
        
        self.confidences_proj = nn.ModuleList([
            PredLayer4Classification(hidden_dim, n_labels = 11, params = params, criterion = criterion, 
                                    dropout = dropout)
            for _ in range(num_workers)
        ])
        
        self.hidden_dim = hidden_dim
        self.topK = params.topK
        
    def forward(self, x, y, weights = None, get_scores=True):
        x = self.proj(x)
        
        b, c = y[:,0], y[:,1]
        
        t_loss = 0
        flag = True
        if flag :
            y_pred = {k : torch.empty_like(y) for k in range(self.topK)}
        else :
            y_pred = torch.empty_like(y).expand(self.topK, *y.shape)
        
        for i in range(self.num_workers) :
            scores, loss = self.scores_proj[i](x, y=b[:,i])
            t_loss = t_loss + loss
            
            confs, loss = self.confidences_proj[i](x, y=c[:,i])
            t_loss = t_loss + loss
            
            k = 0
            y_pred[k][:,0][:,i] = scores.max(dim=1)[1]
            y_pred[k][:,1][:,i] = confs.max(dim=1)[1]
            
            for k in range(1, self.topK) :
                y_pred[k][:,0][:,i] = torch.from_numpy(top_k(logits = scores.cpu(), y = b[:,i].cpu(), k = k+1)[-1]) #scores.max(dim=1)[1]
                y_pred[k][:,1][:,i] = torch.from_numpy(top_k(logits = confs.cpu(), y = c[:,i].cpu(), k = k+1)[-1]) #confs.max(dim=1)[1]
        
        if flag :
            y_pred = torch.stack([y_pred[k] for k in y_pred.keys()])
        
        return y_pred, t_loss/6
    

class PredLayer4BinaryClassification(nn.Module):
    def __init__(self, d_model, params):
        super().__init__()
        
        if False :
            hidden_dim = params.hidden_dim
            self.proj = PredLayer4Classification(d_model, n_labels = hidden_dim, params = params).proj
        else :
            hidden_dim = d_model
            self.proj = nn.Identity()
        
        criterion = F.binary_cross_entropy_with_logits
        self.classifier = PredLayer4Classification(hidden_dim, n_labels = 1, params = params, criterion = criterion)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, y, weights = None, get_scores=True):
        y = y.unsqueeze(1)
        x = self.proj(x)
        scores, loss = self.classifier(x=x, y=y, weights = weights)
        scores = self.sigmoid(scores).round().int() # = (self.sigmoid(scores) >= 0.5).int()
        return scores, loss

def get_pred_layer(d_model, n_labels, params):
    
    if params.version == 7 :
        return PredLayer4BinaryClassification(d_model, params)
    elif params.version == 6 :
        return PredLayer4Workers(d_model, params)
    elif params.version == 5 :
        return PredLayer4Regression(d_model, params)
    else :
        return PredLayer4Classification(d_model, n_labels, params)
    
## Bert for classification
class BertClassifier(nn.Module):
    """BERT model for token-level classification
    debug_num = 0 : Transformer + Linear 
    debug_num = 1 : Transformer + Linear + Tanh + Dropout + Linear
    debug_num = 2 : Transformer + GRU + Dropout + Linear'
    """
    def __init__(self, n_labels, params, logger):
        super().__init__()
        
        logger.warning("Reload dico & transformer model path from %s"%params.model_path)
        reloaded = torch.load(params.model_path, map_location=params.device)
        pretrain_params = AttrDict(reloaded['params'])
        logger.info("Supported languages: %s" % ", ".join(pretrain_params.lang2id.keys()))

        # build dictionary / build encoder / build decoder / reload weights
        try :
            dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'], rest=dico_rest)
        except AssertionError : # assert all(self.id2word[self.rest + i] == SPECIAL_WORD % i for i in range(SPECIAL_WORDS))
            dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
        self.dico = dico
        
        # update dictionary parameters
        pretrain_params.n_words = len(dico)
        pretrain_params.bos_index = dico.index(BOS_WORD)
        pretrain_params.eos_index = dico.index(EOS_WORD)
        pretrain_params.pad_index = dico.index(PAD_WORD)
        pretrain_params.unk_index = dico.index(UNK_WORD)
        pretrain_params.mask_index = dico.index(MASK_WORD)
        for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
            setattr(params, name, getattr(pretrain_params, name))

        #pretrain_params.emb_dim=64*16
        #pretrain_params.n_layers=24 
        #pretrain_params.n_heads=16
        #pretrain_params.dim_feedforward=2048
        try :
            model = TransformerModel(pretrain_params, dico, is_encoder=True, with_output=True).to(params.device)
        except AttributeError :
            # For models trained when TIM was not yet integrated : for example XLM pre-trained by facebook AI
            
            # AttributeError: 'AttrDict' object has no attribute 'dim_feedforward'
            # ...................................................'use_mine'
            # ...
            setattr(pretrain_params, "dim_feedforward", pretrain_params.emb_dim*4) # https://github.com/facebookresearch/XLM/blob/master/xlm/model/transformer.py#L268
            setattr(pretrain_params, "use_mine", params.use_mine)
            setattr(pretrain_params, "tim_layers_pos", params.tim_layers_pos)
            # ...
            model = TransformerModel(pretrain_params, dico, is_encoder=True, with_output=True).to(params.device)
            
        state_dict = reloaded['model']
        # handle models from multi-GPU checkpoints
        if 'checkpoint' in params.model_path:
            state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(params.device)
        self.embedder = SentenceEmbedder(model, dico, pretrain_params) #copy.deepcopy(SentenceEmbedder(model, dico, pretrain_params))
        params.freeze_transformer = params.finetune_layers == ""
        self.freeze_transformer = params.freeze_transformer
        
        if params.freeze_transformer :
            for param in self.embedder.model.parameters():
                param.requires_grad = False
        
        # adding missing parameters
        params.max_batch_size = 0
        params.n_langs = 1
        
        # reload langs from pretrained model
        #params.n_langs = embedder.pretrain_params['n_langs']
        #params.id2lang = embedder.pretrain_params['id2lang']
        #params.lang2id = embedder.pretrain_params['lang2id']
        params.lang = params.lgs
        params.lang_id = pretrain_params.lang2id[params.lang]

        d_model = model.dim
        params.hidden_dim = d_model if params.hidden_dim == -1 else params.hidden_dim
        self.pred_layer = get_pred_layer(d_model, n_labels, params).to(params.device)
        self.pred_layer.eval()

        self.whole_output = params.debug_num == 2
        self.finetune_layers = params.finetune_layers
        self.params = params
        
    def forward(self, x, lengths, y, positions=None, langs=None, weights = None, get_scores = True):
        """
        Inputs:
            `x`        : LongTensor of shape (slen, bs)
            `lengths`  : LongTensor of shape (bs,)
        """
        if self.freeze_transformer :
            with torch.no_grad():
                h = self.embedder.get_embeddings(x, lengths, positions=positions, langs=langs, whole_output = self.whole_output)
        else :
            h = self.embedder.get_embeddings(x, lengths, positions=positions, langs=langs, whole_output = self.whole_output)
        
        if True :
            return self.pred_layer(h, y, weights=weights)
        else :
            # L2 reg : weight_decay in optim handle this
            scores, loss = self.pred_layer(h, y, weights=weights)
            l2_lambda = 0.01
            l2_reg = 0
            for param in self.parameters():
                l2_reg += torch.norm(param)
            return scores, loss + l2_lambda * l2_reg

    def get_optimizers(self, params) :
        optimizer_p = get_optimizer(self.pred_layer.parameters(), params.optimizer_p)
        if params.finetune_layers == "" :
            return [optimizer_p]
        try :
            optimizer_e = get_optimizer(list(self.embedder.get_parameters(self.finetune_layers)), params.optimizer_e)
            return optimizer_e, optimizer_p
        except ValueError: #optimizer got an empty parameter list
            return [optimizer_p]
    
    def __str__(self) :
        return self.embedder.model.__str__() + self.pred_layer.__str__()
    
    def train(self):
        if not self.freeze_transformer :
            self.embedder.train()
        self.pred_layer.train()

    def eval(self):
        self.embedder.eval()
        self.pred_layer.eval()

    def state_dict(self):
        return {"embedder" : self.embedder.model.state_dict(), "pred_layer" : self.pred_layer.state_dict()}
    
    def load_state_dict(self, state_dict) :
        assert 'embedder' in state_dict.keys() and 'pred_layer' in state_dict.keys()
        self.embedder.model.load_state_dict(state_dict["embedder"])
        self.pred_layer.load_state_dict(state_dict["pred_layer"])
        
    def parameters(self) :
        return list(self.embedder.get_parameters(self.finetune_layers, log=False)) + list(self.pred_layer.parameters())

    def to(self, device) :
        self.embedder.model = self.embedder.model.to(device)
        self.pred_layer = self.pred_layer.to(device)
        return self

    def to_tensor(self, sentences):
        if type(sentences) == str :
            sentences = [sentences]
        else :
            assert type(sentences) in [list, tuple]
        
        # These two approaches are equivalent
        if True :
            word_ids = [torch.LongTensor([self.dico.index(w) for w in s.strip().split()]) for s in sentences]
            lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
            batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.params.pad_index)
            batch[0] = self.params.eos_index
            for j, s in enumerate(word_ids):
                if lengths[j] > 2:  # if sentence not empty
                    batch[1:lengths[j] - 1, j].copy_(s)
                batch[lengths[j] - 1, j] = self.params.eos_index
            langs = batch.clone().fill_(self.params.lang_id)
            batch, lengths = truncate(batch, lengths, self.params.max_len, self.params.eos_index)
            return batch, lengths, langs
        else :
            # add </s> sentence delimiters
            sentences = [(('</s> %s </s>' % sent.strip()).split()) for sent in sentences]
            bs = len(sentences)
            lengths = [len(sent) for sent in sentences]
            slen = max(lengths)
            lengths = torch.LongTensor(lengths)
            word_ids = torch.LongTensor(slen, bs).fill_(self.params.pad_index)
            for i in range(bs):
                sent = torch.LongTensor([self.dico.index(w) for w in sentences[i]])
                word_ids[:len(sent), i] = sent
                
            # NOTE: No more language id (removed it in a later version)
            # langs = torch.LongTensor([params.lang2id[lang] for _, lang in sentences]).unsqueeze(0).expand(slen, bs) if params.n_langs > 1 else None
            langs = None
            word_ids, lengths = truncate(word_ids, lengths, self.params.max_len, self.params.eos_index)
            return word_ids, lengths, langs
        
class GoogleBertClassifier(nn.Module):

    def __init__(self, n_labels, params, logger):
        super().__init__()
        from transformers import BertModel, BertTokenizer
        
        model_name = params.bert_model_name
        logger.warning("Reload BertModel : %s"%model_name)
        embedder = BertModel.from_pretrained(model_name) # load the pre-trained model
        embedder.eval()
        self.embedder = embedder.to(params.device)
        params.freeze_transformer = params.finetune_layers == ""
        self.freeze_transformer = params.freeze_transformer
        if params.freeze_transformer :
            for param in self.embedder.parameters():
                param.requires_grad = False

        logger.warning("Reload  BertTokenizer : %s"%model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer
        max_input_length = self.tokenizer.max_model_input_sizes[model_name]
        self.max_input_length = min(max_input_length, params.max_len)

        params.bos_index = tokenizer.cls_token_id
        params.eos_index = tokenizer.sep_token_id
        params.pad_index = tokenizer.pad_token_id
        params.unk_index = tokenizer.unk_token_id
        #params.mask_index = tokenizer.TODO
        params.n_words = len(self.tokenizer.vocab)
        params.n_langs = 1
        
        d_model = self.embedder.config.to_dict()['hidden_size']
        params.hidden_dim = d_model if params.hidden_dim == -1 else params.hidden_dim
        self.pred_layer = get_pred_layer(d_model, n_labels, params).to(params.device)
        self.pred_layer.eval()

        self.whole_output = params.debug_num == 2
        self.finetune_layers = params.finetune_layers
        self.params = params
        self.logger = logger
        
    def forward(self, x, lengths, y, positions=None, langs=None, weights = None, get_scores = True):
        """
        Inputs:
            `x`        : LongTensor of shape (bs, slen)
        """
        if self.freeze_transformer :
            with torch.no_grad():
                h = self.embedder(x)[0] # [batch size, sent len, emb dim]
        else :
            h = self.embedder(x)[0] 
        
        return self.pred_layer(h.transpose(0, 1) if self.whole_output else h[:, 0], y, weights=weights)

    def get_embedder_parameters(self, layer_range, log=True) :
        n_layers = len(self.embedder.encoder.layer)
        i, layers_position = get_layers_positions(layer_range, n_layers)
        
        if i is None :
            return []

        parameters = []
        
        if i == 0:
            # embeddings
            parameters += self.embedder.embeddings.parameters()
            if log :
                self.logger.info("Adding embedding parameters to optimizer")
        for l in layers_position :
            parameters += self.embedder.encoder.layer[l].parameters()
            if log :
                self.logger.info("Adding layer-%s parameters to optimizer" % (l + 1))

        return parameters

    def get_optimizers(self, params) :
        optimizer_p = get_optimizer(self.pred_layer.parameters(), params.optimizer_p)
        if params.finetune_layers == "" :
            return [optimizer_p]
        try :
            optimizer_e = get_optimizer(self.get_embedder_parameters(self.finetune_layers), params.optimizer_e)
            return optimizer_e, optimizer_p
        except ValueError: #optimizer got an empty parameter list
            return [optimizer_p]
        
    def train(self):
        if not self.freeze_transformer :
            self.embedder.train()
        self.pred_layer.train()

    def eval(self):
        self.embedder.eval()
        self.pred_layer.eval()

    def state_dict(self):
        return {"embedder" : self.embedder.state_dict(), "pred_layer" : self.pred_layer.state_dict()}
    
    def load_state_dict(self, state_dict) :
        assert 'embedder' in state_dict.keys() and 'pred_layer' in state_dict.keys()
        self.embedder.load_state_dict(state_dict["embedder"])
        self.pred_layer.load_state_dict(state_dict["pred_layer"])

    def parameters(self) :
        return list(self.get_embedder_parameters(self.finetune_layers, log=False)) + list(self.pred_layer.parameters())

    def to(self, device) :
        self.embedder = self.embedder.to(device)
        self.pred_layer = self.pred_layer.to(device)
        return self

    def tokenize_and_cut(self, sentence):
        tokens = self.tokenizer.tokenize(sentence) 
        tokens = tokens[:self.max_input_length-2]
        return [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.sep_token_id]

    def to_tensor(self, sentences):
        if type(sentences) == str :
            sentences = [sentences]
        else :
            assert type(sentences) in [list, tuple]

        sentences = [self.tokenize_and_cut(s) for s in sentences]
        bs = len(sentences)
        lengths = [len(sent) for sent in sentences]
        slen = max(lengths)
        lengths = torch.LongTensor(lengths)
        word_ids = torch.LongTensor(bs, slen).fill_(self.tokenizer.pad_token_id)
        for i in range(bs):
            sent = torch.LongTensor(sentences[i])
            word_ids[i,:len(sent)] = sent
        langs = None
        return word_ids, lengths, langs

class BiasClassificationDataset(Dataset):
    """ Dataset class for Bias Classification"""
    def __init__(self, file, split, params, model, logger, n_samples = None, min_len=1):
        assert params.version in [1, 2, 3, 4, 5, 6, 7]
        assert split in ["train", "valid", "test"]
        
        # For large data, it is necessary to process them only once
        data_path = get_data_path(params, file, n_samples, split)
        if os.path.isfile(data_path) :
            logger.info("Loading data from %s ..."%data_path)
            loaded_self = torch.load(data_path)
            for attr_name in dir(loaded_self) :
                try :
                    setattr(self, attr_name, getattr(loaded_self, attr_name))
                except AttributeError :
                    pass
            return
        
        logger.info("Loading data from %s ..."%file)

        self.params = params
        self.to_tensor = model.to_tensor
        self.n_samples = n_samples
        self.shuffle = params.shuffle if split == "train" else False
        self.group_by_size = params.group_by_size
        self.version = params.version
        self.in_memory = params.in_memory
        self.threshold = params.threshold
        self.do_augment = params.do_augment
        self.do_downsampling = params.do_downsampling
        self.do_upsampling = params.do_upsampling
        assert not (self.do_downsampling and self.do_upsampling)
        
        if params.data_columns == "" :
            # assume is bias classification
            num_workers = 3
            self.text_column = "content"
            self.scores_columns = ['answerForQ1.Worker%d'%(k+1) for k in range(num_workers)]
            self.confidence_columns = ['answerForQ2.Worker%d'%(k+1) for k in range(num_workers)]
        else :
            # "content,scores_columns1-scores_columns2...,confidence_columns1-confidence_columns2..."
            """
            For text classification tasks other than bias classification. 
            Just make sure that the scores_columns matches the label, 
            and choose version 1 or 4 (4 is more stable and allows the model to converge quickly, we control the loss function)
            
            For example for stackoverflow tags classification task : data_columns='post,tags'
            """
            data_columns = params.data_columns.split(",")
            assert len(data_columns) >= 2
            self.text_column = data_columns[0]
            self.scores_columns = data_columns[1].split("-")
            if len(data_columns) > 2 :
                self.confidence_columns = data_columns[2].split("-")
                assert len(self.scores_columns) == len(self.confidence_columns)
            else :
                assert len(self.scores_columns) == 1
                self.confidence_columns = []
                
        logger.info("Get instances...")
        try :
            data = [inst for inst in self.get_instances(pd.read_csv(file))]
        except ParserError : # https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
            data = [inst for inst in self.get_instances(pd.read_csv(file, lineterminator='\n'))]
        
        # remove short sentence
        l = len(data)
        data = [inst for inst in data if len(inst[0].split(" ")) >= min_len]
        logger.info('Remove %d sentences of length < %d' % (l - len(data), min_len))
        
        sentences = [inst[0] for inst in data]
        
        # lower
        logger.info("Do lower...")
        sentences = [s.lower() for s in sentences]
        
        if not params.google_bert :
            # bpe-ize sentences
            logger.info("bpe-ize sentences...")
            #sentences = to_bpe_cli(sentences, codes=params.codes, logger = logger, vocab = params.vocab)
            sentences = to_bpe_py(sentences, codes=params.codes, vocab = params.vocab)
        
        # check how many tokens are OOV
        if not params.google_bert :
            corpus = ' '.join(sentences).split()
            n_w = len([w for w in corpus])
            n_oov = len([w for w in corpus if w not in model.dico.word2id])
        else :
            corpus = model.tokenizer.tokenize(' '.join(sentences))
            n_w = len([w for w in corpus])
            n_oov = len([w for w in corpus if w not in model.tokenizer.vocab]) 
        p = n_oov/(n_w+1e-12)
        logger.info('Number of out-of-vocab words: %s/%s = %s %s' % (n_oov, n_w, p*100, "%"))
        
        for i in range(len(data)) :
            data[i][0] = sentences[i]
        
        if self.do_augment and split == "train" :
            p = 0.3
            max_change = 5
            logger.info("EDA text augmentation : p = %s, max_change = %s..."%(p, max_change))
            data, self.weights = self.augment(data, p = p, max_change = max_change) 
        if self.do_downsampling and split == "train":
            logger.info("Downsampling ...")
            data, self.weights = self.downsampling(data)
        if self.do_upsampling and split == "train" :
            logger.info("Upsampling ...")
            data, self.weights = self.upsampling(data)
            
        logger.info("Weigths %s"%str(self.weights))
        if self.params.weighted_training :
            weights = [w + 1e-12 for w in self.weights]
            weights = torch.FloatTensor([1.0 / w for w in weights])
            weights = weights / weights.sum()
            self.weights = weights.to(self.params.device)
        else :
            self.weights = None

        self.n_samples = len(data)
        self.batch_size = self.n_samples if self.params.batch_size > self.n_samples else self.params.batch_size
        
        if self.in_memory :
            logger.info("In memory ...")
            i = 0
            tmp = []
            while self.n_samples > i :
                i += self.batch_size
                inst = list(zip(*data[i-self.batch_size:i]))
                tmp.append(tuple([self.to_tensor(inst[0])] + [torch.stack(y) for y in inst[1:]]))
            self.data = tmp
        else :
            self.data = data
        
        # For large data, it is necessary to process them only once
        logger.info("Saving data to %s ..."%data_path)
        torch.save(self, data_path)
        
    def get_instances(self, df):
        columns = list(df.columns[1:]) # excapt "'Unnamed: 0'"
        rows = df.iterrows()
        if self.shuffle :
            if self.n_samples or (not self.group_by_size and not self.n_samples) :
                rows = list(rows)
                random.shuffle(rows)
        if self.n_samples :
            rows = list(rows)[:self.n_samples]
        if self.group_by_size :
            rows = sorted(rows, key = lambda x : len(x[1][self.text_column].split()), reverse=False)
        
        weights = [0] * self.params.n_labels
        
        for row in rows : 
            row = row[1]
            text = row[self.text_column]
            b = [row[col] for col in self.scores_columns]
            c = [row[col] for col in self.confidence_columns] if self.confidence_columns else [1] # Give a score of 1 to the label
        
            s = sum(c)
            s = 1 if s == 0 else s
            if self.version == 1 :
                #  label the average of the scores with the confidence scores as weighting
                label = int(round(sum([ score * conf for score, conf in  zip(b, c) ]) / s))
                weights[label] = weights[label] + 1 
                label = torch.tensor(label, dtype=torch.long)
                yield [text, label, label]
                
            elif self.version in [2, 3, 4] : 
                p_c = [0]*self.params.n_labels
                for (b_i, c_i) in zip(b, c) :
                    p_c[b_i] += c_i/s
                    
                if self.version == 4 :
                    #  label the average of the scores with the confidence scores as weighting
                    label = int(round(sum([ score * conf for score, conf in  zip(b, c) ]) / s))
                else :
                    # the class with the maximum confidence score was used as the target
                    label = b[np.argmax(a = c)] 
                weights[label] = weights[label] + 1
                yield [text, torch.tensor(p_c, dtype=torch.float), torch.tensor(label, dtype=torch.long)]
            
            elif self.version == 5 :
                # bias regression
                b = sum(b) / len(b)
                c = sum(c) / len(c)
                label = int(b >= self.threshold) # 1 if b >= self.threshold else 0
                weights[label] = weights[label] + 1
                yield [text, torch.tensor([b, c], dtype=torch.float), torch.tensor(label, dtype=torch.long)]
            
            elif self.version == 6:
                label = int(round(sum([ score * conf for score, conf in  zip(b, c) ]) / s))
                weights[label] = weights[label] + 1
                yield [text, torch.tensor([b, c], dtype=torch.long), torch.tensor(label, dtype=torch.long)]
            
            elif self.version == 7 :
                #label = int(round(sum([ score * conf for score, conf in  zip(b, c) ]) / s))
                label = sum([ score * conf for score, conf in  zip(b, c) ]) / s
                label = int(label >= self.threshold) # 1 if label >= self.threshold else 0
                weights[label] = weights[label] + 1
                yield [text, torch.tensor(label, dtype=torch.float), torch.tensor(label, dtype=torch.long)]
            
        self.weights = weights

    def __len__(self):
        if self.in_memory :
            return self.n_samples // self.batch_size
        else :
            return self.n_samples

    def __getitem__(self, index):
        if not self.in_memory :
            inst = self.data[index]
            return tuple([self.to_tensor(inst[0])] + [torch.stack(y) for y in inst[1:]])
        else :
            return self.data[index]
        
    def __iter__(self): # iterator to load data
        if self.shuffle :
            random.shuffle(self.data)
        if not self.in_memory :
            i = 0
            while self.n_samples > i :
                i += self.batch_size
                inst = list(zip(*self.data[i-self.batch_size:i]))
                tmp.append(tuple([self.to_tensor(inst[0])] + [torch.stack(y) for y in inst[1:]]))
        else :
            for batch in self.data :
                yield batch

    def downsampling(self, data):
        tmp = []
        count = [0]*self.params.n_labels
        # mean or min
        #n_max = sum(self.weights) / len(self.weights)
        n_max = min(self.weights)
        random.shuffle(data)
        for x in data :
            a = x[-1].item() 
            if n_max > count[a] : # and a != 3 :
                count[a] += 1
                tmp.append(x)
            """
            elif a == 3 :
                if 100 > count[a] :
                    count[a] += 1
                    tmp.append(x)
            """
        return tmp, count
    
    def upsampling(self, data) :
        tmp = []
        count = copy.deepcopy(self.weights)
        # mean or max
        #n_min = sum(self.weights) / len(self.weights)
        n_min = max(self.weights)
            
        for c, _ in enumerate(self.weights) :
            data4c = [inst for inst in data if inst[-1].item() == c]
            i, m = 0, len(data4c)
            if m != 0 :
                while n_min > count[c] :
                    tmp.append(data4c[i % m])
                    i += 1
                    count[c] += 1 
        random.shuffle(tmp)
        return data + tmp, count
    
    def augment(self, data, p = 0.3, max_change = 5) :
        """https://github.com/dsfsi/textaugment
        Do this before :
        pip install textaugment
        >>> import nltk
        >>> nltk.download('stopwords')
        >>> nltk.download('wordnet')
        """
        from textaugment import EDA
        t = EDA()
        tmp = []
        count = copy.deepcopy(self.weights)
        for inst in data :
            s = inst[0]
            label = inst[-1].item()
            l = len(s.split(" "))
            n = min(max_change, max(1, int(round(l*p))))

            aug = [
                t.synonym_replacement(s, n = n),
                t.random_deletion(s, p = p),
                t.random_swap(s, n = n),
                t.random_insertion(s, n = n)
            ]
            for s_aug in aug :
                if type(s_aug) == str and s_aug != "" and s_aug != s :
                    inst_aug = copy.deepcopy(inst)
                    inst_aug[0] = s_aug
                    tmp.append(inst_aug)
                    count[label] += 1
                    
        random.shuffle(tmp)
        return data + tmp, count
    
def load_dataset(params, logger, model) :
    params.train_n_samples = None if params.train_n_samples==-1 else params.train_n_samples
    params.valid_n_samples = None if params.valid_n_samples==-1 else params.valid_n_samples
    
    if not params.eval_only :
        train_dataset = BiasClassificationDataset(params.train_data_file, 'train', params, model, 
                                                logger, params.train_n_samples, min_len = params.min_len)
        setattr(params, "train_num_step", len(train_dataset))
        setattr(params, "train_num_data", train_dataset.n_samples)
    else :
        train_dataset = None
    
    logger.info("")
    val_dataset = BiasClassificationDataset(params.val_data_file, "valid", params, model, logger, params.valid_n_samples)

    logger.info("")
    logger.info("============ Data summary")
    if not params.eval_only :
        logger.info("train : %d"%train_dataset.n_samples)
    logger.info("valid : %d"%val_dataset.n_samples)
    logger.info("")
    
    return train_dataset, val_dataset

""" Training Class"""
#possib = ["%s_%s_%s"%(i, j, k) for i, j, k in itertools.product(["train", "val"], ["mlm", "nsp"], ["ppl", "acc", "loss"])]
possib = []
tmp_type = lambda name : "ppl" in name or "loss" in name

class Trainer(object):
    """Training Helper Class"""
    def __init__(self, params, model, optimizers, train_data_iter, val_data_iter, logger):
        self.params = params
        self.model = model
        self.optimizers = optimizers # optim

        # iterator to load data
        self.train_data_iter = train_data_iter 
        self.val_data_iter = val_data_iter 

        self.device = params.device # device name
        self.logger = logger

        # epoch / iteration size
        self.epoch_size = self.params.epoch_size
        if self.epoch_size == -1 and not params.eval_only:
            self.epoch_size = self.params.train_num_data
        assert self.epoch_size > 0 or params.eval_only
        
        # add metrics and topK to possible metrics
        global possib
        possib.extend(["%s_%s"%(i, j) for i, j in itertools.product(["train", "val"], ["f1_score_weighted", "acc", "loss", "IoU_weighted", "MCC"])])
        tmp = []
        for k in range(1, params.n_labels+1):
            tmp.extend(["%s_%s"%(i, j) for i, j in itertools.product(["top%d"%k], possib)])
        possib.extend(tmp)

        # validation metrics
        self.metrics = []
        metrics = [m for m in self.params.validation_metrics.split(',') if m != '']
        for i in range(len(metrics)) :
            if tmp_type(metrics[i]) :
                metrics[i] = '_%s'%metrics[i]
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            assert m[0] in possib
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # stopping criterion used for early stopping
        if self.params.stopping_criterion != '':
            split = self.params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            assert split[0] in possib
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0

            if tmp_type(split[0]) :
                split[0] = '_%s'%split[0]

            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_criterion = None

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0
        self.stats = OrderedDict([('processed_s', 0), ('processed_w', 0)])
        self.all_scores = []
        self.last_time = time.time()

        self.log_interval = self.params.log_interval
        if self.log_interval == -1 and not params.eval_only:
            self.log_interval = self.params.batch_size
        assert self.log_interval > 0 or params.eval_only

        if params.reload_checkpoint :
            self.load_checkpoint(checkpoint_path = params.reload_checkpoint)        

        self.checkpoint_path = os.path.join(params.dump_path, "checkpoint.pth")
        if os.path.isfile(self.checkpoint_path) :
            # sometime : RuntimeError: [enforce fail at inline_container.cc:145] . PytorchStreamReader failed reading zip archive: failed finding central directory
            self.load_checkpoint()
            
        if params.reload_model :
            logger.warning("Reload model from %s"%params.reload_model)
            assert os.path.isfile(params.reload_model)
            self.load(model_file = params.reload_model)
    
        nb_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'Found {nb_p:,} trainable parameters in model.\n')
        
        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        #assert params.amp >= 0 or params.accumulate_gradients == 1
        #self.model = self.model.to(self.device)
        if params.multi_gpu and params.amp == -1:
            self.logger.info("Using nn.parallel.DistributedDataParallel ...")
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[params.local_rank], output_device=params.local_rank, broadcast_buffers=True)
        
        if params.amp >= 0 and not params.eval_only:
            self.init_amp()
            if params.multi_gpu:
                self.logger.info("Using apex.parallel.DistributedDataParallel ...")
                import apex
                self.model = apex.parallel.DistributedDataParallel(self.model, delay_allreduce=True)

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        assert self.params.amp == 0 and self.params.fp16 is False or self.params.amp in [1, 2, 3] and self.params.fp16 is True
        
        # Allow Amp to perform casts as required by the opt_level : https://nvidia.github.io/apex/amp.html
        import apex
        if len(self.optimizers) == 1 :
            self.model, self.optimizers[0] = apex.amp.initialize(self.model, self.optimizers[0], opt_level='O%i' % self.params.amp)
        else :
            raise RuntimeError("Not supported")
        
    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1

    def optimize(self, loss, retain_graph=False):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            self.logger.warning("NaN detected")
            # exit()

        # regular optimization
        if self.params.amp == -1:
            if self.params.accumulate_gradients == 1 :
                for optimizer in self.optimizers :
                    optimizer.zero_grad()
                    
                loss.backward(retain_graph=retain_graph)
                
                if self.params.clip_grad_norm > 0:
                    # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                    clip_grad_norm_(self.model.parameters(), self.params.clip_grad_norm)
                    # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                    # self.logger.info(norm_check_a, norm_check_b)
                
                for optimizer in self.optimizers :
                    optimizer.step()
                    
            else : # accumulate gradient if need
                # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
                if self.n_iter % self.params.accumulate_gradients == 0:
                    loss.backward(retain_graph=retain_graph)
                    if self.params.clip_grad_norm > 0:
                        # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                        clip_grad_norm_(self.model.parameters(), self.params.clip_grad_norm)
                        # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                    for optimizer in self.optimizers :
                        optimizer.step()
                    for optimizer in self.optimizers :
                        optimizer.zero_grad()
                else :
                    loss.backward(retain_graph=retain_graph)


        # AMP optimization
        else:
            import apex
            if self.n_iter % self.params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, self.optimizers[0]) as scaled_loss:
                    scaled_loss.backward(retain_graph=retain_graph)
                
                if self.params.clip_grad_norm > 0:
                    # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                    clip_grad_norm_(apex.amp.master_params(self.optimizers[0]), self.params.clip_grad_norm)
                    # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                    # self.logger.info(norm_check_a, norm_check_b)

                self.optimizers[0].step()
                self.optimizers[0].zero_grad()
            else:
                with apex.amp.scale_loss(loss, self.optimizers[0], delay_unscale=True) as scaled_loss:
                    scaled_loss.backward(retain_graph=retain_graph)

    def plot_score(self, scores):
        for key, value in scores.items():
            try :
                self.logger.info("%s -> %.6f" % (key, value))
            except TypeError: #must be real number, not dict
                self.logger.info("%s -> %s" % (key, value))
        if self.params.is_master:
            self.logger.info("__log__:%s" % json.dumps(scores))

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                self.logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                self.logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_checkpoint('best_%s' % metric, include_optimizer=False)

    def save_checkpoint(self, name, include_optimizer = True, include_all_scores=False):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        checkpoint_path = os.path.join(self.params.dump_path, '%s.pth' % name)
        self.logger.info("Saving %s to %s ..." % (name, checkpoint_path))

        data = {
            "model" : self.model.state_dict(), 
            "params": {k: v for k, v in self.params.__dict__.items()},
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_criterion': self.best_criterion
        }

        if include_optimizer:
            self.logger.warning(f"Saving optimizer ...")
            data['optimizer'] = [optimizer.state_dict() for optimizer in self.optimizers]
        
        if include_all_scores :
            self.logger.warning(f"Saving all scores ...")
            data['all_scores'] = self.all_scores
            score_path = os.path.join(self.params.dump_path, 'all_scores.pth')
            torch.save(self.all_scores, score_path)

        torch.save(data, checkpoint_path)

    def load_checkpoint(self, checkpoint_path = None):
        """
        Reload a checkpoint if we find one.
        """
        """
        checkpoint_path = self.checkpoint_path
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        """
        checkpoint_path = self.checkpoint_path if checkpoint_path is None else checkpoint_path

        reloading_checkpoint_condition = not self.params.eval_only or (self.params.eval_only and not self.params.reload_model)  

        if reloading_checkpoint_condition : 
            if self.params.eval_only :
                self.logger.warning("You started the evaluation without specifying the model to be used for the evaluation, so the last checkpoint found will be loaded.")
            self.logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")

        assert os.path.isfile(checkpoint_path)
        data = torch.load(checkpoint_path, map_location='cpu')
        # reload model parameters
        self.model.load_state_dict(data["model"])

        if not self.params.eval_only :
            # reload optimizer
            if reloading_checkpoint_condition :
                if False:  # AMP checkpoint reloading is buggy, we cannot do that - TODO: fix - https://github.com/NVIDIA/apex/issues/250
                    self.logger.warning(f"Reloading checkpoint optimizer ...")
                    for optimizer, state_dict in zip(self.optimizers, data['optimizer']) :
                        optimizer.load_state_dict(state_dict)
                else:  # instead, we only reload current iterations / learning rates
                    self.logger.warning(f"Not reloading checkpoint optimizer.")
                    for optimizer in self.optimizers :
                        for group_id, param_group in enumerate(optimizer.param_groups):
                            if 'num_updates' not in param_group:
                                self.logger.warning(f"No 'num_updates' for optimizer.")
                                continue
                            self.logger.warning(f"Reloading 'num_updates' and 'lr' for optimizer.")
                            param_group['num_updates'] = data['optimizer']['param_groups'][group_id]['num_updates']
                            param_group['lr'] = optimizer.get_lr_for_step(param_group['num_updates'])

            # reload main metrics
            self.epoch = data['epoch'] + 1
            self.n_total_iter = data['n_total_iter']
            self.best_metrics = data['best_metrics']
            self.best_criterion = data['best_criterion']
            
            if 'all_scores' in data :
                self.all_scores = data['all_scores']
                #score_path = os.path.join(self.params.dump_path, 'all_scores.pth')
                #if os.path.isfile(score_path) :
                #    self.all_scores = torch.load(score_path)
            
            if reloading_checkpoint_condition :
                self.logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")
            else :
                self.logger.warning(f"Parameters reloaded. Epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_checkpoint('periodic_%i' % self.epoch, include_optimizer=False)

    def load(self, model_file = None, pretrain_file = None):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file and os.path.isfile(model_file):
            #self.logger.info('Loading the model from', model_file)
            data = torch.load(model_file, map_location='cpu')
            if type(data) == dict :
                data = data["model"]
            self.model.load_state_dict(data)

        elif pretrain_file and os.path.isfile(pretrain_file): # use pretrained transformer
            #self.logger.info('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.pth'): # pretrain model file in pytorch
                data = torch.load(pretrain_file, map_location='cpu')
                if type(data) == dict :
                    data = data["model"]
                """
                self.model.transformer.load_state_dict(
                    {key[12:]: # remove 'transformer.' (in 'transformer.embedding.norm.bias' for example)
                        value
                        for key, value in data.items()
                        if key.startswith('transformer')} # load only transformer parts
                )
                """
            else :
                raise RuntimeError("Incorrect file extension")

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (self.params.is_master or not False):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_criterion:
                self.best_criterion = scores[metric]
                self.logger.info("New best validation score: %f" % self.best_criterion)
                self.decrease_counts = 0
            else:
                self.logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                self.logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                if self.params.multi_gpu and 'SLURM_JOB_ID' in os.environ:
                    os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                exit()
        self.save_checkpoint("checkpoint", include_optimizer=True, include_all_scores=True)
        self.epoch += 1
        
    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % self.log_interval != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            #if type(self.stats[k]) is list:
            #    del self.stats[k][:]
            if ("loss" in k or "acc" in k or 'f1_score' in k or "IoU" in k or "MCC" in k) and (not "avg" in k) :
                self.stats[k] = []

        # learning rates
        s_lr = ""
        s_lr = s_lr + (" - LR: ")
        for optimizer in self.optimizers :
            s_lr += " " + " / ".join("{:.4e}".format(group['lr']) for group in optimizer.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
            self.stats['processed_s'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_s'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        progress = str(self.stats['progress'])+"% -"
        # log progress + speed + stats + learning rate
        self.logger.info("")
        self.logger.info(s_iter + progress + s_speed + s_stat + s_lr)

    def train_step(self, get_loss):
        self.model.train() # train mode
        total_stats = []

        for i, batch in enumerate(self.train_data_iter):

            # forward / loss
            loss, stats = get_loss(self.model, batch, self.params, self.train_data_iter.weights)
            #loss = loss.mean() # mean() for Data Parallelism
            
            # optimize
            self.optimize(loss)

            total_stats.append(stats)

            # number of processed sentences / words
            self.n_sentences += self.params.batch_size
            self.stats['processed_s'] += self.params.batch_size
            self.stats['processed_w'] += stats['n_words']
            self.stats['progress'] = min(int(((i+1)/self.params.train_num_step)*100), 100) 

            for name in stats.keys() :
                if ("loss" in name or 'acc' in name or "f1_score" in name or "IoU" in name or "MCC" in name) and not "top" in name:
                    self.stats[name] = self.stats.get(name, []) + [stats[name]]

            self.iter()
            self.print_stats()
    
            if self.epoch_size < self.n_sentences :
                break

        return total_stats

    def eval_step(self, get_loss):
        self.model.eval() # eval mode
        total_stats = []
        with torch.no_grad(): 
            for batch in tqdm(self.val_data_iter, desc='val'):
                _, stats = get_loss(self.model, batch, self.params, self.val_data_iter.weights) 
                total_stats.append(stats)
        return total_stats
    
    def train(self, get_loss, end_of_epoch):
        """ Train Loop """
        
        for _ in range(self.params.max_epoch):
            
            self.logger.info("============ Starting epoch %i ... ============" % self.epoch)
            self.n_sentences = 0
            for k in self.stats.keys():
                if "avg" in k :
                    self.stats[k] = []
            train_stats = self.train_step(get_loss)
            
            self.logger.info("============ End of epoch %i ============" % self.epoch)

            val_stats = self.eval_step(get_loss)

            scores = end_of_epoch([val_stats, train_stats])
            self.all_scores.append(scores)
            
            self.plot_score(scores)

            # end of epoch
            self.save_best_model(scores)
            self.save_periodic()
            self.end_epoch(scores)
            
        plot_all_scores(self.all_scores)
        
    def eval(self, get_loss, end_of_epoch):
        """ Eval Loop """
        val_stats = self.eval_step(get_loss)
        scores = end_of_epoch([val_stats], add_output = True)
        
        predictions = {}
        s = {}
        
        keys = scores.keys()
        for k in keys :
            if 'y2' in k or "logits" in k or "y1" in k :
                predictions[k] = scores[k]
            else :
                s[k] = scores[k]
        
        filename, _ = os.path.splitext(path_leaf(self.params.val_data_file))
        predictions_path = os.path.join(self.params.dump_path, '%s_predictions.pth'%filename)
        torch.save(predictions, predictions_path)
        
        self.plot_score(s)
    
def plot_all_scores(scores=None, from_path="") :
    assert scores is not None or os.path.isfile(from_path)
    if scores is None :
        scores = torch.load(from_path)
        if "all_scores" in scores :
            scores = scores["all_scores"]

    to_plot = ['loss', 'acc', 'f1_score_weighted', 'IoU_weighted']
    prefix = ['train', 'val']
    suptitle=""
    k = 0
    if True :
        to_plot.append("MCC")
        nrows, ncols = len(to_plot), 1
        fig, ax = plt.subplots(nrows, ncols, sharex=False, figsize = (20, 20))
        fig.suptitle(suptitle)
        for i in range(nrows) :
            name = to_plot[k]
            for p in prefix :
                label = "%s_%s"%(p,name)
                y = [s[label] for s in scores]
                x = list(range(len(y)))
                ax[i].plot(x, y, label=label)
            ax[i].set(xlabel='epoch', ylabel=p)
            ax[i].set_title('%s per epoch'%name)
            ax[i].legend()
            #ax[i].label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.
            k += 1
            if k == len(to_plot) :
                break
    else :
        nrows, ncols = 2, 2
        fig, ax = plt.subplots(nrows, ncols, sharex=False, figsize = (20, 8))
        fig.suptitle(suptitle)
        for i in range(nrows) :
            for j in range(ncols) :
                name = to_plot[k]
                for p in prefix :
                    label = "%s_%s"%(p,name)
                    y = [s[label] for s in scores]
                    x = list(range(len(y)))
                    ax[i][j].plot(x, y, label=label)
                ax[i][j].set(xlabel='epoch', ylabel=p)
                ax[i][j].set_title('%s per epoch'%name)
                ax[i][j].legend()
                #ax[i][j].label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.
                k += 1
                if k == len(to_plot) :
                    break
    plt.show()
    
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

# https://github.com/idocx/BP_MLL_Pytorch/blob/master/bp_mll.py
def bp_mll_loss(c: Tensor, y: Tensor, bias=(1, 1), weight=None) -> Tensor:
    r"""compute the loss, which has the form:
        L = \sum_{i=1}^{m} \frac{1}{|Y_i| \cdot |\bar{Y}_i|} \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
    :param c: prediction tensor, size: batch_size * n_labels
    :param y: target tensor, size: batch_size * n_labels
    :return: size: scalar tensor
    """
    assert len(bias) == 2 and all(map(lambda x: isinstance(x, int) and x > 0, bias)), "bias must be positive integers"
    
    y = y.float()
    y_bar = -y + 1
    y_norm = torch.pow(y.sum(dim=(1,)), bias[0])
    y_bar_norm = torch.pow(y_bar.sum(dim=(1,)), bias[1])
    assert torch.all(y_norm != 0) or torch.all(y_bar_norm != 0), "an instance cannot have none or all the labels"
    return torch.mean(1 / torch.mul(y_norm, y_bar_norm) * pairwise_sub_exp(y, y_bar, c))

def pairwise_sub_exp(y: Tensor, y_bar: Tensor, c: Tensor) -> Tensor:
    r"""compute \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}"""
    truth_matrix = y.unsqueeze(2).float() @ y_bar.unsqueeze(1).float()
    exp_matrix = torch.exp(c.unsqueeze(1) - c.unsqueeze(2))
    return (torch.mul(truth_matrix, exp_matrix)).sum(dim=(1, 2))

def hamming_loss(c: Tensor, y: Tensor, threshold=0.8) -> Tensor:
    """compute the hamming loss (refer to the origin paper)
    :param c: size: batch_size * n_labels, output of NN
    :param y: size: batch_size * n_labels, target
    :return: Scalar
    """
    assert 0 <= threshold <= 1, "threshold should be between 0 and 1"
    p, q = c.size()
    return 1.0 / (p * q) * (((c > threshold).int() - y) != 0).float().sum()

def one_errors(c: Tensor, y: Tensor) -> Tensor:
    """compute the one-error function"""
    p, _ = c.size()
    return (y[0, torch.argmax(c, dim=1)] != 1).float().sum() / p

def gaussian_nll_loss(logits, target, weight=None) :
    # https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
    bs, n_class = target.shape
    #var = torch.ones(bs, n_class, requires_grad=True).to(logits.device) #heteroscedastic
    var = torch.ones(bs, 1, requires_grad=True).to(logits.device) #homoscedastic
    return F.gaussian_nll_loss(input = F.softmax(logits, dim=1), target = target, var = var, full=False, eps=1e-06, reduction='mean')