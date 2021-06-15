# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import os
import copy

#from transformers import BertModel, BertTokenizer

from .metrics import top_k
from .utils import special_tokens, to_tensor
from .loss import bias_classification_loss, bce_bias_classification_loss, BiasClassificationLoss, kl_divergence_loss, nll_loss, bp_mll_loss, gaussian_nll_loss
from .loss import FocalLoss, BCEFocalLoss

from ..utils import truncate, AttrDict
from ..optim import get_optimizer
from ..data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from ..model.transformer import TransformerModel, Embedding
from ..model.embedder import SentenceEmbedder, get_layers_positions

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
    def forward(self, x, lengths = None):
        # x : (input_seq_len, batch_size, d_model)
        x = self.drop(x)
        if lengths is not None :
            lengths = lengths.cpu().to(torch.int64)
            packed_embedded = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
            _, x = self.rnn(packed_embedded)
            #packed_output, x = self.rnn(packed_embedded)
            #output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        else :
            x = x.transpose(0, 1) # (batch_size, input_seq_len, d_model)
            _, x = self.rnn(x) # [n_layers * n_directions, batch_size, d_model]
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
            self.bce = False
            self.kl_div = False
            if criterion is not None :
                self.criterion = criterion
            else :
                assert not (self.bce and self.kl_div)
                if params.version in [2, 4] :
                    self.bce = True
                    self.kl_div = False
                    assert not (self.bce and self.kl_div)
                    if self.bce :
                        #self.criterion = nn.BCEWithLogitsLoss().to(params.device)
                        self.criterion = F.binary_cross_entropy_with_logits
                        #self.criterion = bce_bias_classification_loss
                        #self.criterion = bp_mll_loss
                        #self.criterion = gaussian_nll_loss
                        #self.criterion = BCEFocalLoss()
                    elif self.kl_div : 
                        self.criterion = kl_divergence_loss
                    else :
                        #self.criterion = BiasClassificationLoss(softmax = params.log_softmax).to(params.device)
                        self.criterion = bias_classification_loss  
                else :
                    #self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean').to(params.device)
                    self.criterion = F.cross_entropy
                    #self.criterion = FocalLoss()

    def forward(self, x, y, weights = None, get_scores=True):
        """
        Compute the loss, and optionally the scores.
        """
        #x = F.normalize(input = x, p=2, dim=1, eps=1e-12, out=None)
        x = self.proj(x)
        #x = F.dropout(x, p=0.1, training=True)
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
        
        self.do_proj = True
        if self.do_proj :
            hidden_dim = params.hidden_dim
            self.proj = PredLayer4Classification(d_model, n_labels = hidden_dim, params = params).proj
            #self.proj = nn.Sequential(
            #    PredLayer4Classification(d_model, n_labels = hidden_dim, params = params).proj,
            #    #nn.Tanh(),
            #    #nn.ReLU(),
            #)
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
        
        if self.do_proj :
            criterion = F.binary_cross_entropy_with_logits
            #criterion = BCEFocalLoss()
            self.classifier = PredLayer4Classification(hidden_dim, n_labels = 1, params = params, criterion = criterion)
            self.sigmoid = nn.Sigmoid()
        
        self.hidden_dim = hidden_dim
        self.topK = params.topK
        self.threshold = params.threshold
        
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

        if self.do_proj :
            batch_size = c.size(0)
            y_true = (b * c / c.sum(dim=1).reshape(batch_size, -1)).sum(dim=1)
            y_true_bin = (y_true >= self.threshold).float().unsqueeze(1)
            scores, loss = self.classifier(x=x, y=y_true_bin)
            t_loss = t_loss + loss
            scores = self.sigmoid(scores).round().int() # = (self.sigmoid(scores) >= 0.5).int()
            return y_pred, t_loss/7, scores
        else :
            return y_pred, t_loss/6, None
        
class PredLayer4BinaryClassification(nn.Module):
    def __init__(self, d_model, params):
        super().__init__()
        
        self.do_proj = True
        if self.do_proj :
            hidden_dim = params.hidden_dim
            self.proj = PredLayer4Classification(d_model, n_labels = hidden_dim, params = params).proj
            #self.proj = nn.Sequential(
            #    PredLayer4Classification(d_model, n_labels = hidden_dim, params = params).proj,
            #    #nn.Tanh(),
            #    #nn.ReLU(),
            #)    
        else :
            hidden_dim = d_model
            self.proj = nn.Identity()
        
        criterion = F.binary_cross_entropy_with_logits
        #criterion = BCEFocalLoss()
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
class XLMBertClassifier(nn.Module):
    """BERT model for token-level classification
    debug_num = 0 : Transformer + Linear 
    debug_num = 1 : Transformer + Linear + Tanh + Dropout + Linear
    debug_num = 2 : Transformer + GRU + Dropout + Linear'
    """
    def __init__(self, n_labels, params, logger, pre_trainer = None):
        super().__init__()
        
        if pre_trainer is None :
            logger.warning("Reload dico & transformer model path from %s"%params.model_path)
            reloaded = torch.load(params.model_path, map_location=params.device)
            pretrain_params = AttrDict(reloaded['params'])
            logger.info("Supported languages: %s" % ", ".join(pretrain_params.lang2id.keys()))

            # build dictionary / build encoder / build decoder / reload weights
            try :
                dico_rest=4+len(special_tokens) # 4 ~ BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD
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
            
            for name in ['dropout', 'attention_dropout']:
                attr = getattr(params, name, None)
                if attr is not None and getattr(pretrain_params, name, attr) != attr :
                    setattr(pretrain_params, name, attr)

            """
            pretrain_params.emb_dim=64*16
            pretrain_params.n_layers=24 
            pretrain_params.n_heads=16
            pretrain_params.dim_feedforward=2048
            if getattr(pretrain_params, 'tim_layers_pos', '') :
                pretrain_params.tim_layers_pos="" # TODO
                pretrain_params.d_k = pretrain_params.emb_dim
                pretrain_params.d_v = pretrain_params.emb_dim
                pretrain_params.n_s = 2
            #"""
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
            d_model = model.dim
            
            # adding missing parameters
            params.max_batch_size = 0
            params.n_langs = 1
            
            # reload langs from pretrained model
            #params.n_langs = embedder.pretrain_params['n_langs']
            #params.id2lang = embedder.pretrain_params['id2lang']
            #params.lang2id = embedder.pretrain_params['lang2id']
            params.lang = params.lgs
            params.lang_id = pretrain_params.lang2id[params.lang]
            
            params.freeze_transformer = params.finetune_layers == ""    
            if params.freeze_transformer :
                for param in self.embedder.model.parameters():
                    param.requires_grad = False
        
        else :
            self.dico = pre_trainer.data["dico"]
            for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']+['lang2id', 'n_langs']:
                setattr(params, name, getattr(pre_trainer.params, name))
                
            self.embedder = SentenceEmbedder(pre_trainer.model, self.dico,  pre_trainer.params)
            d_model = pre_trainer.model.dim
            if type(params.lgs) == list :
                params.lang = params.lgs[0]
            else :
                params.lang = params.lgs
            params.lang_id = params.lang2id[params.lang]
            params.freeze_transformer = False
            
        self.freeze_transformer = params.freeze_transformer
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

    def predict(self, tensor, y, weights):
        """
        """
        return self.pred_layer(tensor[0] if not self.whole_output else tensor, y, weights=weights)

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
    
    def train(self, mode: bool = True):
        if not self.freeze_transformer :
            self.embedder.train(mode)
        self.pred_layer.train(mode)

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
        for module in embedder.modules() :
            if isinstance(module, nn.Dropout) :
                #module.p = params.attention_dropout
                if module.p != params.dropout :
                    module.p = params.dropout
                
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
        
    def train(self, mode):
        if not self.freeze_transformer :
            self.embedder.train(mode)
        self.pred_layer.train(mode)

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
        return to_tensor(sentences, pad_index = self.tokenizer.pad_token_id, tokenize_and_cut = self.tokenize_and_cut, batch_first = True)
    
class SimpleClassifier(nn.Module):
    def __init__(self, n_labels, params, logger, dico : Dictionary = None):
        super().__init__()

        params.pad_index = 2
        params.freeze_transformer = False
        params.google_bert = False
        params.n_langs = 1

        self.n_labels = n_labels
        embedding_dim = params.emb_dim
        self.pad_index = params.pad_index
        self.use_pretrained_word_embedding = params.use_pretrained_word_embedding
        self.max_input_length = params.max_len
        self.logger = logger
                
        if not self.use_pretrained_word_embedding :
            if dico is not None :
                self.dico = dico
                input_dim = len(self.dico.id2word)
            elif os.path.isfile(params.vocab) :
                logger.info('Loading bpe vocabulary from %s'%params.vocab)
                self.dico = Dictionary.read_vocab(params.vocab, special_tokens)
                self.pad_index = self.dico.pad_index
                params.pad_index = self.pad_index
                input_dim = len(self.dico.id2word)
            else :
                # will be reset
                self.dico = None
                input_dim = self.pad_index+1 # to avoid `AssertionError: Padding_idx must be within num_embeddings`
                
            self.embedding = Embedding(num_embeddings = input_dim, embedding_dim = embedding_dim, padding_idx=self.pad_index) 
            self.train_embedding = True
            params.train_embedding = True
        else :
            self.embedding = nn.Identity()
            self.train_embedding = params.train_embedding 
        
        # will be reset
        self.net = nn.Identity()
            
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(params.dropout)
        self.whole_output = params.debug_num == 2
        
        self.hidden_dim = params.hidden_dim

    def init_net(self, params) :
        if hasattr(self, "pad_index") :
            params.pad_index = self.pad_index
        
    def get_embedder_parameters(self, log=True) : 
        parameters = []
        if self.train_embedding :
            # embeddings
            parameters += self.embedding.parameters()
            if log :
                self.logger.info("Adding embedding parameters to optimizer")
                
        parameters += self.net.parameters()
        if log :
            self.logger.info("Adding %s parameters to optimizer"%type(self.net).__name__)
        return parameters
                
    def train(self, mode: bool = True):
        if self.train_embedding :
            self.embedding.train(mode=mode)
        self.net.train(mode=mode)
        self.pred_layer.train(mode=mode)

    def eval(self):
        self.embedding.eval()
        self.net.eval()
        self.pred_layer.eval()
        
    def self_to(self, device) :
        self.embedding = self.embedding.to(device)
        self.net = self.net.to(device)
        self.pred_layer = self.pred_layer.to(device)
        #return self

    def parameters(self) :
        return list(self.get_embedder_parameters(log=False)) + list(self.pred_layer.parameters())
    
    def get_optimizers(self, params) :
        optimizer_e = get_optimizer(self.get_embedder_parameters(), params.optimizer_e)
        optimizer_p = get_optimizer(self.pred_layer.parameters(), params.optimizer_p)
        return optimizer_e, optimizer_p
    
    def tokenize(self, x) : 
        return x.split(" ")
    
    def tokenize_and_cut(self, sentence):
        tokens = self.tokenize(sentence) 
        tokens = tokens[:self.max_input_length]
        return [self.vocab.stoi[t] for t in tokens]

class RecurrentClassifier(SimpleClassifier):
    def __init__(self, n_labels, params, logger, dico : Dictionary = None, with_output = True):
        super().__init__(n_labels, params, logger, dico)
        
        self.n_layers = params.n_layers
        self.bidirectional = params.bidirectional
        self.with_output = with_output

        if self.with_output :
            self.factor = 2 if self.bidirectional else 1
            self.pred_layer = get_pred_layer(self.hidden_dim * self.factor, n_labels, params).to(params.device)
        else :
            self.pred_layer = nn.Identity()
        
    def to_tensor(self, sentences):
        if not self.use_pretrained_word_embedding :
            return to_tensor(sentences, self.pad_index, dico = self.dico, batch_first = False)
        else :
            return to_tensor(sentences, self.pad_index, tokenize_and_cut = self.tokenize_and_cut, batch_first = False)
    
    def fwd(self, tensor, lengths, mask):
        if False : # RuntimeError: start (4) + length (2) exceeds dimension size (4).
            packed_embedded = nn.utils.rnn.pack_padded_sequence(tensor, lengths, enforce_sorted=False)
            packed_output, hidden = self.net(packed_embedded)
            output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
            hidden = output.transpose(0, 1)
            return self.dropout(hidden)
        else :
            output, _ = self.net(tensor)
            return self.dropout(output)
    
class RNNClassifier(RecurrentClassifier):
    def __init__(self, n_labels, params, logger, dico : Dictionary = None, with_output = True):
        super().__init__(n_labels, params, logger, dico, with_output)
        self.init_net(params)
    
    def init_net(self, params) :
        super().init_net(params)
        self.net = nn.RNN(self.embedding_dim, self.hidden_dim, num_layers = self.n_layers, 
                            bidirectional=self.bidirectional, dropout = 0 if self.n_layers < 2 else params.dropout)
        
    def forward(self, x, lengths, y, positions=None, langs=None, weights = None, get_scores = True):
        """
        Inputs:
            `x`        : LongTensor of shape (slen, bs)
        """
        
        # RuntimeError: 'lengths' argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor
        lengths = lengths.cpu().to(torch.int64)
        
        embedded = self.dropout(self.embedding(x)) # slen x bs x emb_dim       
        #pack sequence
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        # #RuntimeError: `lengths` array must be sorted in decreasing order when `enforce_sorted` is True. You can pass `enforce_sorted=False` to pack_padded_sequence and/or pack_sequence to sidestep this requirement if you do not need ONNX exportability.
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        
        #output, hidden = self.net(embedded) # slen x bs x hid_dim, 1 x bs x hid_dim
        packed_output, hidden = self.net(packed_embedded)
        
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        if not self.whole_output :
            if self.n_layers >= 2 and not self.bidirectional  :
                # TODO : correct it
                hidden = hidden[-1,:,:] # hidden[0] ?
            elif self.bidirectional  :
                #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers 
                hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1) # bs x hid_dim*num_directions
            else :
                #if self.n_layers < 2 and not self.bidirectional:
                assert torch.equal(output[-1,:,:], hidden.squeeze(0))
                hidden = hidden.squeeze(0)
        else :
            hidden = output.transpose(0, 1)
            
        return self.pred_layer(self.dropout(hidden), y, weights=weights)
    
class LSTMClassifier(RecurrentClassifier):
    def __init__(self, n_labels, params, logger, dico : Dictionary = None, with_output = True):
        super().__init__(n_labels, params, logger, dico, with_output)
        self.init_net(params)
    
    def init_net(self, params) :
        super().init_net(params)
        self.net = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers = self.n_layers, 
                            bidirectional=self.bidirectional, 
                            dropout = 0 if self.n_layers < 2 else params.dropout)
        
    def forward(self, x, lengths, y, positions=None, langs=None, weights = None, get_scores = True) :
        """
        Inputs:
            `x`        : LongTensor of shape (slen, bs)
        """
        # RuntimeError: 'lengths' argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor
        lengths = lengths.cpu().to(torch.int64)
        
        embedded = self.dropout(self.embedding(x)) # slen x bs x emb_dim       
        #pack sequence
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        # #RuntimeError: `lengths` array must be sorted in decreasing order when `enforce_sorted` is True. You can pass `enforce_sorted=False` to pack_padded_sequence and/or pack_sequence to sidestep this requirement if you do not need ONNX exportability.
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.net(packed_embedded)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output ~ slen x bs x hid_dim*num_directions  :  output over padding tokens are zero tensors
        #hidden = num_layers*num_directions x bs x hid_dim
        #cell = num_layers*num_directions x bs x hid_dim
        
        if not self.whole_output :
            if self.n_layers >= 2 and not self.bidirectional  :
                # TODO : correct it
                hidden = hidden[-1,:,:] # hidden[0] ?
            elif self.bidirectional  :
                #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers 
                hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1) # bs x hid_dim*num_directions
        else :
            hidden = output.transpose(0, 1)
        
        return self.pred_layer(self.dropout(hidden), y, weights=weights)

class ConvolutionalClassifier(SimpleClassifier):
    def __init__(self, n_labels, params, logger, dico : Dictionary = None, with_output = True):
        super().__init__(n_labels, params, logger, dico)
        
        self.with_output = with_output
        self.n_filters = params.n_filters
        self.filter_sizes = params.filter_sizes
        if with_output :
            self.hidden_dim = len(self.filter_sizes) * self.n_filters
            self.pred_layer = get_pred_layer(self.hidden_dim, n_labels, params).to(params.device)
        else :
            self.pred_layer = nn.Identity()
            
    def to_tensor(self, sentences):
        if not self.use_pretrained_word_embedding :
            return to_tensor(sentences, self.pad_index, dico = self.dico, batch_first = True)
        else :
            return to_tensor(sentences, self.pad_index, tokenize_and_cut = self.tokenize_and_cut, batch_first = True)
    
class CNNClassifier(ConvolutionalClassifier):
    def __init__(self, n_labels, params, logger, dico : Dictionary = None, with_output = True):
        super().__init__(n_labels, params, logger, dico, with_output)
        self.init_net(params)
    
    def init_net(self, params) :
        super().init_net(params)
        self.net = nn.ModuleList([
            nn.Conv2d(in_channels = 1, out_channels = self.n_filters, kernel_size = (fs, self.embedding_dim)) 
            for fs in self.filter_sizes
        ])
        
    def forward(self, x, lengths, y, positions=None, langs=None, weights = None, get_scores = True) :
        """
        Inputs:
            `x`        : LongTensor of shape (bs, slen)
        """
        embedded = self.dropout(self.embedding(x)) # bs x slen x emb_dim
        embedded = embedded.unsqueeze(dim=1) # bs x 1 x slen x emb_dim
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.net]
        #conved_n ~ bs x n_filters x (slen - emb_dim - filter_sizes[n] + 1)
                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = bs x n_filters
        
        if self.whole_output :
            # TODO : correct it 
            cat = self.dropout(torch.stack(pooled))
        else :
            cat = self.dropout(torch.cat(pooled, dim = 1)) # bs x n_filters * len(filter_sizes)
        
        return self.pred_layer(cat, y, weights=weights)
    
    def fwd(self, tensor, lengths, mask):
        print(tensor.shape)
        tensor = tensor.unsqueeze(dim=1) # bs x 1 x slen x emb_dim
        conved = [F.relu(conv(tensor)).squeeze(3) for conv in self.net] 
        print(conved[0].shape)       
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        tensor = self.dropout(torch.stack(pooled))
        print(tensor.shape)
        exit()
        return tensor
    
class CNN1dClassifier(ConvolutionalClassifier):
    def __init__(self, n_labels, params, logger, dico : Dictionary = None, with_output = True):
        super().__init__(n_labels, params, logger, dico, with_output)
        self.init_net(params)
    
    def init_net(self, params) :
        super().init_net(params)
        self.net = nn.ModuleList([
            nn.Conv1d(in_channels = self.embedding_dim, out_channels = self.n_filters, kernel_size = fs)
            for fs in self.filter_sizes
        ])
        
    def forward(self, x, lengths, y, positions=None, langs=None, weights = None, get_scores = True) :
        """
        Inputs:
            `x`        : LongTensor of shape (bs, slen)
        """
            
        embedded = self.dropout(self.embedding(x)) # bs x slen x emb_dim
        embedded = embedded.permute(0, 2, 1) # bs x emb_dim x slen
        
        conved = [F.relu(conv(embedded)) for conv in self.net]
        #conved_n ~ [bs x n_filters x (sent len - filter_sizes[n] + 1)
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = bs x n_filters
        
        if self.whole_output :
            # TODO : correct it 
            cat = self.dropout(torch.stack(pooled))
        else :
            cat = self.dropout(torch.cat(pooled, dim = 1)) # bs x n_filters * len(filter_sizes)
            
        return self.pred_layer(cat, y, weights=weights)

    def fwd(self, tensor, lengths, mask):
        tensor = tensor.permute(0, 2, 1)
        conved = [F.relu(conv(tensor)) for conv in self.net]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        return self.dropout(torch.stack(pooled))