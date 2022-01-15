# Copyright (c) 2019-present, Facebook, Inc.
# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch

from .transformer import TransformerModel
from ..data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from ..utils import AttrDict


logger = getLogger()

def get_layers_positions(layer_range, n_layers) :
    if layer_range == "" :
        return None, []
    if ":" in layer_range and not "," in layer_range :
        s = layer_range.split(':')
        assert len(s) == 2
        i, j = int(s[0].replace('_', '-')), int(s[1].replace('_', '-'))

        # negative indexing
        i = n_layers + i + 1 if i < 0 else i
        j = n_layers + j + 1 if j < 0 else j

        # sanity check
        assert 0 <= i <= n_layers
        assert 0 <= j <= n_layers

        if i > j:
            return None, []
                
        return i, range(max(i - 1, 0), j)
                
    elif not ":" in layer_range  : # and "," in layer_range
        layers_position = [int(i.replace('_', '-')) for i in layer_range.split(',')]
        # negative indexing
        layers_position = [n_layers + i + 1 if i < 0 else i for i in layers_position]
        # sanity check
        assert all([0 <= i <= n_layers for i in layers_position])
                
        layers_position = sorted(layers_position, reverse=False)
        i=layers_position[0]
        if i == 0 :
            del layers_position[0]
        return i, [i-1 for i in layers_position]
                
    else :
        layers_position = []
        for s in layer_range.split(',') :
            if not ':' in s :
                i = int(s.replace('_', '-'))
                i = n_layers + i + 1 if i < 0 else i
                assert 0 <= i <= n_layers
                layers_position.append(i)
            else :
                s = s.split(":")
                assert len(s) == 2
                i, j = int(s[0].replace('_', '-')), int(s[1].replace('_', '-'))
                # negative indexing
                i = n_layers + i + 1 if i < 0 else i
                j = n_layers + j + 1 if j < 0 else j
                # sanity check
                assert 0 <= i <= n_layers
                assert 0 <= j <= n_layers
                        
                layers_position.extend(list(range(i, j+1)))

        i=layers_position[0]
        if i == 0 :
            del layers_position[0]
        return i, [i-1 for i in layers_position]

class SentenceEmbedder(object):

    @staticmethod
    def reload(path, params):
        """
        Create a sentence embedder from a pretrained model.
        """
        # reload model
        reloaded = torch.load(path)
        state_dict = reloaded['model']

        # handle models from multi-GPU checkpoints
        if 'checkpoint' in path:
            state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

        # reload dictionary and model parameters
        dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
        pretrain_params = AttrDict(reloaded['params'])
        pretrain_params.n_words = len(dico)
        pretrain_params.bos_index = dico.index(BOS_WORD)
        pretrain_params.eos_index = dico.index(EOS_WORD)
        pretrain_params.pad_index = dico.index(PAD_WORD)
        pretrain_params.unk_index = dico.index(UNK_WORD)
        pretrain_params.mask_index = dico.index(MASK_WORD)
        
        if not hasattr(pretrain_params, "tim_layers_pos") :
            # For models trained when TIM was not yet integrated : for example XLM pre-trained by facebook AI
            setattr(pretrain_params, "dim_feedforward", pretrain_params.emb_dim*4) # https://github.com/facebookresearch/XLM/blob/master/xlm/model/transformer.py#L268
            setattr(pretrain_params, "use_mine", params.use_mine)
            setattr(pretrain_params, "tim_layers_pos", params.tim_layers_pos)

        # build model and reload weights
        model = TransformerModel(pretrain_params, dico, True, True)
        model.load_state_dict(state_dict)
        model.eval()

        # adding missing parameters
        params.max_batch_size = 0

        return SentenceEmbedder(model, dico, pretrain_params)

    def __init__(self, model, dico, pretrain_params):
        """
        Wrapper on top of the different sentence embedders.
        Returns sequence-wise or single-vector sentence representations.
        """
        self.pretrain_params = {k: v for k, v in pretrain_params.__dict__.items()}
        self.model = model
        self.dico = dico
        self.n_layers = model.n_layers
        self.out_dim = model.dim
        self.n_words = model.n_words

    def train(self, mode: bool = True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()

    def cuda(self):
        self.model.cuda()

    def get_parameters(self, layer_range, log=True):
        """layer_range=0:_1 ===> 0 = embeddings, _1 = last encoder layer
        Others example : 
            0,1,6 : embeddings, first encoder layer, 6th encoder layer
            0:4,6:8,11 : embeddings, 1-2-3-4th encoder layers,  6-7-8th encoder layers, 11 encoder layer
        """
        i, layers_position = get_layers_positions(layer_range, self.n_layers)
        if i is None :
            return []

        parameters = []

        # embeddings
        if i == 0:
            # embeddings
            parameters += self.model.embeddings.parameters()
            if log :
                logger.info("Adding embedding parameters to optimizer")
            # positional embeddings
            if self.pretrain_params['sinusoidal_embeddings'] is False:
                parameters += self.model.position_embeddings.parameters()
                if log :
                    logger.info("Adding positional embedding parameters to optimizer")
            # language embeddings
            if hasattr(self.model, 'lang_embeddings'):
                parameters += self.model.lang_embeddings.parameters()
                if log :
                    logger.info("Adding language embedding parameters to optimizer")
            parameters += self.model.layer_norm_emb.parameters()

        # layers
        if hasattr(self.model, 'tim_layers'):
            m, n = 0, 0
            for l in layers_position:
                if l in self.model.tim_layers_pos :
                    parameters += self.model.tim_layers[n].parameters()
                    n+=1
                else :
                    parameters += self.model.attentions[m].parameters()
                    parameters += self.model.layer_norm1[m].parameters()
                    parameters += self.model.ffns[m].parameters()
                    parameters += self.model.layer_norm2[m].parameters()
                    m+=1
                if log :
                    logger.info("Adding layer-%s parameters to optimizer" % (l + 1))
        else :
            for l in layers_position :
                parameters += self.model.attentions[l].parameters()
                parameters += self.model.layer_norm1[l].parameters()
                parameters += self.model.ffns[l].parameters()
                parameters += self.model.layer_norm2[l].parameters()
                if log :
                    logger.info("Adding layer-%s parameters to optimizer" % (l + 1))
        if log :
            logger.info("Optimizing on %i Transformer elements." % sum([p.nelement() for p in parameters]))

        return parameters

    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)
    
    def forward(self, *args, **kwds):
        return self.model.fwd(*args, **kwds)

    def predict(self, *args, **kwds):
        return self.model.predict(*args, **kwds)

    def get_embeddings(self, x, lengths, positions=None, langs=None, whole_output = False, do_sum=False):
        """
        Inputs:
            `x`        : LongTensor of shape (slen, bs)
            `lengths`  : LongTensor of shape (bs,)
        Outputs:
            `sent_emb` : FloatTensor of shape (bs, out_dim)
        With out_dim == emb_dim
        """
        slen, bs = x.size()
        assert lengths.size(0) == bs and lengths.max().item() == slen

        # get transformer last hidden layer
        tensor = self.model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)#.contiguous()
        assert tensor.size() == (slen, bs, self.out_dim)

        # single-vector sentence representation (first column of last layer)
        if do_sum :
            return tensor.sum(dim=0)
        return tensor[0] if not whole_output else tensor