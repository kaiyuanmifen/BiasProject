# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

from torch._C import default_generator
from ..utils import bool_flag
def __main__(params) :
    params.use_pretrained_word_embedding = False
    #params.simple_model = "model_name:RNN,emb_dim:100,use_pretrained_word_embedding:,train_embedding:True,tokenize:spacy,hidden_dim:256,n_layers:2,bidirectional:True"
    #params.simple_model = "model_name:RNN,emb_dim:100,use_pretrained_word_embedding:glove.6B.50d,train_embedding:True,tokenize:spacy,hidden_dim:256,n_layers:2,bidirectional:True"
    #params.simple_model = "model_name:CNN,emb_dim:100,use_pretrained_word_embedding:,train_embedding:True,tokenize:spacy,hidden_dim:256,n_filters:100,filter_sizes:3-4-5"
    #params.simple_model = "model_name:CNN,emb_dim:100,use_pretrained_word_embedding:glove.6B.50d,hidden_dim:256,train_embedding:True,tokenize:spacy,n_filters:100,filter_sizes:3-4-5"
    
    if params.simple_model :
        #params.max_vocab_size = None
        pretrained_word_embedding = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d',
                                    'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 
                                    'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']
        pattern = {"model_name":[str, ""], "emb_dim":[int, 100], "use_pretrained_word_embedding":[str, ""], 
                    "hidden_dim":[int, 256], "n_layers":[int, 2], "bidirectional":[bool_flag, True], 
                    "n_filters":[int, 100], "filter_sizes":[list, [3,4,5]], "tokenize" : [str, 'spacy'], 
                    "max_vocab_size":[int, None], "train_embedding" : [bool_flag, True]}
        model_params = {p[0] : p[1] for p in [p.split(":") for p in params.simple_model.split(",")]}
        assert "model_name" in model_params
        assert all([k in pattern.keys() for k in model_params.keys()])
        if "filter_sizes" in model_params :
            model_params["filter_sizes"] = [int(fs) for fs in model_params["filter_sizes"].split("-")]
        if False :
            for k, v in model_params.items() :
                setattr(params, k, pattern[k][0](v)) 
        else :
            for k, v in pattern.items() :
                #setattr(params, k, v[0](model_params.get(k, v[1])))  
                setattr(params, k, v[0](model_params[k]) if k in model_params else v[1])  
        assert not params.use_pretrained_word_embedding or params.use_pretrained_word_embedding in pretrained_word_embedding
        #delattr(params, "simple_model")
        
def add_argument(parser) :
    parser.add_argument('--version', default=1, const=1, nargs='?',
                        choices=[1, 2, 3, 4, 5, 6, 7], 
                        help=  '1 : cross entropy(q_c, expected_label), expected_label = averaging the labels with the confidence scores as weights  \
                                2,3,4 ==> computed the coefficient of variation p_c among annotators for each sample in the dataset \
                                2 : bias_classif_loss(q_c, p_c), expected_label = arg_max(p_c) \
                                3 : cross_entropy(q_c, expected_label), expected_label = arg_max(p_c) \
                                4 : bias_classif_loss(q_c, p_c), expected_label = averaging the labels with the confidence scores as weights\
                                    q_c if the output logits give by the model \
                                5 : TODO \
                                6 : TODO \
                                7 : TODO')
    
    #if parser.parse_known_args()[0].version == 2:
    parser.add_argument("--log_softmax", type=bool_flag, default=True, 
                        help="use log_softmax in the loss function instead of log")

    parser.add_argument("--train_data_file", type=str, default="", help="file (.csv) containing the data")
    parser.add_argument("--val_data_file", type=str, default="", help="file (.csv) containing the data")
    parser.add_argument("--data_columns", type=str, default="",
                        help="content,scores_columns1-scores_columns2...,confidence_columns1-confidence_columns2...")
    parser.add_argument("--n_labels", type=int, default=6, 
                        help="number of labels in the dataset: useful for \
                            the output of the classification layer.")
    
    parser.add_argument("--in_memory", type=bool_flag, default=True, 
                        help="")
    parser.add_argument("--do_augment", type=bool_flag, default=False, 
                        help="EDA text augmentation")
    parser.add_argument("--do_downsampling", type=bool_flag, default=False, 
                        help="Downsampling the majority")
    parser.add_argument("--do_upsampling", type=bool_flag, default=False, 
                        help="Upsampling the minority")
    parser.add_argument("--threshold", type=float, default=2.5, 
                        help="threshold : 3 is possible, to avoid unbalanced data and to reduce false negatives")
    
    parser.add_argument("--shuffle", type=bool_flag, default=True, help="shuffle Dataset across epoch")
    #parser.add_argument("--group_by_size", type=bool_flag, default=True, help="Sort sentences by size during the training")
    
    parser.add_argument("--codes", type=str, default="", help="path of bpe code")
    parser.add_argument("--vocab", type=str, default="", help="path of bpe vocab")
    
    parser.add_argument("--min_len", type=int, default=1, 
                        help="minimun sentence length before bpe in training set")
    
    parser.add_argument('--debug_num', default=0, const=0, nargs='?',
                        choices=[0, 1, 2], 
                        help=  '0 : Transformer + Linear \
                                1 : Transformer + Linear + Tanh + Dropout + Linear \
                                2 : Transformer + GRU + Dropout + Linear')
    #if parser.parse_known_args()[0].debug_num in [0, 2] :
    parser.add_argument("--hidden_dim", type=int, default=-1, 
                        help="hidden dimension of classifier")
    
    # GRU
    #if parser.parse_known_args()[0].debug_num == 2 :
    parser.add_argument("--gru_n_layers", type=int, default=1, 
                        help="number of layers, GRU")
    parser.add_argument("--bidirectional", type=bool_flag, default=False, help="bidirectional GRU or not")
    
    #parser.add_argument('--topK', default=3, const=3, nargs='?', choices=[1, 2, 3, 4, 5, 6], help="topK")
    parser.add_argument("--topK", type=int, default=3, help="topK")

    parser.add_argument("--model_path", type=str, default="", 
                        help="Reload transformer model from pretrained model / dico / ...")
    
    parser.add_argument("--finetune_layers", type=str, default='', 
                        help="Layers to finetune. default='' ==> freeze the transformer encoder part of the model \
                            0:_1 or 0:-1 ===> fine_tune all the transformer model (0 = embeddings, _1 = last encoder layer) \
                            0,1,6 or 0:1,6 ===> embeddings, first encoder layer, 6th encoder layer \
                            0:4,6:8,11 ===> embeddings, 1-2-3-4th encoder layers,  6-7-8th encoder layers, 11th encoder layer \
                            Do not include any symbols other than numbers ([0, n_layers]), comma (,) and two points (:)\
                            This supports negative indexes ( _1 or -1 refers to the last layer for example)")
    parser.add_argument("--weighted_training", type=bool_flag, default=False,
                        help="Use a weighted loss during training")
    parser.add_argument("--weighted_out", type=bool_flag, default=False,
                        help="Use a weighted out in loss during training")
    #parser.add_argument("--dropout", type=float, default=0, help="Fine-tuning dropout")
    parser.add_argument("--optimizer_e", type=str, default="adam,lr=0.0001",
                        help="Embedder (pretrained model) optimizer")
    parser.add_argument("--optimizer_p", type=str, default="adam,lr=0.0001",
                        help="Projection (classifier) optimizer")
    
    parser.add_argument("--google_bert", type=bool_flag, default=False,
                        help="Use bert modele pre-trained from google (will be downloaded automatically \
                            thanks to the huggingface transformers library) ")
    #if parser.parse_known_args()[0].google_bert :
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased",
                        help="type of bert model to use : bert-base-uncased, bert-base-cased, ...")
    
    parser.add_argument("--simple_model", type=str, default="",
                        help="RNN/LSTM/CNN/CNN1d : \
                        model_name:RNN,emb_dim:100,use_pretrained_word_embedding:,train_embedding:True,tokenize:spacy,hidden_dim:256,n_layers:2,bidirectional:True \
                        model_name:CNN,emb_dim:100,use_pretrained_word_embedding:,train_embedding:True,tokenize:spacy,hidden_dim:256,n_filters:100,filter_sizes:3-4-5")
    
    parser.add_argument("--pretrain_config", type=str, default="", help="TODO")
    parser.add_argument('--pretrain_type', default=0, const=0, nargs='?',
                        choices=[0, 1], 
                        help=  "0 : MLM step + Classif step \
                                1 : MLM loss + Classif loss")
    import re
    import argparse
    def cv(arg_value):
        if arg_value == "" :
            return arg_value

        int_ = "([1-9]\d*)"
        _prob = "0.([1-9]\d*|0*[1-9]\d*)"
        int_or_prob="(%s|%s)"%(int_, _prob)
        if re.match(pattern="^holdout:test_size=%s$"%int_or_prob, string=arg_value): # holdout
            return arg_value
        elif arg_value=="leave-one-out" or re.match(pattern="^leave-one-out:p=%s$"%int_or_prob, string=arg_value) : # leave-one-out
            return arg_value
        elif re.match(pattern="^%s-fold$"%int_or_prob, string=arg_value):  # k-fold
            return arg_value     
        elif re.match(pattern="^repeated-%s-fold:n_repeats=([1-9]\d*)$"%int_or_prob, string=arg_value):  # repeated-k-fold:n_repeats=...
            return arg_value  
        elif re.match(pattern="^leave-%s-out$"%int_or_prob, string=arg_value):  # leave-p-out
            return arg_value   
        elif re.match(pattern="^shuffle-split:p=%s,test_size=%s$"%(int_or_prob, int_or_prob), string=arg_value):  # shuffle-split:test_size=0....
            return arg_value
        else :
            raise argparse.ArgumentTypeError

    parser.add_argument('--cross_validation', type=cv, default="", 
                        help=", holdout:test_size=0.2, leave-one-out, leave-one-out:p=0.2, 0.2-fold, 1000-fold, repeated-0.2-fold:n_repeats=2, \
                            leave-0.2-out, leave-1000-out, shuffle-split:p=0.2,test_size=0.2")

    parser.add_argument('--outliers', type=float, default=0, help="")
    
    parser.add_argument('--yoshua', type=bool_flag, default=False, help="pred_score = sum_i i * softmax_logits(i)")

    return parser

def check_parameters(params) :
    assert not (params.google_bert and params.simple_model)
    
    __main__(params)
        
    if not params.google_bert :
        if params.simple_model :
            #assert os.path.isfile(params.vocab) or params.use_pretrained_word_embedding
            assert params.model_name in ['RNN', 'LSTM', 'CNN', 'CNN1d'] # TODO : GRU
        else : # XLM
            assert os.path.isfile(params.model_path) or params.pretrain #or os.path.isfile(params.reload_checkpoint)
            assert os.path.isfile(params.codes)
            params.model_name = "XLM"
    else :
        params.model_name = "google_bert"

    assert 0 <= params.outliers < 1