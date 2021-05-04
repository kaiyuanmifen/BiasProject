# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Test : 6-layer, 512-hidden-nodes (64*8), 8-attention-heads (google bert : ___M parameters)
# BERT-Base: 12-layer, 768-hidden-nodes (64*12), 12-attention-heads (google bert : 110M parameters)
# BERT-Large: 24-layer, 1024-hidden-nodes (64*16), 16-attention-heads, (google bert :340M parameters)

import torch
import torch.nn.functional as F
import os

from sklearn.metrics import classification_report

from logging import getLogger

from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import initialize_exp, set_seeds
from src.utils import bool_flag

from src.classification.bias_classification import load_dataset, BertClassifier, GoogleBertClassifier, Trainer
from src.classification.metrics import get_stats, get_collect, get_score, get_acc, f1_score_func, iou_func,  mcc_func

from params import get_parser, from_config_file
    
def get_loss(model, batch, params, weights): 
    (x, lengths, langs), y1, y2 = batch
    
    if params.version in [1, 2, 3, 4] :    
        y = y2 if params.version == 3 else y1
        langs = langs.to(params.device) if params.n_langs > 1 else None
        logits, loss = model(x.to(params.device), lengths.to(params.device), y=y.to(params.device), langs=langs, weights = weights)
        #logits = F.softmax(logits, dim = -1)
        
        stats = {}
        n_words = lengths.sum().item()
        stats['n_words'] = n_words
        stats["avg_loss"] = loss.item()
        stats["loss"] = loss.item()
        stats["y1"] = y1.detach().cpu()#.numpy()
        
        s = get_stats(logits, y2, params)
        assert (s["label_pred"] == s["top%d_label_pred"%1]).all()
        stats = {**stats, **s}
        stats["logits"] = logits.detach().cpu()
        stats["y2"] = y2.cpu()
        
        if params.version in [2, 3] :
            inv_y2, inv_logits = logits.max(1)[1].view(-1).cpu(), y1.to(params.device)
            s = get_stats(inv_logits, inv_y2, params)
            #assert (s["label_pred"] == s["top%d_label_pred"%1]).all()
            for k in s.keys() :
                stats["inv_%s"%k] = s[k]
            stats["inv_logits"] = inv_logits.detach().cpu()
            stats["inv_y2"] = inv_y2.cpu()
            
            #stats["q_c"] = logits.detach().cpu()#.numpy()
            #stats["p_c"] = y1.detach().cpu()#.numpy()
    
    elif params.version in [5, 6, 7] :
        
        langs = langs.to(params.device) if params.n_langs > 1 else None
        scores, loss = model(x.to(params.device), lengths.to(params.device), y=y1.to(params.device), langs=langs, weights = weights)
        
        stats = {}
        n_words = lengths.sum().item()
        stats['n_words'] = n_words
        stats["avg_loss"] = loss.item()
        stats["loss"] = loss.item()
        if params.version != 7 : # y1 = y2
            stats["y1"] = y1.detach().cpu()#.numpy()
        stats["logits"] = scores.detach().cpu()
        stats["y2"] = y2.detach().cpu()
        
        if params.version in [5, 7] :
            label_pred = scores.detach().cpu().numpy()
            label_id = y2.cpu().numpy()
        elif params.version == 6 :
            batch_size = y1.size(0)
            b, c = y1[:,0], y1[:,1]
            y_true = (b * c / c.sum(dim=1).reshape(batch_size, -1)).sum(dim=1)
            y_true_bin = (y_true >= params.threshold).int().detach().cpu().numpy()
            stats["y2"] = y_true.detach().cpu()
            y_true = y_true.round().int()
            #print(y2.cpu().numpy(),  y_true.detach().cpu().numpy())
            #assert (y2.cpu().numpy() ==  y_true.detach().cpu().numpy()).all()
            label_id = y_true.detach().cpu().numpy()
            
            b, c = scores[0][:,0], scores[0][:,1]
            y_pred = (b * c / c.sum(dim=1).reshape(batch_size, -1)).sum(dim=1)
            y_pred_bin = (y_pred >= params.threshold).int().detach().cpu().numpy()
            y_pred = y_pred.round().int()
            label_pred = y_pred.detach().cpu().numpy()
            
            stats["bin_acc"] = get_acc(y_pred_bin, y_true_bin)
            stats["bin_f1_score_weighted"] = f1_score_func(y_pred_bin, y_true_bin)
            stats["bin_IoU_weighted"] = iou_func(y_pred_bin, y_true_bin)
            stats["bin_MCC"] = mcc_func(y_pred_bin, y_true_bin)
            
        stats["acc"] = get_acc(label_pred, label_id)
        stats["f1_score_weighted"] = f1_score_func(label_pred, label_id)
        stats["IoU_weighted"] = iou_func(label_pred, label_id)
        stats["MCC"] = mcc_func(label_pred, label_id)
    
    return loss, stats

def end_of_epoch(stats_list, val_first = True, add_output = False):
    scores = {}
    if params.version in [1, 2, 3, 4] :    
        for prefix, total_stats in zip(["val", "train"] if val_first else ["train","val"], stats_list):
            collect = get_collect(total_stats, params)
            
            l = len(collect['loss'])
            l = l if l != 0 else 1
            scores["%s_loss"%prefix] = sum(collect['loss']) / l
            
            s = get_score(collect, prefix, params, inv="", add_output=add_output) 
            scores = {**scores, **s}
            
            if params.version in [2, 3] :   
                s = get_score(collect, prefix, params, inv="inv_", add_output=add_output) 
                scores = {**scores, **s}
    
    elif params.version in  [5, 6, 7] :
        for prefix, total_stats in zip(["val", "train"] if val_first else ["train","val"], stats_list):
            collect = get_collect(total_stats, params)
            
            l = len(collect['loss'])
            l = l if l != 0 else 1
            scores["%s_loss"%prefix] = sum(collect['loss']) / l
            
            if params.version in [5, 7] :
                label_id = torch.cat(collect["y2"], dim=0).numpy()
                label_pred = torch.cat(collect["logits"], dim=0).numpy()
            
            if params.version == 6 :
                label_id = torch.cat(collect["y2"], dim=0)
                label_id_bin = (label_id >= params.threshold).int().numpy()
                label_id = label_id.round().int().numpy()
            
                for k in range(params.topK) :
                    label_pred = torch.cat([e[k] for e in collect["logits"]], dim=0)
                    b, c = label_pred[:,0], label_pred[:,1]
                    batch_size = c.size(0)
                    label_pred = (b * c / c.sum(dim=1).reshape(batch_size, -1)).sum(dim=1)
                    label_pred_bin = (label_pred >= params.threshold).int()
                    label_pred = label_pred.round().int().numpy()
                
                    scores["top_%s_%s_acc"%(k+1,prefix)] = get_acc(label_pred, label_id)
                    scores["top_%s_%s_f1_score_weighted"%(k+1,prefix)] = f1_score_func(label_pred, label_id)
                    scores["top_%s_%s_IoU_weighted"%(k+1,prefix)] = iou_func(label_pred, label_id)
                    scores["top_%s_%s_MCC"%(k+1,prefix)] = mcc_func(label_pred, label_id)
                    
                    scores["bin_top_%s_%s_acc"%(k+1,prefix)] = get_acc(label_pred_bin, label_id_bin)
                    scores["bin_top_%s_%s_f1_score_weighted"%(k+1,prefix)] = f1_score_func(label_pred_bin, label_id_bin)
                    scores["bin_top_%s_%s_IoU_weighted"%(k+1,prefix)] = iou_func(label_pred_bin, label_id_bin)
                    scores["bin_top_%s_%s_MCC"%(k+1,prefix)] = mcc_func(label_pred_bin, label_id_bin)

                label_pred = torch.cat([e[0] for e in collect["logits"]], dim=0)
                b, c = label_pred[:,0], label_pred[:,1]
                batch_size = c.size(0)
                label_pred = (b * c / c.sum(dim=1).reshape(batch_size, -1)).sum(dim=1)
                label_pred_bin = (label_pred >= params.threshold).int()
                label_pred = label_pred.round().int().numpy()

                scores["bin_%s_acc"%prefix] = get_acc(label_pred_bin, label_id_bin)
                scores["bin_%s_f1_score_weighted"%prefix] = f1_score_func(label_pred_bin, label_id_bin)
                scores["bin_%s_IoU_weighted"%prefix] = iou_func(label_pred_bin, label_id_bin)
                scores["bin_%s_MCC"%prefix] = mcc_func(label_pred_bin, label_id_bin)
            
            scores["%s_acc"%prefix] = get_acc(label_pred, label_id)
            scores["%s_f1_score_weighted"%prefix] = f1_score_func(label_pred, label_id)
            scores["%s_IoU_weighted"%prefix] = iou_func(label_pred, label_id)
            scores["%s_MCC"%prefix] = mcc_func(label_pred, label_id)
                
            report = classification_report(y_true = label_id, y_pred = label_pred, digits=4, output_dict=True, zero_division=0)
            
            m_v = {'precision': [], 'recall': [], 'f1-score':[]}
            num_classes = 2 if params.version in [5, 7] else 6
            if True :
                for k in report :
                    if k in list(range(num_classes)) :
                        v = report.get(k, None)
                        for k1 in m_v.keys():
                            m_v[k1] = m_v.get(k1, []) + [v[k1]]
                        scores["%s_class_%s"%(prefix, k)] = v
                    else :
                        scores["%s_%s"%(prefix, k)] = report.get(k, None)

    return scores

def main(params):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)
    
    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()
    
    # Model
    if not params.google_bert :
        model = BertClassifier(n_labels = params.n_labels, params = params, logger = logger)#.to(params.device)
    else :
        model = GoogleBertClassifier(n_labels = params.n_labels, params = params, logger = logger)#.to(params.device)
    logger.info(model)
    
    # Data 
    train_dataset, val_dataset = load_dataset(params, logger, model)
        
    # optimizers
    optimizers = model.get_optimizers(params) if not params.eval_only else []
    
    # Trainer
    trainer = Trainer(params, model, optimizers, train_dataset, val_dataset, logger)
    
    # Run train/evaluation
    logger.info("")
    if not params.eval_only :
        trainer.train(get_loss, end_of_epoch)
    else :
        trainer.eval(get_loss, end_of_epoch)
        
if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    
    parser.add_argument('--version', default=1, const=1, nargs='?',
                        choices=[1, 2, 3, 4, 5, 6, 7], 
                        help=  '1 : cross entropy(q_c, expected_label), expected_label = averaging the labels with the confidence scores as weights  \
                                2,3,4 ==> computed the coefficient of variation p_c among annotators for each sample in the dataset \
                                2 : bias_classif_loss(q_c, p_c), expected_label = arg_max(p_c) \
                                3 : cross_entropy(q_c, expected_label), expected_label = arg_max(p_c) \
                                4 : bias_classif_loss(q_c, p_c), expected_label = averaging the labels with the confidence scores as weights\
                                    q_c if the output logits give by the model')
    
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
    
    params = parser.parse_args()
    params = from_config_file(params)
    
    set_seeds(params.random_seed)

    if params.device not in ["cpu", "cuda"] :
        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else :
        params.device = torch.device(params.device)

    # check parameters
    if not params.google_bert :
        assert os.path.isfile(params.model_path) #or os.path.isfile(params.reload_checkpoint)
        assert os.path.isfile(params.codes) 
    
    # run experiment
    main(params)
    