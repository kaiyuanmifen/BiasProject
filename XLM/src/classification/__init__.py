# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn.functional as F

import os
from sklearn.metrics import classification_report

from .metrics import get_stats, get_collect, get_score, get_acc, f1_score_func, iou_func,  mcc_func
from .models import XLMBertClassifier, GoogleBertClassifier, RNNClassifier, LSTMClassifier, CNNClassifier, CNN1dClassifier
from .dataset import BiasClassificationDataset
from .utils import  init_pretrained_word_embedding, get_data_path

def build_model(params, logger, pre_trainer = None) :
    if params.model_name == "XLM" :
        model_class = XLMBertClassifier
        if pre_trainer is not None :
            model = model_class(n_labels = params.n_labels, params = params, logger = logger, pre_trainer = pre_trainer).to(params.device)
            logger.info("pred_layer")
            logger.info(model.pred_layer)
            return model
    elif params.model_name == "google_bert" :
        model_class = GoogleBertClassifier
    elif params.model_name == "RNN" :
        model_class = RNNClassifier
    elif params.model_name == "LSTM" :
        model_class = LSTMClassifier
    elif params.model_name == "GRU" :
        raise NotImplementedError('GRU is not implemented')
    elif params.model_name == "CNN" :
        model_class = CNNClassifier
    elif params.model_name == "CNN1d" :
        model_class = CNN1dClassifier
        
    model = model_class(n_labels = params.n_labels, params = params, logger = logger).to(params.device)
    logger.info(model)
    return model

def init_data_info(params, logger, model, n_samples, split) :
    data_info_path = os.path.join(params.dump_path, "data_info.pth")
    data_info_path = get_data_path(params, data_info_path, n_samples, split)
    params.data_info_path = data_info_path
    assert os.path.isfile(data_info_path) or not params.eval_only
    if os.path.isfile(data_info_path) :
        data_info = torch.load(data_info_path)
        if "dico" in data_info :
            dico = data_info["dico"]
            model.__init__(model.n_labels, params, logger, dico = dico)
            model.self_to(params.device)
            logger.info(model)
        if "corpus" in data_info :
            init_pretrained_word_embedding(model, sentences = data_info["corpus"], params = params, logger = logger)
            
def load_dataset(params, logger, model) :
    params.train_n_samples = None if params.train_n_samples==-1 else params.train_n_samples
    params.valid_n_samples = None if params.valid_n_samples==-1 else params.valid_n_samples
    init_data_info(params, logger, model, n_samples = params.train_n_samples, split = "train")
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
        if params.version == 6 :
            scores, loss, scores1 = model(x.to(params.device), lengths.to(params.device), y=y1.to(params.device), langs=langs, weights = weights)
        else :
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
            
            if scores1 is not None :
                y_pred_bin = scores1.detach().cpu()
                stats["y1"] = y_pred_bin
                y_pred_bin = y_pred_bin.numpy()
                stats["true_bin_acc"] = get_acc(y_pred_bin, y_true_bin)
                stats["true_bin_f1_score_weighted"] = f1_score_func(y_pred_bin, y_true_bin)
                stats["true_bin_IoU_weighted"] = iou_func(y_pred_bin, y_true_bin)
                stats["true_bin_MCC"] = mcc_func(y_pred_bin, y_true_bin)
            
            
        stats["acc"] = get_acc(label_pred, label_id)
        stats["f1_score_weighted"] = f1_score_func(label_pred, label_id)
        stats["IoU_weighted"] = iou_func(label_pred, label_id)
        stats["MCC"] = mcc_func(label_pred, label_id)
    
    return loss, stats

def end_of_epoch(stats_list, params, val_first = True, add_output = False):
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
                
                try :
                    label_pred_bin = torch.cat(collect["y1"], dim=0).numpy()
                    scores["true_bin_%s_acc"%prefix] = get_acc(label_pred_bin, label_id_bin)
                    scores["true_bin_%s_f1_score_weighted"%prefix] = f1_score_func(label_pred_bin, label_id_bin)
                    scores["true_bin_%s_IoU_weighted"%prefix] = iou_func(label_pred_bin, label_id_bin)
                    scores["true_bin_%s_MCC"%prefix] = mcc_func(label_pred_bin, label_id_bin)
                    
                    report = classification_report(y_true = label_id_bin, y_pred = label_pred_bin, digits=4, output_dict=True, zero_division=0)
                    m_v = {'precision': [], 'recall': [], 'f1-score':[]}
                    num_classes = 2
                    if True :
                        for k in report :
                            if k in list(range(num_classes)) :
                                v = report.get(k, None)
                                for k1 in m_v.keys():
                                    m_v[k1] = m_v.get(k1, []) + [v[k1]]
                                scores["true_bin_%s_class_%s"%(prefix, k)] = v
                            else :
                                scores["true_bin_%s_%s"%(prefix, k)] = report.get(k, None)
                except ValueError: #Classification metrics can't handle a mix of binary and
                    pass
            
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