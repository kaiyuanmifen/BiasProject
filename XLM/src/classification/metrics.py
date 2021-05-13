# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn.functional as F

from pytorch_lightning.metrics import IoU, AUROC, F1, Accuracy, AveragePrecision
try :
    from pytorch_lightning.metrics import HammingDistance
except ImportError : #cannot import name 'HammingDistance' from 'pytorch_lightning.metrics' (....\Anaconda3\lib\site-packages\pytorch_lightning\metrics\__init__.py)
    HammingDistance = None

from sklearn.metrics import f1_score, accuracy_score, jaccard_score, matthews_corrcoef, classification_report
from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

import seaborn as sns
import matplotlib.pyplot as plt

import os
import numpy as np

import warnings

def plot_conf(y_true, y_pred, label="", figsize=(7,4)) :
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)         # Sample figsize in inches
    fig.suptitle("confusion matrix %s"%label)
    f = sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

def get_acc(pred, label):
    """
    # This approach is exposed to this error for params.version == 7 : 
    # ~ https://github.com/gperftools/gperftools/issues/360
    # ~ https://support.microfocus.com/kb/doc.php?id=7012805
    # ~ local 
    #       >>> TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=21329330176  (failed on colab)
    #       >>> python classify.py ...
    """
    """
    arr = (np.array(pred) == np.array(label)).astype(float)
    if arr.size != 0: # check NaN 
        return arr.mean()*100
    return 0
    """
    return accuracy_score(y_true = label, y_pred = pred)*100
    
def f1_score_func(pred, label):
    # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
    return f1_score(y_true = label, y_pred = pred, average='weighted', labels=np.unique(pred))*100

def iou_func(pred, label):
    """Intersection over Union IoU scores : Jaccard similarity coefficient score"""
    return jaccard_score(y_true = label, y_pred = pred, average="weighted")*100

def mcc_func(pred, label):
    """Compute the Matthews correlation coefficient (MCC)"""
    return matthews_corrcoef(y_true = label, y_pred = pred)

def top_k(logits, y, k : int = 1):
    """
    logits : (bs, n_labels)
    y : (bs,) for multi-class or (bs, n_labels) for multi-label
    """
    labels_dim = 1
    assert 1 <= k <= logits.size(labels_dim)
    
    k_labels = torch.topk(input = logits, k = k, dim=labels_dim, largest=True, sorted=True)[1]

    if y.dim() == 2 :
        yy = torch.topk(input = y, k = k, dim=labels_dim, largest=True, sorted=True)[1]
        yy = torch.cat([torch.abs(yy[:,i].unsqueeze(labels_dim) - k_labels) for i in range(k)], dim=labels_dim)
    else :
        yy = torch.abs(y.unsqueeze(labels_dim) - k_labels)
    
    # True (#0) if `expected label` in k_labels, False (0) if not
    a = ~torch.prod(input = yy, dim=labels_dim).to(torch.bool)
    
    if y.dim() == 2 :
        bs = logits.size(0)
        correct = a.to(torch.int8).numpy()
        acc = sum(correct)/len(correct)*100
        #y = torch.ones((bs,), dtype=y.dtype)
        #a = a.to(torch.int8)
        #y_pred = a * y + (1-a) * torch.zeros((bs,), dtype=y.dtype)
        warnings.warn("`y.dim() == 2` is deprecated when need f1-score, iou and mcc.")
        return acc, 0, 0, 0, None
    else :
        # These two approaches are equivalent
        if False :
            y_pred = torch.empty_like(y)
            for i in range(y.size(0)):
                if a[i] :
                    y_pred[i] = y[i]
                else :
                    y_pred[i] = k_labels[i][0]
            #correct = a.to(torch.int8).numpy()
        else :
            a = a.to(torch.int8)
            y_pred = a * y + (1-a) * k_labels[:,0]
            #correct = a.numpy()
    
    y_pred = y_pred.cpu().numpy()
    f1 = f1_score_func(y_pred, y)
    acc = get_acc(y_pred, y) # sum(correct)/len(correct)*100
    iou = iou_func(y_pred, y)
    mcc = mcc_func(y_pred, y)
    
    return acc, f1, iou, mcc, y_pred

class Metrics():
    def __init__(self, device, num_classes, topK = 3) :
        
        self.device = device
        self.topK = topK
        
        # https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/classification/f_beta.py#L221
        #mdmc_average = "samplewise"
        mdmc_average = "global"
        
        val_metrics = {
            'hamming_dist': HammingDistance() if HammingDistance is not None else None, 
            'iou': IoU(num_classes=num_classes),
            'auroc': AUROC(num_classes=num_classes), 
            'f1': F1(num_classes=num_classes, multilabel=True, mdmc_average = mdmc_average), 
            'avg_precision': AveragePrecision(num_classes=num_classes),
            #'acc': Accuracy(num_classes=num_classes, mdmc_average = mdmc_average)
        }
        
        for k in range(1, topK+1):
            val_metrics["top%d"%k] = Accuracy(top_k=k) 
            val_metrics["top%d_f1"%k] = F1(top_k=k)
        
        self.val_metrics = torch.nn.ModuleDict(val_metrics).to(self.device)

        self.class_names = list(range(num_classes))
        self.label_binarizer = MultiLabelBinarizer(classes=self.class_names)

    def __call__(self, logits, targets, softmax=True) :
        logits = logits.to(self.device)
        targets = targets.to(self.device)
        
        if targets.dim() == 2 :
            y = targets.argmax(dim=1)
        else :
            y = targets.view(-1)
        
        if softmax :
            predictions = F.softmax(logits, dim=-1)
        else :
            predictions = logits
        
        results = {}

        for k in range(1, self.topK+1):
            results["top%d_acc"%k] = self.val_metrics['top%d'%k](predictions, y).item()*100
            results["top%d_f1_good"%k] = self.val_metrics["top%d_f1"%k](predictions, y).item()*100
            self.val_metrics['top%d'%k].reset()
            self.val_metrics["top%d_f1"%k].reset()

        # array of ap for each class
        avg_precision = self.val_metrics['avg_precision'](predictions, y) 
        try :
            results["APc"] = avg_precision.item()*100
            results["mAP"] = results["APc"] 
        except AttributeError : # 'list' object has no attribute 'item' 
            results["APc"] = [p.item()*100 for p in avg_precision]
            results["mAP"] = sum(results["APc"]) / len(results["APc"])
        self.val_metrics['avg_precision'].reset()

        if targets.dim() != 2 :
            return results

        topK = self.topK
        
        pred_matrix = [self.label_binarizer.fit_transform(predictions.cpu().topk(k=j).indices.numpy()) for j in range(1, topK+1)]
        targ_matrix = [self.label_binarizer.fit_transform(targets.cpu().topk(k=j).indices.numpy()) for j in range(1, topK+1)]

        # single value for each k in each metric below
        iou = [0] * topK
        hamming_dist = [0] * topK
        f1_scores = [0] * topK
        auroc = [0] * topK

        for i in range(topK):
            y_pred = torch.as_tensor(pred_matrix[i], device=self.device)
            y_true = torch.as_tensor(targ_matrix[i], device=self.device)
            
            iou[i] = self.val_metrics['iou'](y_pred, y_true).item()*100
            f1_scores[i] = self.val_metrics['f1'](y_pred, y_true).item()*100
            
            self.val_metrics['iou'].reset()
            self.val_metrics['f1'].reset()
            
            if HammingDistance is not None :
                hamming_dist[i] = self.val_metrics['hamming_dist'](y_pred, y_true).item()*100
                self.val_metrics['hamming_dist'].reset()
        
        results["iou"] = iou
        results["f1_scores"] = f1_scores
        if HammingDistance is not None :
            results["hamming_dist"] = hamming_dist
            
        return results

def get_stats(logits, y, params, inclure_pred=True, include_avg=True):
    stats = {}

    label_pred = logits.max(1)[1].view(-1).detach().cpu().numpy()
    label_id = y.view(-1).numpy()
    
    if inclure_pred :
        stats["label_pred"] = label_pred
        stats["label_id"] = label_id
        
    for k in range(1, params.topK+1):
        k_acc, k_f1, iou, mcc, y_pred = top_k(logits = logits.detach().cpu(), y=y, k=k)
        stats["top%d_acc"%k] = k_acc
        stats["top%d_f1_score"%k] = k_f1
        stats["top%d_IoU"%k] = iou
        stats["top%d_MCC"%k] = mcc
        if include_avg :
            stats["top%d_avg_IoU"%k] = iou
            stats["top%d_avg_acc"%k] = k_acc
            stats["top%d_avg_f1_score"%k] = k_f1
            stats["top%d_avg_MCC"%k] = mcc
        if inclure_pred :
            stats["top%d_label_pred"%k] = y_pred
        
    acc = get_acc(label_pred, label_id)
    f1 = f1_score_func(label_pred, label_id)
    iou = iou_func(label_pred, label_id)
    mcc = mcc_func(label_pred, label_id)
    
    stats["acc"] = acc
    stats["f1_score_weighted"] = f1
    stats["IoU_weighted"] = iou
    stats["MCC"] = mcc
    
    if include_avg :
        stats["avg_acc"] = acc
        stats["avg_f1_score_weighted"] = f1
        stats["avg_IoU_weighted"] = iou
        stats["avg_MCC"] = mcc

    return stats

def get_collect(total_stats, params):
    collect = {
        "loss" : [],
        "y1" : [],
        
        "label_pred" : [],
        "label_id" : [],
        "logits" : [],
        "y2" : [],
        
        "inv_label_pred" : [],
        "inv_label_id" : [],
        "inv_logits" : [],
        "inv_y2" : [],
        
        "label_pred_topK" : { k : [] for k in range(1, params.topK+1)},
        "inv_label_pred_topK" : { k : [] for k in range(1, params.topK+1)}
    }
        
    for stats in total_stats :
        """
        for key in collect.keys() :
            v = stats.get(key, [])
            if type(collect[key]) == list :
                if type(v) == list :
                    collect[key].extend(v)
                else :
                    collect[key].append(v)
        """
        collect["loss"].append(stats["loss"])
        collect["y1"].append(stats.get("y1", []))
        
        collect['label_pred'].extend(stats.get("label_pred", []))
        collect['label_id'].extend(stats.get("label_id", []))

        collect["logits"].append(stats["logits"])
        collect["y2"].append(stats["y2"])
        
        collect['inv_label_pred'].extend(stats.get("inv_label_pred", []))
        collect['inv_label_id'].extend(stats.get("inv_label_id", []))
        collect["inv_logits"].append(stats.get("inv_logits", None))
        collect["inv_y2"].append(stats.get("inv_y2", None))
        
        for k in range(1, params.topK+1):
            collect['label_pred_topK'][k].extend(stats.get("top%d_label_pred"%k, []))
            collect['inv_label_pred_topK'][k].extend(stats.get("inv_top%d_label_pred"%k, []))

    return collect

def get_score(collect, prefix, params, inv="", add_output = False) :
    scores = {}
    label_pred = collect["%slabel_pred"%inv]
    label_id = collect["%slabel_id"%inv]
    label_pred_topK = collect["%slabel_pred_topK"%inv]
    
    scores["%s_acc"%prefix] = get_acc(label_pred, label_id) #accuracy_score(label_pred, label_id)*100
    scores["%s_f1_score_weighted"%prefix] = f1_score_func(label_pred, label_id)
    scores["%s_IoU_weighted"%prefix] = iou_func(label_pred, label_id)
    scores["%s_MCC"%prefix] = mcc_func(label_pred, label_id)
        
    for k in range(1, params.topK+1):
        scores["fake_top%d_%s_acc"%(k, prefix)] = get_acc(label_pred_topK[k], label_id)
        scores["fake_top%d_%s_f1_score_weighted"%(k, prefix)] = f1_score_func(label_pred_topK[k], label_id)
        scores["fake_top%d_%s_IoU_weighted"%(k, prefix)] = iou_func(label_pred_topK[k], label_id)
        scores["fake_top%d_%s_MCC"%(k, prefix)] = mcc_func(label_pred_topK[k], label_id)
        
    report = classification_report(y_true = label_id, y_pred = label_pred, digits=4, output_dict=True, zero_division=0)
        
    m_v = {'precision': [], 'recall': [], 'f1-score':[]}
    if True :
        for k in report :
            if k in list(range(params.n_labels)) :
                v = report.get(k, None)
                for k1 in m_v.keys():
                    m_v[k1] = m_v.get(k1, []) + [v[k1]]
                scores["%s_class_%s"%(prefix, k)] = v
            else :
                scores["%s_%s"%(prefix, k)] = report.get(k, None)
            
        # AP
        for k in m_v.keys():
            v = m_v.get(k, [0.0])
            l = len(v)
            #l = l if l != 0 else 1
            if l != 0 :
                scores["%s_mean_average_%s"%(prefix, k)] = sum(v) / l*100
    
    y2 = torch.cat(collect["%sy2"%inv], dim=0)
    logits = torch.cat(collect["%slogits"%inv], dim=0)
    #print("logits", prefix, inv, logits)
    if add_output :
        scores["%sy2_%s"%(inv, prefix)] = y2
        scores["%slogits_%s"%(inv, prefix)] = logits
    s = get_stats(logits, y2, params, inclure_pred=False, include_avg=False)
    for k, v in s.items() :
        scores["%strue_%s_%s"%(inv, prefix, k)] = v 

    try :
        pl_metrics = Metrics(device = params.device, num_classes = params.n_labels, topK = params.topK)
        #pl_metrics = None
    except :
        pl_metrics = None

    if pl_metrics is not None :
        s = pl_metrics(logits, targets = y2)
        for k, v in s.items() :
            scores["%spl_multiclass_%s_%s"%(inv, prefix, k)] = v 

    if inv == "" :
        y1 = torch.cat(collect["y1"], dim=0)
        #print("y1", prefix, inv, y1)
        if add_output :
            scores["%sy1_%s"%(inv, prefix)] = y1
        if pl_metrics is not None :
            s = pl_metrics(logits, targets = y1)
            for k, v in s.items() :
                scores["%spl_multilabel_%s_%s"%(inv, prefix, k)] = v 
            
        for k in range(1, params.topK+1):
            k_acc, _, _, _, _ = top_k(logits = logits.detach().cpu(), y=y1, k=k)
            scores["%ssklearn_multilabel_top%d_%s_acc"%(inv, k, prefix)] = k_acc

    if inv != "" :
        keys = scores.keys()
        for k in keys :
            if not inv in k :
                scores["%s%s"%(inv, k)] = scores.pop(k)
    
    return scores