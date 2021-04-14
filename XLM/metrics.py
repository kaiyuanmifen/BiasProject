import torch
import torch.nn.functional as F

import os
import numpy as np

from sklearn.metrics import f1_score, accuracy_score, jaccard_score, matthews_corrcoef, classification_report
from sklearn.preprocessing import LabelBinarizer

from pytorch_lightning.metrics import IoU, AUROC, F1, Accuracy, AveragePrecision

try :
    from pytorch_lightning.metrics import HammingDistance
except ImportError : #cannot import name 'HammingDistance' from 'pytorch_lightning.metrics' (....\Anaconda3\lib\site-packages\pytorch_lightning\metrics\__init__.py)
    HammingDistance = None

from src.bias_classification import label_dict

def get_acc(pred, label):
    arr = (np.array(pred) == np.array(label)).astype(float)
    if arr.size != 0: # check NaN 
        return arr.mean()*100
    return 0

def f1_score_func(pred, label):
    # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
    return f1_score(y_true = label, y_pred = pred, average='weighted', labels=np.unique(pred))*100

def iou_func(pred, label):
    """Intersection over Union IoU scores : Jaccard similarity coefficient score"""
    return jaccard_score(y_true = label, y_pred = pred, average="weighted")*100

def top_k(logits, y, k : int = 1):
    """
    logits : (bs, n_labels)
    y : (bs,)
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
        y = torch.ones((bs,), dtype=y.dtype)
        #a = a.to(torch.int8)
        #y_pred = a * y + (1-a) * torch.zeros((bs,), dtype=y.dtype)
        
        return acc, 0, 0, y
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
    #f1 = f1_score(y_pred, y, average='weighted')*100
    f1 = f1_score_func(y_pred, y)
    #acc = sum(correct)/len(correct)*100
    #acc = accuracy_score(y_pred, y)*100
    acc = get_acc(y_pred, y)
    
    iou = iou_func(y_pred, y)
    
    return acc, f1, iou, y_pred

class Metrics():
    def __init__(self, device, topK = 3) :
        
        self.val_metrics = torch.nn.ModuleDict({
            'hamming_dist': HammingDistance() if HammingDistance is not None else None, 
            'iou': IoU(num_classes=6),
            'auroc': AUROC(num_classes=6), 
            'f1': F1(num_classes=6, multilabel=True),
            'top1': Accuracy(top_k=1), 
            'top2': Accuracy(top_k=2), 
            'top3': Accuracy(top_k=3), 
            'avg_precision': AveragePrecision(num_classes=6)
        })

        self.label_binarizer = LabelBinarizer()
        self.class_names = [0, 1, 2, 3, 4, 5]
        self.label_binarizer.fit(self.class_names)

        self.device = device

        self.topK = topK

    def __call__(self, logits, targets, softmax=False) :
        if targets.dim() == 2 :
            y = targets.argmax(dim=1)
        else :
            y = targets.view(-1)
        
        if softmax :
            predictions = F.softmax(logits, dim=-1)
        else :
            predictions = logits
        
        results = {}

        results["top1_acc"] = self.val_metrics['top1'](predictions, y).item()
        results["top2_acc"] = self.val_metrics['top2'](predictions, y).item()
        results["top3_acc"] = self.val_metrics['top3'](predictions, y).item()
        self.val_metrics['top1'].reset()
        self.val_metrics['top2'].reset()
        self.val_metrics['top3'].reset()

        # array of ap for each class
        avg_precision = self.val_metrics['avg_precision'](predictions, y) 
        try :
            results["APc"] = avg_precision.item()
            results["mAP"] = results["APc"] 
        except AttributeError : # 'list' object has no attribute 'item' 
            results["APc"] = [p.item() for p in avg_precision]
            results["mAP"] = sum(results["APc"]) / len(results["APc"])
        self.val_metrics['avg_precision'].reset()

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
            iou[i] = self.val_metrics['iou'](torch.as_tensor(pred_matrix[i], device=self.device), torch.as_tensor(targ_matrix[i], device=self.device))
            self.val_metrics['iou'].reset()
            f1_scores[i] = self.val_metrics['f1'](torch.as_tensor(pred_matrix[i], device=self.device), torch.as_tensor(targ_matrix[i], device=self.device))
            self.val_metrics['f1'].reset()
            
            if HammingDistance is not None :
                hamming_dist[i] = self.val_metrics['hamming_dist'](torch.as_tensor(pred_matrix[i], device=self.device), torch.as_tensor(targ_matrix[i], device=self.device))
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
        k_acc, k_f1, iou, y_pred = top_k(logits = logits.detach().cpu(), y=y, k=k)
        stats["top%d_acc"%k] = k_acc
        stats["top%d_f1_score"%k] = k_f1
        stats["top%d_IoU"%k] = iou
        if include_avg :
            stats["top%d_avg_IoU"%k] = iou
            stats["top%d_avg_acc"%k] = k_acc
            stats["top%d_avg_f1_score"%k] = k_f1
        if inclure_pred :
            stats["top%d_label_pred"%k] = y_pred
        
    acc = get_acc(label_pred, label_id)
    f1 = f1_score_func(label_pred, label_id)
    iou = iou_func(label_pred, label_id)
    
    stats["acc"] = acc
    stats["f1_score_weighted"] = f1
    stats["IoU_weighted"] = iou
    
    if include_avg :
        stats["avg_acc"] = acc
        stats["avg_f1_score_weighted"] = f1
        stats["avg_IoU_weighted"] = iou

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
        collect["y1"].append(stats["y1"])
        
        collect['label_pred'].extend(stats["label_pred"])
        collect['label_id'].extend(stats["label_id"])

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

def get_score(collect, prefix, params, inv="", pl_metrics = None) :
    scores = {}
    label_pred = collect["%slabel_pred"%inv]
    label_id = collect["%slabel_id"%inv]
    label_pred_topK = collect["%slabel_pred_topK"%inv]
    
    scores["%s_acc"%prefix] = get_acc(label_pred, label_id) #accuracy_score(label_pred, label_id)*100
    scores["%s_f1_score_weighted"%prefix] = f1_score_func(label_pred, label_id)
    scores["%s_IoU_weighted"%prefix] = iou_func(label_pred, label_id)
        
    for k in range(1, params.topK+1):
        scores["fake_top%d_%s_acc"%(k, prefix)] = get_acc(label_pred_topK[k], label_id)
        scores["fake_top%d_%s_f1_score_weighted"%(k, prefix)] = f1_score_func(label_pred_topK[k], label_id)
        scores["fake_top%d_%s_IoU_weighted"%(k, prefix)] = iou_func(label_pred_topK[k], label_id)
        
    report = classification_report(y_true = label_id, y_pred = label_pred, digits=4, output_dict=True, zero_division=0)
        
    m_v = {'precision': [], 'recall': [], 'f1-score':[]}
    if True :
        for k in report :
            if k in label_dict.keys() :
                v = report.get(k, None)
                for k1 in m_v.keys():
                    m_v[k1] = m_v.get(k1, []) + [v[k1]]
                scores["%s_class_%s"%(prefix, k)] = v
            else :
                scores["%s_%s"%(prefix, k)] = report.get(k, None)
            
        # AP
        for k in m_v.keys():
            v = m_v.get(k, [0.0])
            scores["%s_mean_average_%s"%(prefix, k)] = sum(v) / len(v)*100
    
    y2 = torch.cat(collect["%sy2"%inv], dim=0)
    logits = torch.cat(collect["%slogits"%inv], dim=0)
    s = get_stats(logits, y2, params, inclure_pred=False, include_avg=False)
    for k, v in s.items() :
        scores["true_%s_%s"%(prefix, k)] = v 
        
    pl_metrics = Metrics(device = params.device, topK = params.topK)

    s = pl_metrics(logits, targets = y2)
    for k, v in s.items() :
        scores["%spl_multiclass_%s_%s"%(inv, prefix, k)] = v 

    if inv == "" :
        y1 = torch.cat(collect["y1"], dim=0)
        s = pl_metrics(logits, targets = y1)
        for k, v in s.items() :
            scores["%spl_multilabel_%s_%s"%(inv, prefix, k)] = v 
            
        for k in range(1, params.topK+1):
            k_acc, _, _, _ = top_k(logits = logits.detach().cpu(), y=y1, k=k)
            scores["%smultilabel_top%d_%s_acc"%(inv, k, prefix)] = k_acc

    if inv != "" :
        keys = scores.keys()
        for k in keys :
            if not inv in k :
                scores["%s%s"%(inv, k)] = scores.pop(k)
    
    return scores