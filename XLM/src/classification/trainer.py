# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit, StratifiedShuffleSplit
import os
from tqdm import tqdm
import re
import itertools
import gc
import time
from logging import getLogger
from collections import OrderedDict
import json

#git clone https://github.com/NVIDIA/apex
#pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
#import apex
#############

from .utils import path_leaf
from ..utils import to_cuda
from .. import one_step, end_of_epoch as pretrainer_eoe

logger = getLogger()

def cross_validatation(X, y = None, shuffle : bool = False, random_state = None, kwargs = {}) :
    """https://scikit-learn.org/stable/modules/cross_validation.html"""
    p = kwargs["p"]
    cv_type = kwargs.get("cv_type", "k-fold")
    l = len(X)
    if type(p) == int :
        assert 1 <= p <= l, "p = %s, p/l = %s"%(p, p/l)
        n_splits = p
        p = p/l
    else : # float
        assert 0 < p < 1, "p = %s"%(p)
        n_splits = min(int(round(1/p)), l) 
        n_splits = l - n_splits if p > 0.5 else n_splits

    if cv_type == "k-fold" :
        if y is None :
            cv = KFold(n_splits=n_splits, shuffle = shuffle, random_state = random_state).split(X, y=y)
        else :
            try :
                next(StratifiedKFold(n_splits=n_splits, shuffle = shuffle, random_state = random_state).split(X, y=y))
            except ValueError: #n_splits=? cannot be greater than the number of members in each class.
                cv = KFold(n_splits=n_splits, shuffle = shuffle, random_state = random_state).split(X, y)
            else :
                cv = StratifiedKFold(n_splits=n_splits, shuffle = shuffle, random_state = random_state).split(X, y=y)
    elif cv_type == "repeated-k-fold":
        if y is None :
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=kwargs.get("n_repeats", 1), random_state=random_state).split(X, y=y)
        else :
            try :
                next(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=kwargs.get("n_repeats", 1), random_state = random_state).split(X, y=y))
            except ValueError: #n_splits=? cannot be greater than the number of members in each class.
                cv = RepeatedKFold(n_splits=n_splits, n_repeats=kwargs.get("n_repeats", 1), random_state=random_state).split(X, y=y)
            else :
                cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=kwargs.get("n_repeats", 1), random_state = random_state).split(X, y=y)  
    elif cv_type == "leave-one-out" :
        cv = LeaveOneOut().split(X, y=y)
    elif cv_type == "leave-p-out" :
        cv = LeavePOut(p=kwargs.get("p", 1)).split(X, y=y)
    elif cv_type == "shuffle-split" :
        if y is None :
            cv = ShuffleSplit(n_splits=n_splits, test_size=kwargs.get("test_size", 0.2), random_state=random_state).split(X, y=y)
        else :
            try :
                next(StratifiedShuffleSplit(n_splits=n_splits, test_size=kwargs.get("test_size", 0.2), random_state=random_state).split(X, y=y))
            except ValueError: #n_splits=? cannot be greater than the number of members in each class.
                cv = ShuffleSplit(n_splits=n_splits, test_size=kwargs.get("test_size", 0.2), random_state=random_state).split(X, y=y)
            else :
                cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=kwargs.get("test_size", 0.2), random_state=random_state).split(X, y=y)

    # TODO : GroupKFold, LeaveOneGroupOut, LeavePGroupsOut, GroupShuffleSplit
    else :
        raise NotImplementedError("cv_type = %s is not implemented"%cv_type)

    for train_index, test_index in cv :
        if p > 0.5 :
            yield test_index, train_index
            #yield X[test_index], X[train_index]
        else :
            yield train_index, test_index
            #yield X[train_index], X[test_index]

def to_type(p : str):
  if "." in p :
      return float(p)
  else :
      return int(p)
      
def get_cv_kwargs(cv_type : str, data : None) -> dict:
    int_ = "([1-9]\d*)"
    _prob = "0.([1-9]\d*|0*[1-9]\d*)"
    int_or_prob="(%s|%s)"%(int_, _prob)
    if re.match(pattern="^holdout:test_size=%s$"%int_or_prob, string=cv_type): # holdout
        test_size = to_type(cv_type.split("=")[1])
        if type(test_size) == int :
            assert data is not None
            l = len(data)
            assert test_size <= l, "test_size > len(data)"
            test_size = test_size/l

        return {"cv_type": "holdout", "test_size": test_size}
    elif cv_type=="leave-one-out" or re.match(pattern="^leave-one-out:p=%s$"%int_or_prob, string=cv_type) : # leave-one-out
        return {"cv_type":"leave-one-out", "p": 0.1 if cv_type=="leave-one-out" else  to_type(cv_type.split("=")[1]) }
    elif re.match(pattern="^%s-fold$"%int_or_prob, string=cv_type) : # k-fold
        p = to_type(cv_type.split("-")[0])
        return {"cv_type":"k-fold", "p": p}
    elif re.match(pattern="^repeated-%s-fold:n_repeats=([1-9]\d*)$"%int_or_prob, string=cv_type) : # repeated-k-fold
        p = to_type(cv_type.split("-")[1])
        n_repeats = int(cv_type.split("=")[1])
        return {"cv_type":"repeated-k-fold", "p": p, "n_repeats": n_repeats}
    elif re.match(pattern="^leave-%s-out$"%int_or_prob, string=cv_type) : # leave-p-out
        p=to_type(cv_type.split("-")[1])
        if type(p) == float :
            p = round(p*len(data))
        if data is not None :
            assert p <= len(data) 
        if p == 1 :
             return {"cv_type":"leave-one-out", "p": p}
        else :
            return {"cv_type":"leave-p-out", "p": p}
    elif re.match(pattern="^shuffle-split:p=%s,test_size=%s$"%(int_or_prob, int_or_prob), string=cv_type) : # shuffle-split
        tmp = cv_type.split("=")
        p=to_type(tmp[1].split(',')[0])
        test_size=to_type(tmp[-1])
        if type(test_size) == int :
            assert data is not None
            l = len(data)
            assert test_size <= l, "test_size > len(data)"
            test_size = test_size/l
        return {"cv_type":"shuffle-split", "p": p, "test_size": test_size}
    else :
        raise NotImplementedError("cv_type = %s is not implemented"%cv_type)

class Trainer(object):
    """Training Helper Class"""
    def __init__(self, params, model, optimizers, train_data_iter, val_data_iter, logger, pre_trainer = None, evaluator = None):
        self.params = params
        self.model = model
        self.optimizers = optimizers # optim

        # iterator to load data
        self.train_data_iter = train_data_iter 
        self.val_data_iter = val_data_iter 

        self.device = params.device # device name
        self.logger = logger
        self.pre_trainer = pre_trainer
        self.evaluator = evaluator

        # epoch / iteration size
        self.epoch_size = self.params.epoch_size
        if self.epoch_size == -1 and not params.eval_only:
            self.epoch_size = self.params.train_num_data
        assert self.epoch_size > 0 or params.eval_only
        
        # add metrics and topK to possible metrics
        possib = []
        #possib = ["%s_%s_%s"%(i, j, k) for i, j, k in itertools.product(["train", "val"], ["mlm", "nsp"], ["ppl", "acc", "loss"])]
        possib.extend(["%s_%s"%(i, j) for i, j in itertools.product(["train", "val"], ["f1_score_weighted", "acc", "loss", "IoU_weighted", "MCC"])])
        tmp = []
        for k in range(1, params.n_labels+1):
            tmp.extend(["%s_%s"%(i, j) for i, j in itertools.product(["top%d"%k], possib)])
        possib.extend(tmp)

        tmp_type = lambda name : "ppl" in name or "loss" in name
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
                loss.backward(retain_graph=retain_graph)
                if self.n_iter % self.params.accumulate_gradients == 0:
                    if self.params.clip_grad_norm > 0:
                        # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                        clip_grad_norm_(self.model.parameters(), self.params.clip_grad_norm)
                        # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                    for optimizer in self.optimizers :
                        optimizer.step()
                    for optimizer in self.optimizers :
                        optimizer.zero_grad()

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
                
        if self.params.is_master and self.pre_trainer is None :
            #self.logger.info("__log__:%s" % json.dumps(scores))
            pass

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

    def classif_step(self, get_loss, batch, total_stats, i) :
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
        
    def one_epoch(self, get_loss):
        self.model.train() # train mode
        total_stats = []
        a, b = True, True
        i = 0
        train_dataloader = iter(self.train_data_iter)
        while a or b :
            # pre_training step
            if b :
                one_step(self.pre_trainer, self.pre_trainer.params)
                b = self.pre_trainer.n_sentences < self.pre_trainer.epoch_size
            
            # classif step
            if a :
                try :
                    batch = next(train_dataloader)
                    self.classif_step(get_loss, batch, total_stats, i)
                except StopIteration :
                    #train_dataloader = iter(self.train_data_iter)
                    a = False
                i+=1
        return total_stats
    
    def train_step(self, get_loss):
        self.model.train() # train mode
        total_stats = []

        for i, batch in enumerate(self.train_data_iter):
            if self.test :
                (x, lengths, langs), y1, y2 = batch
                positions = None
                langs = None # langs.to(self.params.device) if self.params.n_langs > 1 else None
                x, lengths, positions, langs, _ = self.pre_trainer.round_batch(x, lengths, positions, langs)
                x, y, pred_mask = self.pre_trainer.mask_out(x, lengths)
                # cuda
                x, y, pred_mask, lengths, positions, langs = to_cuda(x, y, pred_mask, lengths, positions, langs)
                # forward / loss
                tensor = self.pre_trainer.model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
                _, mlm_loss = self.pre_trainer.model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=False)
                mlm_loss  = float(self.params.lambda_mlm) * mlm_loss 
                lang1 = "en"
                lang2 = None
                self.pre_trainer.stats[('MLM-%s' % lang1) if lang2 is None else ('MLM-%s-%s' % (lang1, lang2))].append(mlm_loss.item())
                
                y = y2 if self.params.version == 3 else y1
                logits, classif_loss = self.model.predict(tensor, y, weights = self.train_data_iter.weights)
                loss, stats = get_loss(self.model, batch, self.params, self.train_data_iter.weights)
                
                _, stats = get_loss(None, batch, self.params, None, logits = logits, loss = classif_loss)
                
                loss = mlm_loss  + classif_loss
            else :
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
    
    def generate(self):
        if getattr("self", "data_generator", None) is None :
            self.data_generator = cross_validatation(X=self.data, y = self.labels, shuffle = self.shuffle, 
                                                    random_state = self.random_state, kwargs = self.kwargs)
        try :
            train_index, val_index = next(self.data_generator)
        except StopIteration :
            self.data_generator = cross_validatation(X=self.data, y = self.labels, shuffle = self.shuffle, 
                                                    random_state = self.random_state, kwargs = self.kwargs) 
            train_index, val_index = next(self.data_generator)
            
        #self.train_data_iter.data = [self.data[i] for i in train_index]
        #self.val_data_iter.data = [self.data[i] for i in val_index]
        self.train_data_iter.reset(data = [self.data[i] for i in train_index])
        self.val_data_iter.reset(data = [self.data[i] for i in val_index])
    
    def data_summary(self) :
        logger.info("")
        logger.info("============ Data summary")
        if self.params.in_memory :
            l1, l2 = len(self.train_data_iter.data), len(self.val_data_iter.data) 
            logger.info("train : %d (x batch_size : %d)"%(l1, l1*self.params.batch_size))
            logger.info("valid : %d (x batch_size : %d)"%(l2, l2*self.params.batch_size))
        else :
            logger.info("train : %d"%len(self.train_data_iter.data))
            logger.info("valid : %d"%len(self.val_data_iter.data))
            
        logger.info("")
            
    def train(self, get_loss, end_of_epoch):
        """ Train Loop """
        
        data = self.train_data_iter.data + self.val_data_iter.data
        if self.params.cross_validation == "" :
            self.cross_validation = False
        else :
            self.cross_validation = True
            self.kwargs = get_cv_kwargs(cv_type = self.params.cross_validation, data = data)

        self.random_state = self.params.random_seed
        self.shuffle = self.params.shuffle
        
        self.labels = None
        if not self.params.in_memory :
            self.labels = []
            for _, y1, y2 in data :
                y = y2 if self.params.version == 3 else y1
                try :
                    y.item()
                    scalar = True
                except ValueError: #only one element tensors can be converted to Python scalars
                    scalar = False
                break
                    
            for _, y1, y2 in data :
                y = y2 if self.params.version == 3 else y1
                y = y if scalar else torch.argmax(y)
                self.labels.append(y.item())   
                
        if self.cross_validation :
            if self.kwargs["cv_type"] == "holdout" :
                self.cross_validation = False
                try :
                    train_data, val_data = train_test_split(
                        data, 
                        test_size = self.kwargs["test_size"], 
                        random_state = self.random_state,
                        shuffle = self.shuffle,
                        stratify = self.labels
                    )
                except ValueError: #The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
                    train_data, val_data = train_test_split(
                        data, 
                        test_size = self.kwargs["test_size"], 
                        random_state = self.random_state,
                        shuffle = self.shuffle,
                        stratify = None
                    )
                self.train_data_iter.reset(data = train_data)
                self.val_data_iter.reset(data = val_data)
            else :
                self.data = data
                
        self.data_summary()

        self.test = not self.params.pretrain_type == 0 and self.pre_trainer is not None
        
        for _ in range(self.params.max_epoch):
            
            self.logger.info("============ Starting epoch %i ... ============" % self.epoch)
            self.n_sentences = 0
            
            if self.cross_validation :
                self.generate() 
                self.data_summary()
            
            if self.pre_trainer is not None :
                self.pre_trainer.n_sentences = 0
                self.pre_trainer.stats['progress'] = 0
        
            for k in self.stats.keys():
                if "avg" in k :
                    self.stats[k] = []
            
            if self.params.pretrain and not self.test :
                train_stats = self.one_epoch(get_loss)
            else :
                train_stats = self.train_step(get_loss)
            
            self.logger.info("============ End of epoch %i ============" % self.epoch)

            val_stats = self.eval_step(get_loss)
            
            if self.params.pretrain :
                pre_train_scores = pretrainer_eoe(self.pre_trainer.params, trainer = self.pre_trainer, evaluator = self.evaluator, end = False)
            else :
                pre_train_scores = {}
                
            scores = end_of_epoch([val_stats, train_stats], self.params)
            
            s = {**scores, **pre_train_scores}
            self.all_scores.append(s)
            
            self.plot_score(scores)

            # end of epoch
            if self.params.pretrain :
                pretrainer_eoe(self.pre_trainer.params, logger = self.logger, trainer = self.pre_trainer, scores = pre_train_scores)

            self.save_best_model(s)
            self.save_periodic()
            self.end_epoch(s)
            
            self.logger.info("============ garbage collector collecting %d ..." % gc.collect())
            
        plot_all_scores(self.all_scores)
        
    def eval(self, get_loss, end_of_epoch):
        """ Eval Loop """
        val_stats = self.eval_step(get_loss)
        scores = end_of_epoch([val_stats], self.params, add_output = True)
        
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