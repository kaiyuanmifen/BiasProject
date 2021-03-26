# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Training Config & Helper Classes  """

import os
import json
from tqdm import tqdm
import itertools
import gc

import torch
import torch.nn as nn

import checkpoint

class Trainer(object):
    """Training Helper Class"""
    def __init__(self, params, model, train_data_iter, val_data_iter, optimizer, device, logger):
        self.params = params # config for training : see class Config
        self.model = model
        # iterator to load data
        self.train_data_iter = train_data_iter 
        self.val_data_iter = val_data_iter 
        self.optimizer = optimizer
        self.device = device # device name
        self.logger = logger

        self.params.stop = 0
        sc = self.params.stopping_criterion.split(',')
        self.params.stopping_criterion = sc[0]
        self.params.patience = int(sc[1])

        self.check_params()

        if not os.path.exists(params.dump_path) :
            os.mkdir(params.dump_path)

        if not os.path.exists(os.path.join(params.dump_path, params.exp_name)) :
            os.mkdir(os.path.join(params.dump_path, params.exp_name))
  
        if params.reload_transformer :
            logger.info("===== Reload transformer model path from %s ====="%params.reload_transformer)
            self.load(pretrain_file = params.reload_transformer)

        if params.reload_model :
            logger.info("===== Reload model path from %s ====="%params.reload_model)
            self.load(model_file = params.reload_model)

        if params.reload_checkpoint :
            logger.info("===== Reload checkpoint from %s ====="%params.reload_checkpoint)
            self.load_checkpoint(checkpoint_path = params.reload_checkpoint)
            

        self.save_dir = os.path.join(params.dump_path, params.exp_name, params.exp_id)
        self.save_file = os.path.join(self.save_dir, "best_%s.pt"%self.params.validation_metrics)
        self.checkpoint_path = os.path.join(self.save_dir, "checkpoint.pt")

        if not os.path.exists(self.save_dir) :
            os.mkdir(self.save_dir)
        else :
            if os.path.isfile(self.save_file) :
                logger.info("===== Reload model from %s ====="%self.save_file)
                self.load(model_file = self.save_file)

            if os.path.isfile(self.checkpoint_path) :
                logger.info("===== Reload checkpoint from %s ====="%self.checkpoint_path)
                self.load_checkpoint()
       
    @staticmethod
    def prefix_keys(stat_dic, prefix):
        keys = stat_dic.keys()
        for key in keys :
            stat_dic["%s_%s"%(prefix, key)] = stat_dic.pop(key)
        return stat_dic

    def check_params(self):
        possib = ["%s_%s_%s"%(i, j, k) for i, j, k in itertools.product(["train", "val"], ["mlm", "nsp"], ["ppl", "acc", "loss"])]
        possib.extend(["%s_%s"%(i, j) for i, j in itertools.product(["train", "val"], ["ppl", "acc", "loss"])])
        assert self.params.validation_metrics in possib
        assert self.params.stopping_criterion in possib

        if "ppl" in self.params.validation_metrics or "loss" in self.params.validation_metrics :
            self.params.best_score = float("-inf")
        else : # acc
            self.params.best_score = 0

        if "ppl" in self.params.stopping_criterion or "loss" in self.params.stopping_criterion :
            self.params.best_criterion = float("-inf")
        else : # acc
            self.params.best_criterion = 0

    def _type(self, name):
        if "ppl" in name or "loss" in name :
            return 1
        else : # acc
            return 2

    def train_step(self, model, get_loss):
        self.model.train() # train mode
        total_stats = []
        iter_bar = tqdm(self.train_data_iter, desc='Iter (train loss=X.XXX)')
        for i, batch in enumerate(iter_bar):
            batch = [t.to(self.device) for t in batch]

            self.optimizer.zero_grad()
            loss, stats = get_loss(model, batch)
            loss = loss.mean() # mean() for Data Parallelism

            loss.backward()
            self.optimizer.step()

            #total_stats.append(Trainer.prefix_keys(stats, "train"))
            total_stats.append(stats)
            iter_bar.set_description('Iter (train loss=%5.3f)'%loss.item())

        return total_stats

    def eval_step(self, model, get_loss):
        self.model.eval() # eval mode
        total_stats = []
        iter_bar = tqdm(self.val_data_iter, desc='Iter (val loss=X.XXX)')
        for i, batch in enumerate(iter_bar):
            batch = [t.to(self.device) for t in batch]

            loss, stats = get_loss(model, batch)
            loss = loss.mean() # mean() for Data Parallelism

            #total_stats.append(Trainer.prefix_keys(stats, "val"))
            total_stats.append(stats)
            iter_bar.set_description('Iter (val loss=%5.3f)'%loss.item())

        return total_stats

    def is_best(self, score, best, type_ = 1):
        if type_ == 1 : #ppl, loss
            return score < best
        else : # acc
            return score >  best

    def plot_score(self, scores):
        for key, value in scores.items():
            self.logger.info("%s -> %.6f" % (key, value))
            #print("===== {} : {}".format(key, value))
            #print("===== %s : %d ====="%(key, value))

    def train(self, get_loss, end_of_epoch, data_parallel=True):
        """ Train Loop """
        #self.load(model_file, pretrain_file)
        model = self.model.to(self.device)
        if data_parallel and torch.cuda.device_count() > 1 : # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        for e in range(self.params.n_epochs):
            self.logger.info("===== start of epoch %d ====="%(e+1))

            val_stats = self.eval_step(model, get_loss)
            train_stats = self.train_step(model, get_loss)

            self.logger.info("===== end of epoch %d ====="%(e+1))

            scores = end_of_epoch([train_stats, val_stats])
            self.plot_score(scores)

            self.logger.info("===== Save checkpoint to %s ====="%self.checkpoint_path)
            self.save_checkpoint()

            type_ = self._type(self.params.validation_metrics)
            if self.is_best(scores[self.params.validation_metrics], self.params.best_score, type_) :
                self.logger.info("===== Save model to %s ====="%self.save_file)
                self.params.best_score = scores[self.params.validation_metrics]
                self.save()
           
            type_ = self._type(self.params.stopping_criterion)
            if self.is_best(scores[self.params.stopping_criterion], self.params.best_criterion, type_) :
                self.logger.info("===== New best validation socre : %d ====="%scores[self.params.stopping_criterion])
                self.params.stop = 1
                self.params.best_criterion = scores[self.params.stopping_criterion]
            else :
                self.params.stop += 1
                self.logger.info("===== No better validation score %d/%d ====="%(self.params.stop, self.params.patience))

            self.logger.info("============ garbage collector collecting %d ..." % gc.collect())
            self.logger.info("")
            if self.params.stop == self.params.patience :
                #exit()
                break

    def eval(self, get_loss, end_of_epoch, data_parallel=True):
        """ Eval Loop """
        #self.load(model_file, pretrain_file)
        self.model.eval() # eval mode
        model = self.model.to(self.device)
        if data_parallel and torch.cuda.device_count() > 1 : # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        val_stats = self.eval_step(model, get_loss)

        scores = end_of_epoch([val_stats])
        self.plot_score(scores)

    def load(self, model_file = None, pretrain_file = None):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file and os.path.isfile(model_file):
            #self.logger.info('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

        elif pretrain_file and os.path.isfile(pretrain_file): # use pretrained transformer
            #self.logger.info('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'): # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: # remove 'transformer.' (in 'transformer.embedding.norm.bias' for example)
                        value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')} # load only transformer parts
                ) 

    def save(self):
        """ save current model """
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
                    self.save_file)
 
    def save_checkpoint(self):
        """ save current model """
        torch.save({"model" : self.model.state_dict(), "params":self.params}, self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path = None):
        """ save current model """
        checkpoint = torch.load(self.checkpoint_path if checkpoint_path is None else checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        params = checkpoint["params"]
        self.params.stop = params.stop
