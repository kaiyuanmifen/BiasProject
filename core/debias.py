# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from platform import version
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
import random
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from bleu import list_bleu, multi_list_bleu

from src.utils import to_cuda, restore_segmentation_py, restore_segmentation
from src.model.transformer import TransformerModel
from src.optim import get_optimizer
from src.evaluation.evaluator import convert_to_text, eval_moses_bleu
from src.classification.trainer import Trainer

from classify import __clf__main__

eps = 1e-8
KEYS = {k : k for k in ["input", "gen", "deb"]}

def calc_bleu(reference, hypothesis):
    "https://www.nltk.org/_modules/nltk/translate/bleu_score.html"
    weights = (0.25, 0.25, 0.25, 0.25)
    #sf = SmoothingFunction().method1
    sf = None
    return 100. * nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights, smoothing_function=sf)
    #return list_bleu([reference], hypothesis)

def write_text_z_in_file(output_file, text_z_prime) :
    source_text = []
    generated_text = []
    debias_text = []
    with open(output_file, 'w') as f:
        keys = list(text_z_prime.keys())
        k1 = keys.index(KEYS["input"])
        k2 = keys.index(KEYS["gen"])
        k3 = keys.index(KEYS["deb"])
        K_ =  len(keys)
        L = len(text_z_prime[keys[0]])
        for i in range(L) :
            item = [text_z_prime[k][i] for k in keys]
            for j in range(len(item[0])) :
                f.writelines(["%s : %s\n"% (keys[k], item[k][j]) for k in range(K_)])
                source = item[k1][j].split(" ")
                before = item[k2][j].split(" ")
                after = item[k3][j].split(" ")
                source_text.append(item[k1][j])
                generated_text.append(item[k2][j])
                debias_text.append(item[k3][j])
                b1 = round(calc_bleu(source, before), 4)
                b2 = round(calc_bleu(source, after), 4)
                b3 = round(calc_bleu(before, after), 4)
                f.writelines([f"bleu --> {KEYS['input']} vs {KEYS['gen']} = %s, {KEYS['input']} vs {KEYS['deb']} = %s, {KEYS['gen']} vs {KEYS['deb']} = %s\n"%(b1, b2, b3)])
                f.write("\n")
    with open(output_file + ".source.txt", 'w') as f:
        f.writelines(["%s\n"%t for t in source_text])
    with open(output_file + ".gen.txt", 'w') as f:
        f.writelines(["%s\n"%t for t in generated_text])
    with open(output_file + ".deb.txt", 'w') as f:
        f.writelines(["%s\n"%t for t in debias_text])
        
class LossDebias:
    """"""
    def __init__(self,  penalty="lasso"):
        assert penalty in ["lasso", "ridge"]
        if penalty == "lasso" :
            #self.criterion = F.l1_loss
            self.ord = 1
        else :
            #self.criterion = F.mse_loss
            self.ord = 2

    def __call__(self, z, z_prime, is_list = True):
        """
        z, z_prime : (n_layers, bs, seq_len, dim) if is_list, else (bs, seq_len, dim)
        """
        if is_list:
            z = torch.stack(z).transpose(0, 1) # (bs, n_layers, seq_len, dim)
            z_prime = torch.stack(z_prime).transpose(0, 1) # (bs, n_layers, seq_len, dim)
        #    loss = torch.stack([self.criterion(z_i, z_prime_i) for z_i, z_prime_i in zip(z, z_prime)]) # (bs, )
        #else :
        #    loss = self.criterion(z, z_prime) 

        #loss = torch.linalg.matrix_norm(z-z_prime, ord=self.ord, dim=(-2, -1), keepdim=False) # (bs, n_layers)
        loss = torch.linalg.norm(z-z_prime, ord=self.ord, dim=(-1)) # (bs, n_layers, seq_len)
        #loss = torch.linalg.vector_norm(z-z_prime, ord=self.ord, dim=(-2, -1)) # (bs, n_layers)      
        #return loss.sum()
        return loss.sum(dim=-1).sum(dim=-1).mean()

def to_var(x, volatile=False):
    #if torch.cuda.is_available():
    #    x = x.cuda()
    return Variable(x, volatile=volatile)

def arrange_x(x, lengths, langs) :
    """
    Let's suppose that we have at the beginning (pad_index = 2, </s>_index = 1)
    x = LongTensor([[1, x_11, x_12, 1, 2],
                    [1, x_21, x_22, x_23, 1],
                    [1, x_31, 1, 2, 2]]
                    ).transpose(0, 1) # (seq_len, bs)
    So lengths = LongTensor([4, 5, 3])

    If we select some examples base on the mask : examples = tensor([True, False, True]) 
    We end-up with :
    x = x[:,examples] = LongTensor(
                            [[1,   1],
                            [x_11, x_31],
                            [x_12, 1],
                            [1,    2],
                            [2,    2]])
    and : lengths = lengths[examples] = LongTensor([4, 3])
    So : max(lengths) != len(x)
    This method corrects this by making x become :
    x = LongTensor(
                [[1,   1],
                [x_11, x_31],
                [x_12, 1],
                [1,    2]])
    """
    #alen = torch.arange(max(lengths), dtype=torch.long, device=lengths.device)
    alen = torch.arange(max(lengths))
    return x[alen], langs[alen]

def select_with_mask(batch, mask) :
    flag = mask.any()
    if flag :
        (x, lengths, langs), y1, y2, weight_out = batch
        _, bs = x.shape
        x = x[:,mask] # (seq_len, bs-ϵ)
        lengths = lengths[mask]
        langs = langs[:,mask] # (seq_len, bs-ϵ)
        y1 = y1[mask]
        y2 = y2[mask]
        weight_out = weight_out[mask]
        if bs == 1 :
            x = x.squeeze(1)
            langs = langs.squeeze(1)
            lengths = lengths.squeeze(0)
            y1 = y1.squeeze(0)
            y2 = y2.squeeze(0)
            weight_out = weight_out.squeeze(0)
        x, langs = arrange_x(x, lengths, langs)
        batch = (x, lengths, langs), y1, y2, weight_out
    return batch, flag

class LinearDeb(torch.nn.Module):
    def __init__(self, pretrain_params):
        super().__init__()
        if True :
            self.deb = torch.nn.Linear(pretrain_params.emb_dim, pretrain_params.emb_dim)
        else :
            intermediate_dim = 100
            self.deb = torch.nn.Sequential(
                torch.nn.Linear(pretrain_params.emb_dim, intermediate_dim),
                torch.nn.Linear(intermediate_dim, pretrain_params.emb_dim)
            )
        self.n_layers = pretrain_params.n_layers

    def forward(self, mode, **kwargs):
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(self, x, lengths, causal, src_enc=None, src_len=None, positions=None, langs=None, cache=None,
            intermediate_states = False
    ):  
        """
        x : (bs, seq_len, emb_dim)
        """
        h = self.deb(x).transpose(0, 1) # (seq_len, bs, emb_dim)
        if intermediate_states :
            return h, [h]*self.n_layers
        return h

    def predict(self, tensor, pred_mask, y, get_scores, reduction='mean'):
        pass
    
class Debias_Trainer(Trainer) :
    def __init__(self, pretrain_params, *args, **kwds):
        super().__init__(pretrain_params, *args, **kwds)
        # only on negative example
        # self.params.train_only_on_negative_examples = True
        assert self.params.penalty in ["lasso", "ridge"]
        assert self.params.type_penalty in ["last", "group"]
        assert self.params.yoshua 
        self.bin_classif = self.params.version == 7
        self.max_label = 2.0 * self.params.threshold # 1.0 if threshold = 0.5, 5.0 if threshold=2.5 
        if self.bin_classif :
            self.params.threshold = 0.5
            self.max_label = 1.0
        else :
            self.max_label = 5.0
            pass

        self.params.pretrain_type = 1 # for evaluation (text gen)
        self.params.eval_pretrainer = False # for evaluation (classification)
        self.alpha, self.beta = [float(coef) for coef in self.params.deb_alpha_beta.split(",")]
        self.denoising_ae = self.pre_trainer.params.ae_steps != []    
        self.lambda_ = self.params.threshold
        #self.on_init(pretrain_params)
        self.deb_optimizer = get_optimizer(self.deb.parameters(), self.params.deb_optimizer)
        self.deb_criterion = LossDebias(penalty=self.params.penalty)
        self.after_init()

        if self.params.fgim :
            self.train = self.fgim_algorithm
            self.eval = self.fgim_algorithm

    def on_init(self, params, p_params):
        dump_path = os.path.join(params.dump_path, "debias")
        checkpoint_path = os.path.join(dump_path, "checkpoint.pth")
        if os.path.isfile(checkpoint_path):
            self.params.dump_path = dump_path
            self.checkpoint_path = checkpoint_path
            self.from_deb = True
        else :
            self.checkpoint_path = os.path.join(params.dump_path, "checkpoint.pth")
            self.from_deb = False
        deb = TransformerModel(p_params, self.model.dico, is_encoder=True, 
                                    with_output = False, with_emb = False)
        #deb = LinearDeb(p_params)
        self.deb = deb.to(params.device)
        #self.deb_optimizer = get_optimizer(self.deb.parameters(), self.params.deb_optimizer)
        #self.deb_criterion = LossDebias(penalty=self.params.penalty)

    def after_init(self) :
        self.params.dump_path = os.path.join(self.params.dump_path, "debias")
        os.makedirs(self.params.dump_path, exist_ok=True)
        self.checkpoint_path = os.path.join(self.params.dump_path, "checkpoint.pth")
        #self.model.embedder.model = self.pre_trainer.encoder
        #self.pre_trainer.encoder = self.model.embedder.model
        
    def classif_step(self, get_loss, y, batch):
        (x, lengths, langs), _, _, _ = batch
        z, z_list = self.pre_trainer.encoder('fwd', x=x, lengths=lengths, langs=langs, 
                causal=False, intermediate_states = True)
        logits, classif_loss = self.model.predict(
            z, y, weights = self.val_data_iter.weights)
        _, stats = get_loss(None, batch, self.params, None, logits = logits, loss = classif_loss, mode="train", epoch = self.epoch)
        if self.bin_classif :
            y_hat = stats["logits"]
        else :
            y_hat = stats["label_pred"]
        return classif_loss, logits, z, z_list, stats, y_hat

    def debias_step(self, y, lengths, z, z_list, mask_deb, bs):
        if mask_deb is not None:
            lengths_deb = lengths[mask_deb].squeeze(0) if bs == 1 else lengths[mask_deb]
            z_deb = z[mask_deb].squeeze(0) if bs == 1 else z[mask_deb] # (bs-ϵ, seq_len, dim)
        else :
            lengths_deb = lengths 
            z_deb = z + 0.0
        z_prime, z_prime_list = self.deb('fwd', x=z_deb, lengths=lengths_deb, 
            causal=False, intermediate_states = True)
        z_prime = z_prime.transpose(0, 1)
        if self.params.type_penalty == "last" :
            #z_prime = z_prime.transpose(0, 1)
            loss_deb = self.deb_criterion(z_deb, z_prime, is_list = False) 
        elif self.params.type_penalty == "group" :
            z_list, z_prime_list = z_list[1:], z_prime_list[1:] # exclude words embedding
            if mask_deb is not None :
                z_deb_list = [z_[mask_deb] for z_ in z_list] # (n_layers, bs, seq_len, dim)
            else :
                z_deb_list = z_list # (n_layers, bs, seq_len, dim)
            #z_prime_list = [z_.transpose(0, 1) for z_ in z_prime_list] # (n_layers, bs, seq_len, dim)
            #assert len(z_deb_list) == len(z_prime_list)
            loss_deb = self.deb_criterion(z_deb_list, z_prime_list, is_list = True) 
        
        #self.params.positive_label==0
        if mask_deb is not None:
            y_deb = y[mask_deb]
            y_prime = self.max_label - y_deb # "1"-1=0, "1"-0=1 ...
            #y_prime = m - (y > self.params.threshold).float()
            z_prime = z_prime.transpose(0, 1)
            logits_deb, classif_loss = self.model.predict(z_prime, y_prime, weights = None)
            if self.bin_classif :
                #debias_label_loss = - F.logsigmoid(logits_deb)
                debias_label_loss = - torch.log(1 - torch.sigmoid(logits_deb)).sum()
            else :
                debias_label_loss = - F.log_softmax(logits_deb, dim = 1)[:,0].sum()
                #debias_label_loss = - F.log_softmax(logits_deb.T, dim = 0)[0].sum()
        else :
            y_prime = self.max_label - y # "1"-1=0, "1"-0=1 ...
            #y_prime = m - (y > self.params.threshold).float()
            z_prime = z_prime.transpose(0, 1)
            logits_deb, classif_loss = self.model.predict(z_prime, y_prime, weights = None)
            if self.bin_classif :
                prob = torch.sigmoid(logits_deb)
                debias_label_loss = - torch.log(y * (1.0 - prob) + (1.0 - y) * prob).sum()
            else :
                prob = F.softmax(logits_deb)
                debias_label_loss = - torch.log(prob[:,0] + 1 - prob[:,-1]).sum()

        debias_label_loss = 1.0 * debias_label_loss + 1.0 * classif_loss
        loss_deb = self.alpha * loss_deb + self.beta * debias_label_loss
                    
        self.deb_optimizer.zero_grad()
        loss_deb.backward(retain_graph=True)
        self.deb_optimizer.step()

        return loss_deb, z_prime, lengths_deb

    def enc_dec(self, x, lengths, langs, z, non_mask_deb, bs):
        if non_mask_deb is not None :
            x_non_deb = x[:,non_mask_deb] # (seq_len, bs)
            lengths_non_deb = lengths[non_mask_deb]
            langs_non_deb = langs[:,non_mask_deb]
            if bs == 1 :
                x_non_deb = x_non_deb.squeeze(1)
                langs_non_deb = langs_non_deb.squeeze(1)
                lengths_non_deb = lengths_non_deb.squeeze(0)
            x_non_deb, langs_non_deb = arrange_x(x_non_deb, lengths_non_deb, langs_non_deb)
            z = z[non_mask_deb]#.squeeze(0) if bs == 1 else z[non_mask_deb] # (bs, seq_len, dim)
        else :
            x_non_deb = x + 0 # (seq_len, bs)
            lengths_non_deb = lengths
            langs_non_deb = langs
        max_len = lengths_non_deb.max()
        if self.denoising_ae :
            (x2, len2) = (x_non_deb.cpu(), lengths_non_deb.cpu())
            #(x1, len1) = (x_non_deb, lengths_non_deb)
            (x1, len1) = self.pre_trainer.add_noise(x_non_deb.cpu(), lengths_non_deb.cpu())
            # target words to predict
            alen = torch.arange(max_len, dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()
            # cuda
            x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)
            # encode source sentence
            langs1 = langs_non_deb[torch.arange(x1.size(0))]
            enc1 = self.pre_trainer.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            #lambda_coeff = self.pre_trainer.params.lambda_ae
            lambda_coeff = 1
        else :
            x2, len1, len2 = x_non_deb, lengths_non_deb, lengths_non_deb
            enc1 = z
            # target words to predict
            alen = torch.arange(max_len, dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()
            # cuda
            y = y.to(x_non_deb.device)
            lambda_coeff = 1
        dec2 = self.pre_trainer.decoder('fwd', x=x2, lengths=len2, langs=langs_non_deb, causal=True, src_enc=enc1, src_len=len1)
        word_scores, loss_rec = self.pre_trainer.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        loss_rec = lambda_coeff * loss_rec

        return loss_rec, word_scores, y

    def generate(self, x, lengths, langs, z, z_prime = None, log = True, max_print=2):
        input_sent = convert_to_text(x, lengths, self.evaluator.dico, self.pre_trainer.params)
        with torch.no_grad(): 
            if z_prime is None :
                z_prime = self.deb('fwd', x=z, lengths=lengths, causal=False)
                z_prime = z_prime.transpose(0, 1)
            """
            #lang1, lang2 = self.pre_trainer.params.mt_steps[0]
            lang1, lang2 = self.pre_trainer.params.langs
            #lang1_id = self.pre_trainer.params.lang2id[lang1]
            lang2_id = self.pre_trainer.params.lang2id[lang2]
            """
            lang2_id = 1
            max_len = int(1.5 * lengths.max().item() + 10)
            seq_len = lengths.max()
            if seq_len >= max_len :
                scr_langs = langs[torch.arange(max_len)]
            else :
                #tgt_langs = torch.cat((langs, langs[torch.arange(max_len - seq_len)]), dim=0)
                scr_langs = torch.cat((langs, langs[0].repeat(max_len - seq_len, 1)), dim=0)
            tgt_langs = 1 - scr_langs # the target langs is the opposite of the source lang
            
            self.pre_trainer.params.beam_size = 1
            if self.pre_trainer.params.beam_size == 1 :       
                generated_1, lengths_1 = self.pre_trainer.decoder.generate(z, lengths, lang2_id, 
                            max_len=max_len, sample_temperature=None, langs = tgt_langs)
                generated_2, lengths_2 = self.pre_trainer.decoder.generate(z_prime, lengths, lang2_id, 
                            max_len=max_len, sample_temperature=None, langs = tgt_langs)
            else :
                pass
                """
                beam_size = self.pre_trainer.params.beam_size
                tgt_langs = tgt_langs.repeat(1, beam_size) # (max_len, bs * beam_size)
                generated_1, lengths_1 = self.pre_trainer.decoder.generate_beam(
                        z, lengths, lang2_id, beam_size = beam_size,
                        length_penalty = self.pre_trainer.params.length_penalty,
                        early_stopping = self.pre_trainer.params.early_stopping,
                        max_len = max_len, langs = tgt_langs)
                generated_2, lengths_2 = self.pre_trainer.decoder.generate_beam(
                        z_prime, lengths, lang2_id, beam_size = beam_size,
                        length_penalty = self.pre_trainer.params.length_penalty,
                        early_stopping = self.pre_trainer.params.early_stopping,
                        max_len = max_len, langs = tgt_langs)
                #"""
            gen_text = convert_to_text(generated_1, lengths_1, self.evaluator.dico, self.pre_trainer.params)
            deb_sent = convert_to_text(generated_2, lengths_2, self.evaluator.dico, self.pre_trainer.params)
        if log :
            i = random.randint(0, len(z)-1)
            max_print = min(i+max_print, len(x))
            self.logger.info("input : %s"%restore_segmentation_py(input_sent[i:max_print]))
            self.logger.info("gen : %s"%restore_segmentation_py(gen_text[i:max_print]))
            self.logger.info("deb : %s"%restore_segmentation_py(deb_sent[i:max_print]))
        else :
            input_sent = restore_segmentation_py(input_sent)
            gen_text = restore_segmentation_py(gen_text)
            deb_sent = restore_segmentation_py(deb_sent)
            return {KEYS["input"] : input_sent, KEYS["gen"] : gen_text, KEYS["deb"] : deb_sent}
            #return input_sent, gen_text, deb_sent

    def train_step(self, get_loss) :
        # train mode
        self.deb.train() 
        self.model.train()
        self.pre_trainer.encoder.train()
        self.pre_trainer.decoder.train()
        
        total_stats = []
        for i, batch in enumerate(self.train_data_iter):
            stats_ = {}
            n_words, xe_loss, n_valid = 0, 0, 0
            (x, lengths, langs), y1, y2, weight_out = batch
            stats_['n_words'] = lengths.sum().item()
            flag = True
            if self.params.train_only_on_negative_examples :
                #negative_examples = ~(y2.squeeze() < self.params.threshold)
                negative_examples = y2.squeeze() > self.params.threshold
                batch, flag = select_with_mask(batch, mask = negative_examples)
                (x, lengths, langs), y1, y2, weight_out = batch
            if flag :    
                y = y2 if self.params.version == 3 else y1
                x, y, lengths, langs = to_cuda(x, y, lengths, langs)
                #langs = langs if self.params.n_langs > 1 else None
                #langs = None
                batch = (x, lengths, langs), y1, y2, weight_out
                classif_loss, logits, z, z_list, stats, y_hat = self.classif_step(get_loss, y, batch)
                #self.optimize(classif_loss, retain_graph = True)
                stats_ = {**stats, **stats_}
                
                version = 1
                if version == 0: 
                    mask_deb = y_hat.squeeze()>=self.lambda_ if self.params.positive_label==0 else y_hat.squeeze()<self.lambda_
                    non_mask_deb = ~mask_deb
                    flag = mask_deb.any()
                    rec_step = non_mask_deb.any()
                else :
                    mask_deb = None
                    non_mask_deb = None
                    flag = True
                    rec_step = True 
                ###############
                z = z.transpose(0, 1) # (bs-ϵ, seq_len, dim)
                bs = z.size(0)
                loss_deb = 0 # torch.tensor(float("nan"))
                loss_rec = 0 # torch.tensor(float("nan"))
                if flag :  # if f(z) > lambda :
                    loss_deb, _, _ = self.debias_step(y, lengths, z, z_list, mask_deb, bs)
                if rec_step : # else :
                    loss_rec, word_scores, y_ = self.enc_dec(x, lengths, langs, z, non_mask_deb, bs)
                    # update stats
                    n_words += y_.size(0)
                    xe_loss += loss_rec.item() * len(y_)
                    n_valid += (word_scores.max(1)[1] == y_).sum().item()
                    # compute perplexity and prediction accuracy
                    n_words = n_words+eps
                    stats_['rec_ppl'] = np.exp(xe_loss / n_words)
                    stats_['rec_acc'] = 100. * n_valid / n_words

                # optimize
                loss = classif_loss + loss_deb + loss_rec
                #self.pre_trainer.optimize(loss)  
                stats_["loss_"] = loss.item() 

                #if True :
                if self.n_total_iter % self.log_interval == 0 :
                    self.generate(x, lengths, langs, z)

            # number of processed sentences / words
            self.n_sentences += self.params.batch_size
            self.stats['processed_s'] += self.params.batch_size
            self.stats['processed_w'] += stats_['n_words']
            self.stats['progress'] = min(int(((i+1)/self.params.train_num_step)*100), 100)

            total_stats.append(stats_)
            self.put_in_stats(stats_)

            self.iter()
            self.print_stats()

            if self.epoch_size < self.n_sentences :
                break

        return total_stats    

    def one_epoch(self, get_loss) :
        self.train_step(get_loss)

    def eval_step(self, get_loss, test = False, prefix =""):
        # eval mode
        self.deb.eval() 
        self.model.eval()
        self.pre_trainer.encoder.eval()
        self.pre_trainer.decoder.eval()

        total_stats = []
        text_z_prime = {KEYS["input"] : [], KEYS["gen"] : [], KEYS["deb"] : [], 
                        "origin_labels" : [], "pred_label" : []}
        references = []
        hypothesis = []
        hypothesis2 = []
        with torch.no_grad(): 
            for batch in tqdm(self.val_data_iter, desc='val'):
                n_words, xe_loss, n_valid = 0, 0, 0
                (x, lengths, langs), y1, y2, weight_out = batch
                flag = True
                """
                # only on negative example
                #negative_examples = ~(y2.squeeze() < self.params.threshold)
                negative_examples = y2.squeeze() > self.params.threshold
                batch, flag = select_with_mask(batch, mask = negative_examples)
                (x, lengths, langs), y1, y2, weight_out = batch
                #"""
                if flag :    
                    y = y2 if self.params.version == 3 else y1
                    x, y, lengths, langs = to_cuda(x, y, lengths, langs)
                    #langs = langs if self.params.n_langs > 1 else None
                    #langs = None
                    batch = (x, lengths, langs), y1, y2, weight_out
                    _, _, z, _, stats, y_hat = self.classif_step(get_loss, y, batch)
                    z = z.transpose(0, 1) # (bs-ϵ, seq_len, dim)
                    bs = z.size(0)

                    z_prime = self.deb('fwd', x=z, lengths=lengths, causal=False)
                    z_prime = z_prime.transpose(0, 1) # (bs-ϵ, seq_len, dim)

                    non_mask_deb = torch.BoolTensor([True]*bs)
                    loss_rec, word_scores, y_ = self.enc_dec(x, lengths, langs, z, non_mask_deb, bs)
                    # update stats
                    n_words += y_.size(0)
                    xe_loss += loss_rec.item() * len(y_)
                    n_valid += (word_scores.max(1)[1] == y_).sum().item()
                    # compute perplexity and prediction accuracy
                    n_words = n_words+eps
                    stats['rec_ppl'] = np.exp(xe_loss / n_words)
                    stats['rec_acc'] = 100. * n_valid / n_words
    
                    texts = self.generate(x, lengths, langs, z, z_prime = z_prime, log = False)
                    for k, v in texts.items():
                        text_z_prime[k].append(v)
                    references.extend(texts[KEYS["input"]])
                    hypothesis.extend(texts[KEYS["gen"]])
                    hypothesis2.extend(texts[KEYS["deb"]])
                    text_z_prime["origin_labels"].append(y.cpu().numpy())
                    text_z_prime["pred_label"].append(y_hat.cpu().numpy())

                    total_stats.append(stats)

        self.end_eval(text_z_prime, references, hypothesis, hypothesis2)

        if test :
            pre_train_scores = {}
            return total_stats, pre_train_scores

        return total_stats

    def end_eval(self, text_z_prime, references, hypothesis, hypothesis2):
        output_file, i = "output", 1
        while os.path.isfile(os.path.join(self.params.dump_path, output_file+str(i)+'.txt')):
            i += 1
        output_file = os.path.join(self.params.dump_path, output_file+str(i)+'.txt')
        write_text_z_in_file(output_file, text_z_prime)
        restore_segmentation(output_file)

        # compute BLEU
        eval_bleu = True
        if eval_bleu and references:
            if False :
                bleu = multi_list_bleu(references, hypothesis)
                self.bleu = sum(bleu) / len(bleu)
                self.logger.info("average BLEU %s %s : %f" % ("input", "gen", self.bleu))
            else :
                # hypothesis / reference paths
                hyp_name, ref_name, i ="hyp", "ref", 1
                while os.path.isfile(os.path.join(self.params.dump_path, hyp_name+str(i)+'.txt')):
                    i += 1
                ref_path = os.path.join(self.params.dump_path, ref_name+str(i)+'.txt')
                hyp_path = os.path.join(self.params.dump_path, hyp_name+str(i)+'.txt')
                hyp_path2 = os.path.join(self.params.dump_path, hyp_name+'_deb' +str(i)+'.txt')

                # export sentences to reference and hypothesis file
                with open(ref_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(references) + '\n') 
                with open(hyp_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(hypothesis) + '\n')
                with open(hyp_path2, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(hypothesis2) + '\n')
                
                restore_segmentation(ref_path)
                restore_segmentation(hyp_path)
                restore_segmentation(hyp_path2)

                # evaluate BLEU score
                self.bleu = eval_moses_bleu(ref_path, hyp_path)
                self.logger.info("BLEU input-gen : %f (%s, %s)" % (self.bleu, hyp_path, ref_path))
                bleu = eval_moses_bleu(ref_path, hyp_path2)
                self.logger.info("BLEU input-deb : %f (%s, %s)" % (bleu, hyp_path2, ref_path))
                bleu = eval_moses_bleu(hyp_path, hyp_path2)
                self.logger.info("BLEU gen-deb : %f (%s, %s)" % (bleu, hyp_path, hyp_path2))

    def fgim_algorithm(self, get_loss, end_of_epoch):
        """
        Controllable Unsupervised Text Attribute Transfer 
        via Editing Entangled Latent Representation
        """
        threshold = 0.001
        lambda_ = 0.9
        max_iter_per_epsilon = 100
        w = [2.0,3.0,4.0,5.0,6.0,7.0,8.0]
        limit_batches = 10
        
        # eval mode
        #self.deb.eval() 
        #self.model.eval()
        self.pre_trainer.encoder.eval()
        self.pre_trainer.decoder.eval()

        text_z_prime = {KEYS["input"] : [], KEYS["gen"] : [], KEYS["deb"] : [],  
                        "origin_labels" : [], "pred_label" : [], "change" : [], "w_i":[]}
        references = []
        hypothesis = []
        hypothesis2 = []
        n_batches = 0

        def get_y_hat(logits) :
            if self.bin_classif :
                probs = torch.sigmoid(logits)
                y_hat = probs.round().int()
            else :
                probs, y_hat = logits.max(dim=1)
            return y_hat, probs

        for batch in tqdm(self.train_data_iter):
            stats_ = {}
            (x, lengths, langs), y1, y2, weight_out = batch
            stats_['n_words'] = lengths.sum().item()
            flag = True
            if self.params.train_only_on_negative_examples :
                #negative_examples = ~(y2.squeeze() < self.params.threshold)
                negative_examples = y2.squeeze() > self.params.threshold
                batch, flag = select_with_mask(batch, mask = negative_examples)
                (x, lengths, langs), y1, y2, weight_out = batch
            if flag :    
                y = y2 if self.params.version == 3 else y1
                x, y, y2, lengths, langs = to_cuda(x, y, y2, lengths, langs)
                #langs = None
                batch = [(x, lengths, langs), y1, y2, weight_out]
                origin_data = self.pre_trainer.encoder('fwd', x=x, lengths=lengths, langs=langs, causal=False)
                # Define target label
                if self.bin_classif :
                    #y_prime = self.max_label - y
                    y_prime = self.max_label - (y > self.params.threshold).float()
                else :
                    #y_prime = self.max_label - (y > self.params.threshold).float()
                    y_prime = self.max_label - y#.float()
                batch[2 if self.params.version == 3 else 1] = y_prime
                flag = False
                for w_i in w:
                    #print("---------- w_i:", w_i)
                    data = to_var(origin_data.clone())  # (batch_size, seq_length, latent_size)
                    b = True
                    if b :
                        data.requires_grad = True
                        logits, classif_loss = self.model.predict(
                            data, y_prime, weights = self.train_data_iter.weights)
                        #_, stats = get_loss(None, batch, self.params, None, logits = logits, loss = classif_loss, mode="train", epoch = self.epoch)
                        #y_hat = stats["label_pred"]
                        y_hat, _ = get_y_hat(logits)
                        self.model.zero_grad()
                        classif_loss.backward()
                        data = data - w_i * data.grad.data
                    else :
                        logits, classif_loss = self.model.predict(
                            data, y_prime, weights = self.train_data_iter.weights)
                        #_, stats = get_loss(None, batch, self.params, None, logits = logits, loss = classif_loss, mode="train", epoch = self.epoch)
                        #y_hat = stats["label_pred"]
                        y_hat, _ = get_y_hat(logits)

                    it = 0 
                    while True:
                        #if torch.cdist(y_hat, y_prime) < threshold :
                        #if ((y_hat - y_prime)**2).sum().float().sqrt() < threshold :
                        #if (y_hat - y_prime).abs().float().mean() < threshold :
                        if y_hat == y_prime :
                            flag = True
                            break
            
                        data = to_var(data.clone())  # (batch_size, seq_length, latent_size)
                        # Set requires_grad attribute of tensor. Important for Attack
                        data.requires_grad = True
                        logits, classif_loss = self.model.predict(
                            data, y_prime, weights = self.train_data_iter.weights)
                        # Calculate gradients of model in backward pass
                        self.model.zero_grad()
                        classif_loss.backward()
                        data = data - w_i * data.grad.data
                        it += 1
                        # data = perturbed_data
                        w_i = lambda_ * w_i
                        if it > max_iter_per_epsilon:
                            break
                    
                data = data.transpose(0, 1)
                origin_data = origin_data.transpose(0, 1)      
                texts = self.generate(x, lengths, langs, origin_data, z_prime = data, log = False)
                for k, v in texts.items():
                    text_z_prime[k].append(v)
                references.extend(texts[KEYS["input"]])
                hypothesis.extend(texts[KEYS["gen"]])
                hypothesis2.extend(texts[KEYS["deb"]])
                text_z_prime["origin_labels"].append(y2.cpu().numpy())
                text_z_prime["pred_label"].append(y_hat.cpu().numpy())
                text_z_prime["change"].append([flag]*len(y2))
                text_z_prime["w_i"].append([w_i]*len(y2))

            n_batches += 1
            if n_batches > limit_batches:
                break   
        self.end_eval(text_z_prime, references, hypothesis, hypothesis2)

    def modify_pretrainer_score(self, pre_train_scores) :
        data_set = "valid"
        lang1, lang2 = self.pre_trainer.params.mt_steps[0]
        bleu = getattr(self, 'bleu', 0)
        pre_train_scores['%s_%s-%s_mt_bleu' % (data_set, lang1, lang2)] = bleu
        return pre_train_scores

    def save_checkpoint(self, name, include_optimizer=True, include_all_scores=False, do_save=True):
        data, checkpoint_path = super().save_checkpoint(name, include_optimizer, include_all_scores, do_save=False)
        self.logger.warning(f"Saving deb parameters ...")
        data["deb"] = self.deb.state_dict()
        for name in self.pre_trainer.MODEL_NAMES:
            self.logger.warning(f"Saving {name} parameters ...")
            data[name] = getattr(self.pre_trainer, name).state_dict()
        if include_optimizer:
            self.logger.warning(f"Saving deb optimizer ...")
            data['deb_optimizer'] = self.deb_optimizer.state_dict() 
            for name in self.pre_trainer.optimizers.keys():
                self.logger.warning(f"Saving {name} optimizer ...")
                data[f'{name}_optimizer'] = self.pre_trainer.optimizers[name].state_dict()
        
        data['dico_id2word'] = self.pre_trainer.data['dico'].id2word
        data['dico_word2id'] = self.pre_trainer.data['dico'].word2id
        data['dico_counts'] = self.pre_trainer.data['dico'].counts
        data['pretrainer_params'] = {k: v for k, v in self.pre_trainer.params.__dict__.items()}

        torch.save(data, checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path=None, do_print = True):
        data, rcc = super().load_checkpoint(checkpoint_path=checkpoint_path, do_print = False)
        if self.from_deb :
            self.deb.load_state_dict(data["deb"])
            for name in self.pre_trainer.MODEL_NAMES:
                getattr(self.pre_trainer, name).load_state_dict(data[name])

            self.epoch = data['epoch'] + 1
            self.n_total_iter = data['n_total_iter']
            self.best_metrics = data['best_metrics']
            self.best_criterion = data['best_criterion']
        if not self.params.eval_only :
            if rcc :
                self.logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")
            else :
                self.logger.warning(f"Parameters reloaded. Epoch {self.epoch} / iteration {self.n_total_iter} ...")

if __name__ == '__main__':
    __clf__main__(trainer_class = Debias_Trainer)