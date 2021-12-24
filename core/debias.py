# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn.functional as F
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
                b1 = round(calc_bleu(source, before), 4)
                b2 = round(calc_bleu(source, after), 4)
                b3 = round(calc_bleu(before, after), 4)
                f.writelines([f"bleu --> {KEYS['input']} vs {KEYS['gen']} = %s, {KEYS['input']} vs {KEYS['deb']} = %s, {KEYS['gen']} vs {KEYS['deb']} = %s\n"%(b1, b2, b3)])
                f.write("\n")

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
        return loss.sum()

class Debias_Trainer(Trainer) :
    def __init__(self, pretrain_params, *args, **kwds):
        super().__init__(pretrain_params, *args, **kwds)
        # only on negative example
        self.params.train_only_on_negative_examples = True
        assert self.params.penalty in ["lasso", "ridge"]
        assert self.params.type_penalty in ["last", "group"]
        assert self.params.yoshua 
        self.bin_classif = self.params.version == 7

        self.params.pretrain_type = 1 # for evaluation (text gen)
        self.params.eval_pretrainer = False # for evaluation (classification)
        self.alpha, self.beta = [float(coef) for coef in self.params.deb_alpha_beta.split(",")]
        self.denoising_ae = self.pre_trainer.params.ae_steps != []    
        self.lambda_ = self.params.threshold
        #self.on_init(pretrain_params)
        self.deb_optimizer = get_optimizer(self.deb.parameters(), self.params.deb_optimizer)
        self.deb_criterion = LossDebias(penalty=self.params.penalty)
        self.after_init()

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
        self.deb = TransformerModel(p_params, self.model.dico, is_encoder=True, 
                                    with_output = False, with_emb = False)
        #self.deb_optimizer = get_optimizer(self.deb.parameters(), self.params.deb_optimizer)
        #self.deb_criterion = LossDebias(penalty=self.params.penalty)

    def after_init(self) :
        self.params.dump_path = os.path.join(self.params.dump_path, "debias")
        os.makedirs(self.params.dump_path, exist_ok=True)
        self.checkpoint_path = os.path.join(self.params.dump_path, "checkpoint.pth")
        #self.model.embedder.model = self.pre_trainer.encoder
        #self.pre_trainer.encoder = self.model.embedder.model
        
    def classif_step(self, get_loss, y, batch):
        (x, lengths, langs), _, _, weight_out = batch
        z, z_list = self.pre_trainer.encoder('fwd', x=x, lengths=lengths, langs=langs, 
                causal=False, intermediate_states = True)
        logits, classif_loss = self.model.predict(
            z, y, weights = self.train_data_iter.weights)
        _, stats = get_loss(None, batch, self.params, None, logits = logits, loss = classif_loss, mode="train", epoch = self.epoch)
        y_hat = stats["label_pred"]
        return classif_loss, logits, z, z_list, stats, y_hat

    def debias_step(self, logits, lengths, z, z_list, mask_deb, bs):
        lengths_deb = lengths[mask_deb].squeeze(0) if bs == 1 else lengths[mask_deb]
        z_deb = z[mask_deb].squeeze(0) if bs == 1 else z[mask_deb] # (bs-ϵ, seq_len, dim)
        z_prime, z_prime_list = self.deb('fwd', x=z_deb, lengths=lengths_deb, 
            causal=False, intermediate_states = True)
        z_prime = z_prime.transpose(0, 1)
        if self.params.type_penalty == "last" :
            #z_prime = z_prime.transpose(0, 1)
            loss_deb = self.deb_criterion(z_deb, z_prime, is_list = False) 
        elif self.params.type_penalty == "group" :
            z_list, z_prime_list = z_list[1:], z_prime_list[1:] # exclude words embedding
            z_deb_list = [z_[mask_deb] for z_ in z_list] # (n_layers, bs, seq_len, dim)
            #z_prime_list = [z_.transpose(0, 1) for z_ in z_prime_list] # (n_layers, bs, seq_len, dim)
            #assert len(z_deb_list) == len(z_prime_list)
            loss_deb = self.deb_criterion(z_deb_list, z_prime_list, is_list = True) 
                    
        #y_hat_deb = y_hat[mask_deb]
        #loss_deb = self.alpha * loss_deb + self.beta * y_hat_deb.sum()
        logits_deb = logits[mask_deb] # (bs, n_class) if not bin_classif else (bs,)
        if self.bin_classif :
            debias_label_loss = - F.logsigmoid(logits_deb)
        else :
            debias_label_loss = - F.log_softmax(logits_deb, dim = 1)[:,0].sum()
            #debias_label_loss = - F.log_softmax(logits_deb.T, dim = 0)[0].sum()
        
        loss_deb = self.alpha * loss_deb + self.beta * debias_label_loss
                    
        self.deb_optimizer.zero_grad()
        loss_deb.backward(retain_graph=True)
        self.deb_optimizer.step()

        return loss_deb, z_prime, lengths_deb

    def enc_dec(self, x, lengths, langs, z, non_mask_deb, bs, max_len):
        x_non_deb = x[:,non_mask_deb] # (seq_len, bs)
        lengths_non_deb = lengths[non_mask_deb]
        if self.denoising_ae :
            (x2, len2) = (x_non_deb, lengths_non_deb)
            #(x1, len1) = (x_non_deb, lengths_non_deb)
            (x1, len1) = self.pre_trainer.add_noise(x_non_deb, lengths_non_deb)
            # target words to predict
            alen = torch.arange(max_len, dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()
            # cuda
            x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)
            # encode source sentence
            enc1 = self.pre_trainer.encoder('fwd', x=x1, lengths=len1, langs=langs, causal=False)
            enc1 = enc1.transpose(0, 1)
            #lambda_coeff = self.pre_trainer.params.lambda_ae
            lambda_coeff = 1
        else :
            x2, len1, len2 = x_non_deb, lengths_non_deb, lengths_non_deb
            enc1 = z[non_mask_deb]#.squeeze(0) if bs == 1 else z[non_mask_deb] # (bs, seq_len, dim)
            # target words to predict
            alen = torch.arange(max_len, dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()
            # cuda
            y = y.to(x_non_deb.device)
            lambda_coeff = 1
        dec2 = self.pre_trainer.decoder('fwd', x=x2, lengths=len2, langs=langs, causal=True, src_enc=enc1, src_len=len1)
        word_scores, loss_rec = self.pre_trainer.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        loss_rec = lambda_coeff * loss_rec

        return loss_rec, word_scores, y

    def generate(self, x, lengths, z, z_prime = None, log = True, max_print=2):
        input_sent = convert_to_text(x, lengths, self.evaluator.dico, self.pre_trainer.params)
        with torch.no_grad(): 
            lang1, lang2 = self.pre_trainer.params.mt_steps[0]
            #lang1_id = self.pre_trainer.params.lang2id[lang1]
            lang2_id = self.pre_trainer.params.lang2id[lang2]
   
            max_len = int(1.5 * lengths.max().item() + 10)
            self.pre_trainer.params.beam_size = 1                         
            generated, lengths_1 = self.pre_trainer.decoder.generate(z, lengths, lang2_id, max_len=max_len)
            gen_text = convert_to_text(generated, lengths_1, self.evaluator.dico, self.pre_trainer.params)
            if z_prime is None :
                z_prime = self.deb('fwd', x=z, lengths=lengths, causal=False)
                z_prime = z_prime.transpose(0, 1)
            generated, lengths_2 = self.pre_trainer.decoder.generate(z_prime, lengths, lang2_id, max_len=max_len)
            deb_sent = convert_to_text(generated, lengths_2, self.evaluator.dico, self.pre_trainer.params)
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
        total_stats = []
        for i, batch in enumerate(self.train_data_iter):
            stats_ = {}
            n_words, xe_loss, n_valid = 0, 0, 0
            (x, lengths, langs), y1, y2, weight_out = batch
            max_len = lengths.max()
            stats_['n_words'] = lengths.sum().item()
            flag = True
            if self.params.train_only_on_negative_examples :
                #negative_examples = ~(y2.squeeze() < self.params.threshold)
                negative_examples = y2.squeeze() > self.params.threshold
                x = x[:,negative_examples] # (seq_len, bs-ϵ)
                lengths = lengths[negative_examples]
                y1 = y1[negative_examples]
                y2 = y2[negative_examples]
                weight_out = weight_out[negative_examples]
                flag = negative_examples.any()
            if flag :    
                y = y2 if self.params.version == 3 else y1
                x, y, lengths, langs = to_cuda(x, y, lengths, langs)
                #langs = langs if self.params.n_langs > 1 else None
                langs = None
                batch = (x, lengths, langs), y1, y2, weight_out
                classif_loss, logits, z, z_list, stats, y_hat = self.classif_step(get_loss, y, batch)
                self.optimize(classif_loss, retain_graph = True)
                stats_ = {**stats, **stats_}
                mask_deb = y_hat.squeeze()>=self.lambda_ if self.params.positive_label==0 else y_hat.squeeze()<self.lambda_
                non_mask_deb = ~mask_deb
                ###############
                z = z.transpose(0, 1) # (bs-ϵ, seq_len, dim)
                bs = z.size(0)
                flag = mask_deb.any()
                loss_deb = 0 # torch.tensor(float("nan"))
                loss_rec = 0 # torch.tensor(float("nan"))
                if flag :  # if f(z) > lambda :
                    loss_deb, _, _ = self.debias_step(logits, lengths, z, z_list, mask_deb, bs)
                if non_mask_deb.any() : # else :
                    try :
                        loss_rec, word_scores, y = self.enc_dec(x, lengths, langs, z, 
                                                        non_mask_deb, bs, max_len)
                        # update stats
                        n_words += y.size(0)
                        xe_loss += loss_rec.item() * len(y)
                        n_valid += (word_scores.max(1)[1] == y).sum().item()
                        # compute perplexity and prediction accuracy
                        n_words = n_words+eps
                        stats_['rec_ppl'] = np.exp(xe_loss / n_words)
                        stats_['rec_acc'] = 100. * n_valid / n_words
                          
                    except RuntimeError :
                        # TODO
                        """
                        transformer.py, line 221, in forward
                            mask = (mask == 0).view(mask_reshape).expand_as(scores)   # (bs, n_heads, qlen, klen)
                        RuntimeError: shape '[1, 1, 1, 10]' is invalid for input of size 5
                        """
                        pass                  

                # optimize
                loss = classif_loss + loss_deb + loss_rec
                self.pre_trainer.optimize(loss)  
                stats_["loss_"] = loss.item() 

                #if True :
                if self.n_total_iter % self.log_interval == 0 :
                    try :
                        # TODO
                        self.generate(x, lengths, z)
                    except AssertionError : # assert lengths.max() == slen and lengths.shape[0] == bs
                        pass

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
        self.deb.eval() # eval mode
        total_stats = []
        text_z_prime = {KEYS["input"] : [], KEYS["gen"] : [], KEYS["deb"] : [], 
                        "origin_labels" : [], "pred_label" : []}
        refences = []
        hypothesis = []

        with torch.no_grad(): 
            for batch in tqdm(self.val_data_iter, desc='val'):
                n_words, xe_loss, n_valid = 0, 0, 0
                (x, lengths, langs), y1, y2, weight_out = batch
                max_len = lengths.max()
                # only on negative example
                flag = True
                #negative_examples = ~(y2.squeeze() < self.params.threshold)
                #"""
                negative_examples = y2.squeeze() > self.params.threshold
                x = x[:,negative_examples] # (seq_len, bs-ϵ)
                lengths = lengths[negative_examples]
                y1 = y1[negative_examples]
                y2 = y2[negative_examples]
                weight_out = weight_out[negative_examples]
                flag = negative_examples.any()
                #"""
                if flag :    
                    y = y2 if self.params.version == 3 else y1
                    x, y, lengths, langs = to_cuda(x, y, lengths, langs)
                    #langs = langs if self.params.n_langs > 1 else None
                    langs = None
                    batch = (x, lengths, langs), y1, y2, weight_out
                    _, _, z, _, stats, y_hat = self.classif_step(get_loss, y, batch)
                    z = z.transpose(0, 1) # (bs-ϵ, seq_len, dim)
                    bs = z.size(0)

                    z_prime = self.deb('fwd', x=z, lengths=lengths, causal=False)
                    z_prime = z_prime.transpose(0, 1) # (bs-ϵ, seq_len, dim)

                    non_mask_deb = torch.BoolTensor([True]*bs)
                    loss_rec, word_scores, y = self.enc_dec(x, lengths, langs, z, 
                                                            non_mask_deb, bs, max_len)
                    # update stats
                    n_words += y.size(0)
                    xe_loss += loss_rec.item() * len(y)
                    n_valid += (word_scores.max(1)[1] == y).sum().item()
                    # compute perplexity and prediction accuracy
                    n_words = n_words+eps
                    stats['rec_ppl'] = np.exp(xe_loss / n_words)
                    stats['rec_acc'] = 100. * n_valid / n_words
                    try :            
                        texts=self.generate(x, lengths, z, z_prime = z_prime, log = False)
                        for k, v in texts.items():
                            text_z_prime[k].append(v)
                        refences.extend(texts[KEYS["input"]])
                        hypothesis.extend(texts[KEYS["gen"]])
                        text_z_prime["origin_labels"].append(y2.cpu().numpy())
                        text_z_prime["pred_label"].append(y_hat.cpu().numpy())
                    except AssertionError :
                        pass
                
                    total_stats.append(stats)

        output_file, i = "output", 1
        while os.path.isfile(os.path.join(self.params.dump_path, output_file+str(i)+'.txt')):
            i += 1
        output_file = os.path.join(self.params.dump_path, output_file+str(i)+'.txt')
        write_text_z_in_file(output_file, text_z_prime)
        restore_segmentation(output_file)

        # compute BLEU
        eval_bleu = True
        if eval_bleu:
            if False :
                bleu = multi_list_bleu(refences, hypothesis)
                self.bleu = sum(bleu) / len(bleu)
                self.logger.info("average BLEU %s %s : %f" % ("input", "gen", self.bleu))
            else :
                # hypothesis / reference paths
                hyp_name, ref_name, i ="hyp", "ref", 1
                while os.path.isfile(os.path.join(self.params.dump_path, hyp_name+str(i)+'.txt')):
                    i += 1
                hyp_path = os.path.join(self.params.dump_path, hyp_name+str(i)+'.txt')
                ref_path = os.path.join(self.params.dump_path, ref_name+str(i)+'.txt')

                # export sentences to reference and hypothesis file
                with open(ref_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(refences) + '\n') 
                with open(hyp_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(hypothesis) + '\n')

                #restore_segmentation(hyp_path)
                #restore_segmentation(ref_path)

                # evaluate BLEU score
                self.bleu = eval_moses_bleu(ref_path, hyp_path)
                self.logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, self.bleu))
        if test :
            pre_train_scores = {}
            return total_stats, pre_train_scores

        return total_stats

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