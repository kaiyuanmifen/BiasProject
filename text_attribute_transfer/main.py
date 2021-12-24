# coding: utf-8
import time
import argparse
import os
import torch
import torch.nn as nn
import copy
import tqdm
import numpy as np
import itertools
import random

from model import make_model, make_deb, Classifier, LabelSmoothing, fgim_attack, fgim, LossSedat
from data import prepare_data, non_pair_data_loader, get_cuda, id2text_sentence, to_var, load_human_answer, calc_bleu
from utils import bool_flag, add_log, add_output, write_text_z_in_file
from optim import get_optimizer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

eps = 1e-8

def preparation(args):
    # set model save path
    if not args.exp_id :
        args.exp_id = str(int(time.time()))
    args.current_save_path = os.path.join(args.dump_path, args.exp_name, args.exp_id)
    if not os.path.exists(args.current_save_path) :
        os.makedirs(args.current_save_path, exist_ok=True)
    local_time = time.localtime()
    args.log_file = os.path.join(args.current_save_path, time.strftime("log_%Y_%m_%d_%H_%M_%S.txt", local_time))
    args.output_file = os.path.join(args.current_save_path, time.strftime("output_%Y_%m_%d_%H_%M_%S.txt", local_time))
    
    add_log(args, "exp id : %s" % args.exp_id)
    add_log(args, "Path: %s is created" % args.current_save_path)
    add_log(args, "create log file at path: %s" % args.log_file)
    
    # prepare data
    args.id_to_word, args.vocab_size = prepare_data(args)
    
def get_n_v_w(tensor_tgt_y, out):
    #tensor_tgt_y : bs, max_seq_len
    #out : bs, max_seq_len, dim
    if False :
        n_w = tensor_tgt_y.size(0) * tensor_tgt_y.size(1)
        n_v = (out.max(-1)[1] == tensor_tgt_y).sum().item()
    if False :
        n_v = (out.max(-1)[1] == tensor_tgt_y).sum().item()
        n_v_tmp = (tensor_tgt_y == 0).sum().item()
        n_v = n_v - n_v_tmp
        n_w = n_w - n_v_tmp
    if True :
        non_padded = tensor_tgt_y != 0
        tensor_tgt_y_ = tensor_tgt_y[non_padded]
        out_ = out[non_padded]
        n_w = tensor_tgt_y_.size(0) 
        n_v = (out_.max(-1)[1] == tensor_tgt_y_).sum().item()
    return n_v, n_w

def train_step(args, data_loader, ae_model, dis_model, ae_optimizer, dis_optimizer, ae_criterion, dis_criterion, epoch):
    ae_model.train()
    dis_model.train()

    loss_ae, n_words_ae, xe_loss_ae, n_valid_ae = 0, 0, 0, 0
    loss_clf, total_clf, n_valid_clf = 0, 0, 0
    epoch_start_time = time.time()
    for it in range(data_loader.num_batch):
        flag_rec = True
        batch_sentences, tensor_labels, \
        tensor_src, tensor_src_mask, tensor_src_attn_mask, tensor_tgt, tensor_tgt_y, \
        tensor_tgt_mask, tensor_ntokens = data_loader.next_batch()

        # Forward pass
        latent, out = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_src_attn_mask, tensor_tgt_mask)

        # Loss calculation
        if not args.sedat :
            loss_rec = ae_criterion(out.contiguous().view(-1, out.size(-1)),
                                    tensor_tgt_y.contiguous().view(-1)) / (tensor_ntokens.data+eps)

        else :
            # only on positive example
            positive_examples = tensor_labels.squeeze()==args.positive_label
            out = out[positive_examples] # or out[positive_examples,:,:]
            tensor_tgt_y = tensor_tgt_y[positive_examples] # or tensor_tgt_y[positive_examples,:] 
            tensor_ntokens = (tensor_tgt_y != 0).data.sum().float() 
            loss_rec = ae_criterion(out.contiguous().view(-1, out.size(-1)),
                                    tensor_tgt_y.contiguous().view(-1)) / (tensor_ntokens.data+eps)
            flag_rec = positive_examples.any()
            out = out.squeeze(0)
            tensor_tgt_y = tensor_tgt_y.squeeze(0)

        if flag_rec :
            n_v, n_w = get_n_v_w(tensor_tgt_y, out)
        else :
            n_w = float("nan")                
            n_v = float("nan")
                
        x_e = loss_rec.item() * n_w 
        loss_ae += loss_rec.item()    
        n_words_ae += n_w
        xe_loss_ae += x_e
        n_valid_ae += n_v
        ae_acc = 100. * n_v / (n_w+eps)
        avg_ae_acc = 100. * n_valid_ae / (n_words_ae+eps)
        avg_ae_loss = loss_ae / (it+1)
        ae_ppl = np.exp(x_e / (n_w+eps))
        avg_ae_ppl = np.exp(xe_loss_ae / (n_words_ae+eps))
        
        ae_optimizer.zero_grad()
        loss_rec.backward(retain_graph=not args.detach_classif)
        ae_optimizer.step()

        # Classifier
        if args.detach_classif :
            dis_lop = dis_model.forward(to_var(latent.clone()))
        else :
            dis_lop = dis_model.forward(latent)

        loss_dis = dis_criterion(dis_lop, tensor_labels)

        dis_optimizer.zero_grad()
        loss_dis.backward()
        dis_optimizer.step()
            
        t_c = tensor_labels.view(-1).size(0) 
        n_v = (dis_lop.round().int() == tensor_labels).sum().item()
        loss_clf += loss_dis.item()
        total_clf += t_c
        n_valid_clf += n_v
        clf_acc = 100. * n_v / (t_c+eps)
        avg_clf_acc = 100. * n_valid_clf / (total_clf+eps)
        avg_clf_loss = loss_clf / (it + 1)
        if it % args.log_interval == 0:
            add_log(args, 'epoch {:3d} | {:5d}/{:5d} batches |'.format(epoch, it, data_loader.num_batch))
            add_log(args, 
                'Train : rec acc {:5.4f} | rec loss {:5.4f} | ppl {:5.4f} | dis acc {:5.4f} | dis loss {:5.4f} |'.format(
                    ae_acc, loss_rec.item(), ae_ppl, clf_acc, loss_dis.item() 
                )
            )
            add_log(args, 
                'Train, avg : rec acc {:5.4f} | rec loss {:5.4f} | ppl {:5.4f} |  dis acc {:5.4f} | dis loss {:5.4f} |'.format(
                    avg_ae_acc, avg_ae_loss, avg_ae_ppl, avg_clf_acc, avg_clf_loss
                )
            )
            if flag_rec :
                i = random.randint(0, len(tensor_tgt_y)-1)
                reference = id2text_sentence(tensor_tgt_y[i], args.id_to_word)
                add_log(args, "input : %s"%reference)
                generator_text = ae_model.greedy_decode(latent,
                                                        max_len=args.max_sequence_length,
                                                        start_id=args.id_bos)
                # batch_sentences
                hypothesis = id2text_sentence(generator_text[i], args.id_to_word)
                add_log(args, "gen : %s"%hypothesis)
                add_log(args, "bleu : %s"%calc_bleu(reference.split(" "), hypothesis.split(" ")))
        
        
    s = {}
    L = data_loader.num_batch + eps
    s["train_ae_loss"] = loss_ae / L
    s["train_ae_acc"] = 100. * n_valid_ae / (n_words_ae+eps)
    s["train_ae_ppl"] = np.exp(xe_loss_ae / (n_words_ae+eps))
    s["train_clf_loss"] = loss_clf / L
    s["train_clf_acc"] = 100. * n_valid_clf / (total_clf+eps)
        
    add_log(args, '| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
        
    add_log(args, 
        '| rec acc {:5.4f} | rec loss {:5.4f} | rec ppl {:5.4f} | dis acc {:5.4f} | dis loss {:5.4f} |'
        .format(s["train_ae_acc"], s["train_ae_loss"], s["train_ae_ppl"], s["train_clf_acc"], s["train_clf_loss"])
    )
    return s
        
        
def eval_step(args, data_loader, ae_model, dis_model, ae_criterion, dis_criterion):
    ae_model.eval()
    dis_model.eval()

    loss_ae, n_words_ae, xe_loss_ae, n_valid_ae = 0, 0, 0, 0
    loss_clf, total_clf, n_valid_clf = 0, 0, 0
    for it in range(data_loader.num_batch):
        flag_rec = True
        batch_sentences, tensor_labels, \
        tensor_src, tensor_src_mask, tensor_src_attn_mask, tensor_tgt, tensor_tgt_y, \
        tensor_tgt_mask, tensor_ntokens = data_loader.next_batch()
        # Forward pass
        latent, out = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_src_attn_mask, tensor_tgt_mask)
        # Loss calculation
        if not args.sedat :
            loss_rec = ae_criterion(out.contiguous().view(-1, out.size(-1)),
                                        tensor_tgt_y.contiguous().view(-1)) / (tensor_ntokens.data+eps)
        else :
            # only on positive example
            positive_examples = tensor_labels.squeeze()==args.positive_label
            out = out[positive_examples] # or out[positive_examples,:,:]
            tensor_tgt_y = tensor_tgt_y[positive_examples] # or tensor_tgt_y[positive_examples,:] 
            tensor_ntokens = (tensor_tgt_y != 0).data.sum().float() 
            loss_rec = ae_criterion(out.contiguous().view(-1, out.size(-1)),
                                    tensor_tgt_y.contiguous().view(-1)) / (tensor_ntokens.data+eps)
            flag_rec = positive_examples.any()
            out = out.squeeze(0)
            tensor_tgt_y = tensor_tgt_y.squeeze(0)
        
        if flag_rec :
            n_v, n_w = get_n_v_w(tensor_tgt_y, out)
        else :
            n_w = float("nan")                
            n_v = float("nan")
                
        x_e = loss_rec.item() * n_w 
        loss_ae += loss_rec.item()    
        n_words_ae += n_w
        xe_loss_ae += x_e
        n_valid_ae += n_v
                        
        # Classifier
        dis_lop = dis_model.forward(to_var(latent.clone()))
        loss_dis = dis_criterion(dis_lop, tensor_labels)
    
        t_c = tensor_labels.view(-1).size(0) 
        n_v = (dis_lop.round().int() == tensor_labels).sum().item()
        loss_clf += loss_dis.item()
        total_clf += t_c
        n_valid_clf += n_v
        
    s = {}
    L = data_loader.num_batch + eps
    s["eval_ae_loss"] = loss_ae / L
    s["eval_ae_acc"] = 100. * n_valid_ae / (n_words_ae+eps)
    s["eval_ae_ppl"] = np.exp(xe_loss_ae / (n_words_ae+eps))
    s["eval_clf_loss"] = loss_clf / L
    s["eval_clf_acc"] = 100. * n_valid_clf / (total_clf+eps)
            
    add_log(args, 
        'Val : rec acc {:5.4f} | rec loss {:5.4f} | rec ppl {:5.4f} | dis acc {:5.4f} | dis loss {:5.4f} |'
        .format(s["eval_ae_acc"], s["eval_ae_loss"], s["eval_ae_ppl"], s["eval_clf_acc"], s["eval_clf_loss"])
    )
    return s

def settings(args, possib):
    
    tmp_type = lambda name : "ppl" in name or "loss" in name
    # validation metrics
    metrics = []
    metrics = [m for m in args.validation_metrics.split(',') if m != '']
    for i in range(len(metrics)) :
        m = metrics[i]
        if tmp_type(m) :
            m = '_%s'%m
        m = (m[1:], False) if m[0] == '_' else (m, True)
        assert m[0] in possib
        metrics[i] = m

    # stopping criterion used for early stopping
    if args.stopping_criterion != '':
        split = args.stopping_criterion.split(',')
        assert len(split) == 2 and split[1].isdigit()
        assert split[0] in possib
        decrease_counts_max = int(split[1])
        decrease_counts = 0
        if tmp_type(split[0]) :
            split[0] = '_%s'%split[0]
        if split[0][0] == '_':
            stopping_criterion = (split[0][1:], False)
        else:
            stopping_criterion = (split[0], True)
        #best_criterion = -1e12 if stopping_criterion[1] else 1e12
        best_criterion = float("-inf") if stopping_criterion[1] else float("inf")
    else:
        stopping_criterion = None
        best_criterion = None
        
    return stopping_criterion, best_criterion, decrease_counts, decrease_counts_max

def pretrain(args, ae_model, dis_model):
    train_data_loader = non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    train_data_loader.create_batches(args, [args.train_data_file], if_shuffle=True, n_samples = args.train_n_samples)
    
    val_data_loader = non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    val_data_loader.create_batches(args, [args.val_data_file], if_shuffle=True, n_samples = args.valid_n_samples)

    ae_model.train()
    dis_model.train()

    ae_optimizer = get_optimizer(parameters=ae_model.parameters(), s=args.ae_optimizer, noamopt=args.ae_noamopt)
    dis_optimizer = get_optimizer(parameters=dis_model.parameters(), s=args.dis_optimizer)
    
    ae_criterion = get_cuda(LabelSmoothing(size=args.vocab_size, padding_idx=args.id_pad, smoothing=0.1), args)
    dis_criterion = nn.BCELoss(size_average=True)
    
    possib = ["%s_%s"%(i, j) for i, j in itertools.product(["train", "eval"], ["ae_loss", "ae_acc", "ae_ppl", "clf_loss", "clf_acc"])]
    stopping_criterion, best_criterion, decrease_counts, decrease_counts_max = settings(args, possib)
    metric, biggest = stopping_criterion
    factor = 1 if biggest else -1
    
    stats = []
    
    add_log(args, "Start train process.")
    for epoch in range(args.max_epochs):
        print('-' * 94)
        add_log(args, "")
        s_train = train_step(args, train_data_loader, ae_model, dis_model, 
                        ae_optimizer, dis_optimizer, ae_criterion, dis_criterion, epoch)
        add_log(args, "")
        s_eval = eval_step(args, val_data_loader, ae_model, dis_model, ae_criterion, dis_criterion)
        scores = {**s_train, **s_eval}
        stats.append(scores)
        add_log(args, "")
        if factor * scores[metric] > factor * best_criterion:
            best_criterion = scores[metric]
            add_log(args, "New best validation score: %f" % best_criterion)
            decrease_counts = 0
            # Save model
            add_log(args, "Saving model to %s ..." % args.current_save_path)
            torch.save(ae_model.state_dict(), os.path.join(args.current_save_path, 'ae_model_params.pkl'))
            torch.save(dis_model.state_dict(), os.path.join(args.current_save_path, 'dis_model_params.pkl'))
        else:
            add_log(args, "Not a better validation score (%i / %i)." % (decrease_counts, decrease_counts_max))
            decrease_counts += 1
        if decrease_counts > decrease_counts_max:
            add_log(args, "Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % decrease_counts_max)
            #exit()
            break
        
    s_test = None
    if os.path.exists(args.test_data_file) :
        add_log(args, "")
        test_data_loader = non_pair_data_loader(
            batch_size=args.batch_size, id_bos=args.id_bos,
            id_eos=args.id_eos, id_unk=args.id_unk,
            max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
        )
        test_data_loader.create_batches(args, [args.test_data_file], if_shuffle=True, n_samples = args.test_n_samples)
        s = eval_step(args, test_data_loader, ae_model, dis_model, ae_criterion, dis_criterion)
        add_log(args, 
            'Test | rec acc {:5.4f} | rec loss {:5.4f} | rec ppl {:5.4f} | dis acc {:5.4f} | dis loss {:5.4f} |'
            .format(s["eval_ae_acc"], s["eval_ae_loss"], s["eval_ae_ppl"], s["eval_clf_acc"], s["eval_clf_loss"])
        )
        s_test = s
    add_log(args, "")
    add_log(args, "Saving training statistics %s ..." % args.current_save_path)
    torch.save(stats, os.path.join(args.current_save_path, 'stats_train_eval.pkl'))
    if s_test is not None :
        torch.save(s_test, os.path.join(args.current_save_path, 'stat_test.pkl'))
    return stats, s_test

def fgim_algorithm(args, ae_model, dis_model):
    batch_size=1
    test_data_loader = non_pair_data_loader(
        batch_size=batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    file_list = [args.test_data_file]
    test_data_loader.create_batches(args, file_list, if_shuffle=False, n_samples = args.test_n_samples)
    if args.references_files :
        gold_ans = load_human_answer(args.references_files, args.text_column)
        assert len(gold_ans) == test_data_loader.num_batch
    else :
        gold_ans = [[None]*batch_size]*test_data_loader.num_batch

    add_log(args, "Start eval process.")
    ae_model.eval()
    dis_model.eval()

    fgim_our = True
    if fgim_our :
        # for FGIM
        z_prime, text_z_prime = fgim(test_data_loader, args, ae_model, dis_model, gold_ans = gold_ans)
        write_text_z_in_file(args, text_z_prime)
        add_log(args, "Saving model modify embedding %s ..." % args.current_save_path)
        torch.save(z_prime, os.path.join(args.current_save_path, 'z_prime_fgim.pkl'))
    else :
        for it in range(test_data_loader.num_batch):
            batch_sentences, tensor_labels, \
            tensor_src, tensor_src_mask, tensor_src_attn_mask, tensor_tgt, tensor_tgt_y, \
            tensor_tgt_mask, tensor_ntokens = test_data_loader.next_batch()

            print("------------%d------------" % it)
            print(id2text_sentence(tensor_tgt_y[0], args.id_to_word))
            print("origin_labels", tensor_labels)

            latent, out = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_src_attn_mask, tensor_tgt_mask)
            generator_text = ae_model.greedy_decode(latent,
                                                    max_len=args.max_sequence_length,
                                                    start_id=args.id_bos)
            print(id2text_sentence(generator_text[0], args.id_to_word))

            # Define target label
            target = get_cuda(torch.tensor([[1.0]], dtype=torch.float), args)
            if tensor_labels[0].item() > 0.5:
                target = get_cuda(torch.tensor([[0.0]], dtype=torch.float), args)
            add_log(args, "target_labels : %s"%target)

            modify_text = fgim_attack(dis_model, latent, target, ae_model, args.max_sequence_length, args.id_bos,
                                            id2text_sentence, args.id_to_word, gold_ans[it])
                    
            add_output(args, modify_text)
            

def sedat_train(args, ae_model, f, deb) :
    """
    Input: 
        Original latent representation z : (n_batch, batch_size, seq_length, latent_size)
    Output: 
        An optimal modified latent representation z'
    """
    # TODO : fin a metric to control the evelotuion of training, mainly for deb model
    lambda_ = args.sedat_threshold
    alpha, beta = [float(coef) for coef in args.sedat_alpha_beta.split(",")]
    # only on negative example
    only_on_negative_example = args.sedat_only_on_negative_example
    penalty = args.penalty
    type_penalty = args.type_penalty
        
    assert penalty in ["lasso", "ridge"]
    assert type_penalty in ["last", "group"]
            
    train_data_loader = non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    file_list = [args.train_data_file]
    if os.path.exists(args.val_data_file) :
        file_list.append(args.val_data_file)
    train_data_loader.create_batches(args, file_list, if_shuffle=True, n_samples = args.train_n_samples)
    
    add_log(args, "Start train process.")
            
    #add_log("Start train process.")
    ae_model.train()
    f.train()
    deb.train()
    
    ae_optimizer = get_optimizer(parameters=ae_model.parameters(), s=args.ae_optimizer, noamopt=args.ae_noamopt)
    dis_optimizer = get_optimizer(parameters=f.parameters(), s=args.dis_optimizer)
    deb_optimizer = get_optimizer(parameters=deb.parameters(), s=args.dis_optimizer)
    
    ae_criterion = get_cuda(LabelSmoothing(size=args.vocab_size, padding_idx=args.id_pad, smoothing=0.1), args)
    dis_criterion = nn.BCELoss(size_average=True)
    deb_criterion = LossSedat(penalty=penalty)

    stats = []
    for epoch in range(args.max_epochs):
        print('-' * 94)
        epoch_start_time = time.time()
        
        loss_ae, n_words_ae, xe_loss_ae, n_valid_ae = 0, 0, 0, 0
        loss_clf, total_clf, n_valid_clf = 0, 0, 0
        for it in range(train_data_loader.num_batch):
            _, tensor_labels, \
            tensor_src, tensor_src_mask, tensor_src_attn_mask, tensor_tgt, tensor_tgt_y, \
            tensor_tgt_mask, _ = train_data_loader.next_batch()
            flag = True
            # only on negative example
            if only_on_negative_example :
                negative_examples = ~(tensor_labels.squeeze()==args.positive_label)
                tensor_labels = tensor_labels[negative_examples].squeeze(0) # .view(1, -1)
                tensor_src = tensor_src[negative_examples].squeeze(0) 
                tensor_src_mask = tensor_src_mask[negative_examples].squeeze(0)  
                tensor_src_attn_mask = tensor_src_attn_mask[negative_examples].squeeze(0)
                tensor_tgt_y = tensor_tgt_y[negative_examples].squeeze(0) 
                tensor_tgt = tensor_tgt[negative_examples].squeeze(0) 
                tensor_tgt_mask = tensor_tgt_mask[negative_examples].squeeze(0) 
                flag = negative_examples.any()
            if flag :
                # forward
                z, out, z_list = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_src_attn_mask, tensor_tgt_mask, return_intermediate=True)
                #y_hat = f.forward(to_var(z.clone()))
                y_hat = f.forward(z)
                
                loss_dis = dis_criterion(y_hat, tensor_labels)
                dis_optimizer.zero_grad()
                loss_dis.backward(retain_graph=True)
                dis_optimizer.step()

                dis_lop = f.forward(z)
                t_c = tensor_labels.view(-1).size(0) 
                n_v = (dis_lop.round().int() == tensor_labels).sum().item()
                loss_clf += loss_dis.item()
                total_clf += t_c
                n_valid_clf += n_v
                clf_acc = 100. * n_v / (t_c+eps)
                avg_clf_acc = 100. * n_valid_clf / (total_clf+eps)
                avg_clf_loss = loss_clf / (it + 1)

                mask_deb = y_hat.squeeze()>=lambda_ if args.positive_label==0 else y_hat.squeeze()<lambda_
                # if f(z) > lambda :
                if mask_deb.any() :
                    y_hat_deb = y_hat[mask_deb]
                    if type_penalty == "last" :
                        z_deb = z[mask_deb].squeeze(0) if args.batch_size == 1 else z[mask_deb] 
                    elif type_penalty == "group" :
                        # TODO : unit test for bach_size = 1
                        z_deb = z_list[-1][mask_deb]
                    z_prime, z_prime_list = deb(z_deb, mask=None, return_intermediate=True)
                    if type_penalty == "last" :
                        z_prime = torch.sum(ae_model.sigmoid(z_prime), dim=1)
                        loss_deb = alpha * deb_criterion(z_deb, z_prime, is_list = False) + beta * y_hat_deb.sum()
                    elif type_penalty == "group" :
                        z_deb_list = [z_[mask_deb] for z_ in z_list]
                        #assert len(z_deb_list) == len(z_prime_list)
                        loss_deb = alpha * deb_criterion(z_deb_list, z_prime_list, is_list = True) + beta * y_hat_deb.sum()
                
                    deb_optimizer.zero_grad()
                    loss_deb.backward(retain_graph=True)
                    deb_optimizer.step()
                else :
                    loss_deb = torch.tensor(float("nan"))
                
                # else :
                if (~mask_deb).any() :
                    out_ = out[~mask_deb] 
                    tensor_tgt_y_ = tensor_tgt_y[~mask_deb] 
                    tensor_ntokens = (tensor_tgt_y_ != 0).data.sum().float() 
                    loss_rec = ae_criterion(out_.contiguous().view(-1, out_.size(-1)),
                                                    tensor_tgt_y_.contiguous().view(-1)) / (tensor_ntokens.data+eps)
                else :
                    loss_rec = torch.tensor(float("nan"))
                    
                ae_optimizer.zero_grad()
                (loss_dis + loss_deb +  loss_rec).backward()
                ae_optimizer.step()
                
                if True :
                    n_v, n_w = get_n_v_w(tensor_tgt_y, out)
                else :
                    n_w = float("nan")                
                    n_v = float("nan")
                        
                x_e = loss_rec.item() * n_w 
                loss_ae += loss_rec.item()    
                n_words_ae += n_w
                xe_loss_ae += x_e
                n_valid_ae += n_v
                ae_acc = 100. * n_v / (n_w+eps)
                avg_ae_acc = 100. * n_valid_ae / (n_words_ae+eps)
                avg_ae_loss = loss_ae / (it+1)
                ae_ppl = np.exp(x_e / (n_w+eps))
                avg_ae_ppl = np.exp(xe_loss_ae / (n_words_ae+eps))
                        
                x_e = loss_rec.item() * n_w 
                loss_ae += loss_rec.item()    
                n_words_ae += n_w
                xe_loss_ae += x_e
                n_valid_ae += n_v
                
                if it % args.log_interval == 0:
                    add_log(args, "")
                    add_log(args, 'epoch {:3d} | {:5d}/{:5d} batches |'.format(epoch, it, train_data_loader.num_batch))
                    add_log(args, 
                        'Train : rec acc {:5.4f} | rec loss {:5.4f} | ppl {:5.4f} | dis acc {:5.4f} | dis loss {:5.4f} |'.format(
                            ae_acc, loss_rec.item(), ae_ppl, clf_acc, loss_dis.item() 
                        )
                    )
                    add_log(args, 
                        'Train : avg : rec acc {:5.4f} | rec loss {:5.4f} | ppl {:5.4f} |  dis acc {:5.4f} | diss loss {:5.4f} |'.format(
                            avg_ae_acc, avg_ae_loss, avg_ae_ppl, avg_clf_acc, avg_clf_loss
                        )
                    )
                        
                    add_log(args, "input : %s"%id2text_sentence(tensor_tgt_y[0], args.id_to_word))
                    generator_text = ae_model.greedy_decode(z,
                                                            max_len=args.max_sequence_length,
                                                            start_id=args.id_bos)
                    # batch_sentences
                    add_log(args, "gen : %s"%id2text_sentence(generator_text[0], args.id_to_word))
                    if mask_deb.any() :
                        generator_text_prime = ae_model.greedy_decode(z_prime,
                                                                max_len=args.max_sequence_length,
                                                                start_id=args.id_bos)

                        add_log(args, "deb : %s"%id2text_sentence(generator_text_prime[0], args.id_to_word))
                
                
        s = {}
        L = train_data_loader.num_batch + eps
        s["train_ae_loss"] = loss_ae / L
        s["train_ae_acc"] = 100. * n_valid_ae / (n_words_ae+eps)
        s["train_ae_ppl"] = np.exp(xe_loss_ae / (n_words_ae+eps))
        s["train_clf_loss"] = loss_clf / L
        s["train_clf_acc"] = 100. * n_valid_clf / (total_clf+eps)
        stats.append(s)        

        add_log(args, "")
        add_log(args, '| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
            
        add_log(args, 
            '| rec acc {:5.4f} | rec loss {:5.4f} | rec ppl {:5.4f} | dis acc {:5.4f} | dis loss {:5.4f} |'
            .format(s["train_ae_acc"], s["train_ae_loss"], s["train_ae_ppl"], s["train_clf_acc"], s["train_clf_loss"])
        )
        
        # Save model
        torch.save(ae_model.state_dict(), os.path.join(args.current_save_path, 'ae_model_params_deb.pkl'))
        torch.save(f.state_dict(), os.path.join(args.current_save_path, 'dis_model_params_deb.pkl'))
        torch.save(deb.state_dict(), os.path.join(args.current_save_path, 'deb_model_params_deb.pkl'))

    add_log(args, "Saving training statistics %s ..." % args.current_save_path)
    torch.save(stats, os.path.join(args.current_save_path, 'stats_train_deb.pkl'))
        
def sedat_eval(args, ae_model, f, deb) :
    """
    Input: 
        Original latent representation z : (n_batch, batch_size, seq_length, latent_size)
    Output: 
        An optimal modified latent representation z'
    """
    max_sequence_length = args.max_sequence_length
    id_bos = args.id_bos
    id_to_word = args.id_to_word
    limit_batches = args.limit_batches

    eval_data_loader = non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    file_list = [args.test_data_file]
    eval_data_loader.create_batches(args, file_list, if_shuffle=False, n_samples = args.test_n_samples)
    if args.references_files :
        gold_ans = load_human_answer(args.references_files, args.text_column)
        assert len(gold_ans) == eval_data_loader.num_batch
    else :
        gold_ans = None

    add_log(args, "Start eval process.")
    ae_model.eval()
    f.eval()
    deb.eval()
    
    text_z_prime = {}
    text_z_prime = {"source" : [], "origin_labels" : [], "before" : [], "after" : [], "change" : [], "pred_label" : []}
    if gold_ans is not None :
        text_z_prime["gold_ans"] = []
    z_prime = []
    n_batches = 0
    for it in tqdm.tqdm(list(range(eval_data_loader.num_batch)), desc="SEDAT"):
        
        _, tensor_labels, \
        tensor_src, tensor_src_mask, tensor_src_attn_mask, tensor_tgt, tensor_tgt_y, \
        tensor_tgt_mask, _ = eval_data_loader.next_batch()
        # only on negative example
        negative_examples = ~(tensor_labels.squeeze()==args.positive_label)
        tensor_labels = tensor_labels[negative_examples].squeeze(0) # .view(1, -1)
        tensor_src = tensor_src[negative_examples].squeeze(0) 
        tensor_src_mask = tensor_src_mask[negative_examples].squeeze(0) 
        tensor_src_attn_mask = tensor_src_attn_mask[negative_examples].squeeze(0)
        tensor_tgt_y = tensor_tgt_y[negative_examples].squeeze(0) 
        tensor_tgt = tensor_tgt[negative_examples].squeeze(0) 
        tensor_tgt_mask = tensor_tgt_mask[negative_examples].squeeze(0) 
        if negative_examples.any():
            if gold_ans is not None :
                text_z_prime["gold_ans"].append(gold_ans[it])
            
            text_z_prime["source"].append([id2text_sentence(t, args.id_to_word) for t in tensor_tgt_y])
            text_z_prime["origin_labels"].append(tensor_labels.cpu().numpy())
            
            origin_data, _ = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_src_attn_mask, tensor_tgt_mask)
            
            generator_id = ae_model.greedy_decode(origin_data, max_len=max_sequence_length, start_id=id_bos)
            generator_text = [id2text_sentence(gid, id_to_word) for gid in generator_id]
            text_z_prime["before"].append(generator_text)
            
            data = deb(origin_data, mask = None)
            data = torch.sum(ae_model.sigmoid(data), dim=1)  # (batch_size, d_model)
            #logit = ae_model.decode(data.unsqueeze(1), tensor_tgt, tensor_tgt_mask)  # (batch_size, max_tgt_seq, d_model)
            #output = ae_model.generator(logit)  # (batch_size, max_seq, vocab_size)   
            y_hat = f.forward(data)  
            y_hat = y_hat.round().int() 
            z_prime.append(data)
            generator_id = ae_model.greedy_decode(data, max_len=max_sequence_length, start_id=id_bos)
            generator_text = [id2text_sentence(gid, id_to_word) for gid in generator_id]
            text_z_prime["after"].append(generator_text)
            text_z_prime["change"].append([True]*len(y_hat))
            text_z_prime["pred_label"].append([y_.item() for y_ in y_hat])
            
            n_batches += 1
            if n_batches > limit_batches:
                break 
    write_text_z_in_file(args, text_z_prime)
    add_log(args, "")
    add_log(args, "Saving model modify embedding %s ..." % args.current_save_path)
    torch.save(z_prime, os.path.join(args.current_save_path, 'z_prime_sedat.pkl'))       
    return z_prime, text_z_prime
    
    
def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Here is your model discription.")

    # main parameters
    ######################################################################################
    #  Environmental parameters
    ######################################################################################
    parser.add_argument('--id_pad', type=int, default=0, help='')
    parser.add_argument('--id_unk', type=int, default=1, help='')
    parser.add_argument('--id_bos', type=int, default=2, help='')
    parser.add_argument('--id_eos', type=int, default=3, help='')

    ######################################################################################
    #  File parameters
    ######################################################################################
    parser.add_argument("--dump_path", type=str, default="save", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment id")
    parser.add_argument('--data_path', type=str, default='', help='')
    parser.add_argument('--train_data_file', type=str, default='', help='')
    parser.add_argument('--val_data_file', type=str, default='', help='')
    parser.add_argument('--test_data_file', type=str, default='', help='')
    parser.add_argument('--references_files', type=str, default='', help='')
    parser.add_argument('--word_to_id_file', type=str, default='', help='')
    parser.add_argument("-dc", "--data_columns", type=str, default="c1,c2,..", help="")

    ######################################################################################
    #  Model parameters
    ######################################################################################
    parser.add_argument('--word_dict_max_num', type=int, default=5, help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--num_layers_AE', type=int, default=2)
    parser.add_argument('--transformer_model_size', type=int, default=256)
    parser.add_argument('--transformer_ff_size', type=int, default=1024)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--word_dropout', type=float, default=1.0)
    parser.add_argument('--embedding_dropout', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--label_size', type=int, default=1)
    
    ######################################################################################
    # Training
    ###################################################################################### 
    parser.add_argument('--max_epochs', type=int, default=10) 
    parser.add_argument('--log_interval', type=int, default=100)  
    parser.add_argument('--eval_only', type=bool_flag, default=False) 
    parser.add_argument('--sedat', type=bool_flag, default=False)
    parser.add_argument('--positive_label', type=int, default=0) 
    
    parser.add_argument('--w', type=str, default="2.0,3.0,4.0,5.0,6.0,7.0,8.0")
    parser.add_argument('--lambda_', type=float, default=0.9)
    parser.add_argument('--threshold', type=float, default=0.001)
    parser.add_argument('--max_iter_per_epsilon', type=int, default=100) 
    parser.add_argument('--limit_batches', type=int, default=-1)
    
    parser.add_argument('--task', type = str, default="pretrain", choices=["pretrain", "debias"], help=  "")
    
    parser.add_argument('--sedat_alpha_beta', type=str, default="1.0,1.0")
    parser.add_argument('--sedat_threshold', type=float, default=0.5)
    parser.add_argument('--sedat_only_on_negative_example', type=bool_flag, default=True)
    parser.add_argument('--penalty', type = str, default="lasso", choices=["lasso", "ridge"], help="") 
    parser.add_argument('--type_penalty', type = str, default="group", choices=["last", "group"], help="") 
    
    parser.add_argument('--detach_classif', type=bool_flag, default=True)
    
    # itertools.product(["train", "eval"], ["ae_loss", "ae_acc", "ae_ppl", "clf_loss", "clf_acc"])]
    parser.add_argument('--validation_metrics', type=str, default="eval_ae_acc")
    parser.add_argument('--stopping_criterion', type=str, default="eval_ae_acc,10")
    
    parser.add_argument('--device', type = str, default="cuda", choices=["cuda", "cpu"], help="")
    
    parser.add_argument("--train_n_samples", type=int, default=-1, help="Just consider train_n_sample train data")
    parser.add_argument("--valid_n_samples", type=int, default=-1, help="Just consider valid_n_sample validation data")
    parser.add_argument("--test_n_samples", type=int, default=-1, help="Just consider test_n_sample test data for")

    ######################################################################################
    # Optimizer
    ######################################################################################     
    parser.add_argument('--ae_noamopt', type=str, default="factor_ae=1,warmup_ae=200")
    parser.add_argument("--ae_optimizer", type=str, default="adam,lr=0,beta1=0.9,beta2=0.98,eps=0.000000001")
    parser.add_argument("--dis_optimizer", type=str, default="adam,lr=0.0001")
    parser.add_argument("--deb_optimizer", type=str, default="adam,lr=0.0001")

    ######################################################################################
    # Checkpoint
    ######################################################################################
    parser.add_argument('--load_from_checkpoint', type=str, default="None") 

    ######################################################################################
    #  End of hyper parameters
    ######################################################################################
    
    return parser

def main(args):
    preparation(args)
    
    ae_model = get_cuda(make_model(d_vocab=args.vocab_size,
                                    N=args.num_layers_AE,
                                    d_model=args.transformer_model_size,
                                    latent_size=args.latent_size,
                                    d_ff=args.transformer_ff_size,
                                    h=args.n_heads, 
                                    dropout=args.attention_dropout
    ), args)
    dis_model = get_cuda(Classifier(latent_size=args.latent_size, output_size=args.label_size), args)
    
    if args.task == "debias" :
        load_db_from_ae_model = False
        if load_db_from_ae_model :
            deb_model = copy.deepcopy(ae_model.encoder)
        else :
            deb_model= get_cuda(make_deb(N=args.num_layers_AE, 
                                d_model=args.transformer_model_size, 
                                d_ff=args.transformer_ff_size, h=args.n_heads, 
                                dropout=args.attention_dropout), args)

    if os.path.exists(args.load_from_checkpoint):
        # Load models' params from checkpoint
        add_log(args, "Load pretrained weigths, pretrain : ae, dis %s ..." % args.load_from_checkpoint)
        try :
            ae_model.load_state_dict(torch.load(os.path.join(args.load_from_checkpoint, 'ae_model_params.pkl')))
            dis_model.load_state_dict(torch.load(os.path.join(args.load_from_checkpoint, 'dis_model_params.pkl')))
        except FileNotFoundError :
            assert args.task == "debias"
        f1 = os.path.join(args.load_from_checkpoint, 'ae_model_params_deb.pkl')
        f2 = os.path.join(args.load_from_checkpoint, 'dis_model_params_deb.pkl')
        if os.path.exists(f1):
            add_log(args, "Load pretrained weigths, debias : ae, dis %s ..." % args.current_save_path)
            ae_model.load_state_dict(torch.load(f1))
            dis_model.load_state_dict(torch.load(f2))
        f3 = os.path.join(args.load_from_checkpoint, 'deb_model_params_deb.pkl')
        if args.task == "debias" and os.path.exists(f1) :
            add_log(args, "Load pretrained weigths, debias : deb %s ..." % args.current_save_path)
            deb_model.load_state_dict(torch.load(f3))
        
    if not args.eval_only:
        if args.task == "pretrain":
            stats, s_test = pretrain(args, ae_model, dis_model)
        if args.task == "debias" :
            sedat_train(args, ae_model, f=dis_model, deb=deb_model) 
            
    if os.path.exists(args.test_data_file) :
        if args.task == "pretrain":
            fgim_algorithm(args, ae_model, dis_model)
        if args.task == "debias" :
            sedat_eval(args, ae_model, f=dis_model, deb=deb_model) 

    print("Done!")

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    args = parser.parse_args()

    # check parameters
    #assert os.path.exists(args.data_path)
    data_columns = args.data_columns
    data_columns = data_columns.split(",")
    assert len(data_columns) == 2
    args.text_column = data_columns[0]
    args.label_column = data_columns[1]
    
    references_files = args.references_files.strip().strip('"')
    if references_files != "" :
        references_files = references_files.split(",")
        #assert all([os.path.isfile(f) or f==""  for f in references_files])
        #args.references_files = references_files
        args.references_files = []
        for f in references_files :
            assert os.path.isfile(f) or f==""
            if f != "":
                args.references_files.append(f)
    else :
        args.references_files = []

    assert args.load_from_checkpoint == "None" or os.path.exists(args.load_from_checkpoint)
    if args.eval_only:
        assert os.path.exists(args.test_data_file)
        assert os.path.exists(args.load_from_checkpoint)

    args.w = [float(x) for x in args.w.split(",")]
    args.limit_batches = float("inf") if args.limit_batches < 0 else args.limit_batches  
    
    if args.ae_noamopt !=  "" :
        args.ae_noamopt = "d_model=%s,%s"%(args.transformer_model_size, args.ae_noamopt)
    
    if not args.device :
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else :
        args.device = torch.device(args.device)
    
    for attr in ["train", "valid", "test"] :
        v = getattr(args, "%s_n_samples"%attr)
        setattr(args, "%s_n_samples"%attr, None if v < 0 else v)
        
    # run the experiments
    main(args)