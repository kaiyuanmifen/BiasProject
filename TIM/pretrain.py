""" Pretrain transformer with Masked LM and Sentence Classification """

import torch
import torch.nn as nn

import os
import numpy as np

from train import Trainer
from models import Transformer, BertModel4Pretrain, TIM_EncoderLayer
from utils import set_seeds, get_device, special_tokens
from tokenization1 import FullTokenizer
#from tokenization2 import BertTokenizer
#from tokenization3 import build_tokenizer
from dataset import Preprocess4Pretrain, SentPairDataLoader, SentPairDataLoader
from optim import optim4GPU
from params import get_parser, from_config_file

from slurm import init_signal_handler, init_distributed_mode
from logger import initialize_exp
 
def main(params):
    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)
    
    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()
    
    option = 1
    if option == 1 :
        tokenizer = FullTokenizer(vocab_file=params.vocab_file, do_lower_case=True)
    if option == 2 :
        from tokenization2 import BertTokenizer
        name="bias/data/BertWordPieceTokenizer3"
        path="bias/data/BertWordPieceTokenizer"
        train_path="bias/data/bias_corpus3.txt"
        vocab_size = params.max_vocab if params.max_vocab != -1 else 10000
        st = special_tokens
        min_frequency=2
        tokenizer = BertTokenizer(params.max_len, path, name, train_path, vocab_size, min_frequency, st)
    if option == 3 :
        from tokenization3 import build_tokenizer
        tokenizer_path= "bias/data/tfds.SubwordTextEncoder_vocab3.txt"
        with open("bias/data/bias_corpus3.txt", "r") as f:
            corpus = f.readlines()
        vocab_size = params.max_vocab if params.max_vocab != -1 else 10000
        st = special_tokens
        tokenizer = build_tokenizer(tokenizer_path, corpus, vocab_size, st)

    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    option = 2
    if option == 1 :
        params.train_n_samples = float("inf") if params.train_n_samples==-1 else params.train_n_samples
        params.val_n_samples = float("inf") if params.val_n_samples==-1 else params.val_n_samples
        pipeline = [Preprocess4Pretrain(params.max_pred,
                                        params.mask_prob,
                                        list(tokenizer.vocab.keys()),
                                        tokenizer.convert_tokens_to_ids,
                                        params.max_len)]

        if not params.eval_only :
            train_data_iter = SentPairDataLoader(params.train_data_file,
                                           params.batch_size,
                                           tokenize,
                                           params.max_len,
                                           pipeline=pipeline,
                                           n_samples = params.train_n_samples)
            i = 0
            for _ in train_data_iter :
                i += 1
            train_num_data = i*params.batch_size 
            setattr(params, "train_num_step", i)

        val_data_iter = SentPairDataLoader(params.val_data_file,
                                             params.batch_size,
                                             tokenize,
                                             params.max_len,
                                             pipeline=pipeline,
                                             n_samples = params.val_n_samples)


        i = 0
        for _ in val_data_iter :
            i += 1
        val_num_data = i*params.batch_size 
        setattr(params, "val_num_step", i)

        if not params.eval_only :
            logger.info("======= num_data : train -> %d, val -> %d ======="%(train_num_data, val_num_data))
            assert train_num_data != 0 and val_num_data != 0
            setattr(params, "train_num_data", train_num_data)
            params.train_total_steps = params.n_epochs*(train_num_data/params.batch_size)
        else :
            logger.info("======= num_data : val -> %d ======="%val_num_data)
            assert val_num_data != 0
            params.train_total_steps = 0
        
        setattr(params, "val_num_data", val_num_data)
        params.val_total_steps = params.n_epochs*(val_num_data/params.batch_size)

        if not params.eval_only :
            train_data_iter = SentPairDataLoader(params.train_data_file,
                                        params.batch_size,
                                        tokenize,
                                        params.max_len,
                                        pipeline=pipeline,
                                        n_samples = params.train_n_samples)
        else :
            train_data_iter = None

        val_data_iter = SentPairDataLoader(params.val_data_file,
                                             params.batch_size,
                                             tokenize,
                                             params.max_len,
                                             pipeline=pipeline,
                                             n_samples = params.val_n_samples)
    if option == 2 :
        from torch.utils.data import DataLoader
        from dataset2 import BERTDataset
        from vocab import WordVocab
        encoding="utf-8"
        num_workers = 1

        max_size = params.max_vocab if params.max_vocab != -1 else None
        with open(params.vocab_file, "r", encoding=encoding) as f:
            vocab = WordVocab(f, max_size=max_size)
        
        params.train_n_samples = None if params.train_n_samples==-1 else params.train_n_samples
        params.val_n_samples = None if params.val_n_samples==-1 else params.val_n_samples
        
        if not params.eval_only :
            train_dataset = BERTDataset(corpus_path = params.train_data_file, vocab = vocab, 
                                 seq_len = params.max_len, tokenize = tokenize, 
                                 params = params, encoding=encoding, 
                                 corpus_lines=None, on_memory=True, n_samples = params.train_n_samples)
            
            train_data_iter = DataLoader(train_dataset, batch_size=params.batch_size, 
                                        num_workers=num_workers, shuffle=False)

            train_num_data = len(train_dataset)
            i = train_num_data // params.batch_size # len(train_data_iter)
            setattr(params, "train_num_step", i)

        else :
            train_data_iter = None

        val_dataset = BERTDataset(corpus_path = params.val_data_file, vocab = vocab, 
                                seq_len = params.max_len, tokenize = tokenize, 
                                params = params, encoding=encoding, 
                                corpus_lines=None, on_memory=True, n_samples = params.val_n_samples)
            
        val_data_iter = DataLoader(val_dataset, batch_size=params.batch_size, 
                                  num_workers=num_workers, shuffle=False)

        val_num_data = len(train_dataset)
        i = val_num_data // params.batch_size # len(train_data_iter)
        setattr(params, "val_num_step", i)

        if not params.eval_only :
            logger.info("======= num_data : train -> %d, val -> %d ======="%(train_num_data, val_num_data))
            assert train_num_data != 0 and val_num_data != 0
            setattr(params, "train_num_data", train_num_data)
            params.train_total_steps = params.n_epochs*(train_num_data/params.batch_size)
        else :
            logger.info("======= num_data : val -> %d ======="%val_num_data)
            assert val_num_data != 0
            params.train_total_steps = 0
        
        setattr(params, "val_num_data", val_num_data)
        params.val_total_steps = params.n_epochs*(val_num_data/params.batch_size)

    vocab_size = max(tokenizer.vocab.values()) 

    tim_layers_pos = None
    tim_encoder_layer = None
    if params.tim_layers_pos != "" :
        tim_layers_pos = [int(pos) for pos in params.tim_layers_pos.split(",")]
        tim_encoder_layer = TIM_EncoderLayer(params.d_model, params.dim_feedforward, 
                                            params.n_s, params.d_k, params.d_v, params.H, params.H_c)
    
    transformer = Transformer(d_model = params.d_model, 
                                num_heads = params.num_heads, 
                                d_k = params.d_k, d_v = params.d_k,
                                num_encoder_layers = params.num_encoder_layers,
                                dim_feedforward = params.dim_feedforward,
                                dropout = params.dropout_rate,
                                vocab_size = vocab_size,
                                max_len = params.max_len, n_segments = params.n_segments,
                                tim_encoder_layer = tim_encoder_layer, tim_layers_pos = tim_layers_pos
                                )  

    model = BertModel4Pretrain(transformer)
    logger.info(model)

    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()

    optimizer = optim4GPU(model=model, lr=params.lr, warmup=params.warmup, total_steps=params.train_total_steps)
    trainer = Trainer(params, model, train_data_iter, val_data_iter, optimizer, get_device(), logger)

    def get_loss(model, batch): # make sure loss is tensor
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

        logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
        loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids) # for masked LM
        loss_lm = (loss_lm*masked_weights.float()).mean()
        loss_clsf = criterion2(logits_clsf, is_next) # for sentence classification

        stats = {}
        n_words = masked_ids.size(0)
        stats['n_words'] = n_words
        stats['mlm_loss'] = loss_lm.item() #* n_words
        stats["mlm_label_pred"] = logits_lm.max(-1)[1].view(-1).cpu().numpy()
        stats["mlm_masked_ids"] = masked_ids.view(-1).cpu().numpy()     
           
        stats["nsp_loss"] = loss_clsf.item()
        stats["nsp_label_pred"] = logits_clsf.max(1)[1].view(-1).cpu().numpy()
        stats["nsp_is_next"] = is_next.view(-1).cpu().numpy()
        
        return loss_lm + loss_clsf, stats

    def end_of_epoch(stats_list):
    
        scores = {}
        for prefix, total_stats in zip(["val", "train"], stats_list):
            n_words = 0
            mlm_loss = 0
            nsp_loss = 0
            mlm_label_pred = []
            mlm_masked_ids = []
            nsp_label_pred = []
            nsp_is_next = []
            for stats in total_stats :
                mlm_label_pred.extend(stats["mlm_label_pred"])
                mlm_masked_ids.extend(stats["mlm_masked_ids"])
                nsp_label_pred.extend(stats["nsp_label_pred"])
                nsp_is_next.extend(stats["nsp_is_next"])
                n_words += stats['n_words']
                mlm_loss += stats['mlm_loss']
                nsp_loss += stats['nsp_loss']

            scores["%s_mlm_acc"%prefix] = (np.array(mlm_label_pred) == np.array(mlm_masked_ids)).astype(float).mean()
            scores["%s_mlm_ppl"%prefix] = np.exp(mlm_loss / n_words) if n_words > 0 else 1e9
            #scores["%s_mlm_acc"%prefix] = 100. * n_valid / n_words if n_words > 0 else 0.
            scores["%s_mlm_loss"%prefix] = mlm_loss #/ len(total_stats)

            scores["%s_nsp_acc"%prefix] = (np.array(nsp_label_pred) == np.array(nsp_is_next)).astype(float).mean()
            #scores["%s_nsp_ppl"%prefix] = 2 ** (nsp_loss / len(total_stats)
            scores["%s_nsp_loss"%prefix] = nsp_loss #/ len(total_stats)

            scores["%s_acc"%prefix] = (scores["%s_mlm_acc"%prefix]+scores["%s_nsp_acc"%prefix])/2

        return scores

    logger.info("")
    if not params.eval_only :
        trainer.train(get_loss, end_of_epoch, params.data_parallel)
    else :
        trainer.eval(get_loss, end_of_epoch, params.data_parallel)


if __name__ == '__main__':
    
    params = get_parser().parse_args()
    set_seeds(params.seed)
    params = from_config_file(params)
    if params.device not in ["cpu", "cuda"] :
        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else :
        params.device = torch.device(params.device)
    main(params)
