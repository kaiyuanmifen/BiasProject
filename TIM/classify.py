""" Fine-tuning on A Classification Task with pretrained Transformer """

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import argparse 
import numpy as np

from train import Trainer
from models import Transformer, BertClassifier, TIM_EncoderLayer
from utils import set_seeds, get_device, special_tokens
from tokenization1 import FullTokenizer
#from tokenization2 import BertTokenizer
#from tokenization3 import build_tokenizer
from dataset import Tokenizing, AddSpecialTokensWithTruncation, TokenIndexing, dataset_class
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
        tokenizer = build_tokenizer(tokenizer_path, corpus, vocab_size, st)

    TaskDataset = dataset_class(params.task) # task dataset class according to the task
    pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                AddSpecialTokensWithTruncation(params.max_len),
                TokenIndexing(tokenizer.convert_tokens_to_ids,
                              TaskDataset.labels, params.max_len)]
    params.train_n_samples = None if params.train_n_samples==-1 else params.train_n_samples
    params.val_n_samples = None if params.val_n_samples==-1 else params.val_n_samples
    if not params.eval_only :
        train_dataset = TaskDataset(params.train_data_file, pipeline, params.train_n_samples, params.shuffle)
        train_data_iter = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=params.shuffle)

        i = len(train_data_iter)
        train_num_data = i*params.batch_size 
        setattr(params, "train_num_step", i)
    else :
        train_data_iter = None

    val_dataset = TaskDataset(params.val_data_file, pipeline, params.val_n_samples, params.shuffle)
    val_data_iter = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=params.shuffle)

    i = len(val_data_iter)
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
    n_labels = len(TaskDataset.labels)
    model = BertClassifier(transformer, n_labels=n_labels, dropout=params.dropout_rate)
    logger.info(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim4GPU(model=model, lr=params.lr, warmup=params.warmup, total_steps=params.train_total_steps)
    trainer = Trainer(params, model, train_data_iter, val_data_iter, optimizer, get_device(), logger)

    def get_loss(model, batch): # make sure loss is a scalar tensor
        input_ids, segment_ids, input_mask, label_id = batch
        logits = model(input_ids, segment_ids, input_mask)
        loss = criterion(logits, label_id)

        stats = {}
        stats["loss"] = loss.item()
        stats["label_pred"] = logits.max(1)[1].view(-1).cpu().numpy()
        stats["label_id"] = label_id.view(-1).cpu().numpy()

        return loss, stats

    def end_of_epoch(stats_list):
        scores = {}
        for prefix, total_stats in zip(["val", "train"], stats_list):
            loss = 0
            label_pred = []
            label_ids = []
            for stats in total_stats :
                label_pred.extend(stats["label_pred"])
                label_ids.extend(stats["label_id"])
                loss += stats['loss']

            scores["%s_acc"%prefix] = (np.array(label_pred) == np.array(label_ids)).astype(float).mean()
            scores["%s_loss"%prefix] = loss #/ len(total_stats)

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
