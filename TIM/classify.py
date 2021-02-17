""" Fine-tuning on A Classification Task with pretrained Transformer """

#pretrain_file='../uncased_L-12_H-768_A-12/bert_model.ckpt',
#pretrain_file='../exp/bert/pretrain_100k/model_epoch_3_steps_9732.pt',

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
import json
import os
import argparse 

from train import Trainer
from models import Transformer, BertClassifier, TIM_EncoderLayer
from utils import set_seeds, get_device, special_tokens
from tokenization1 import FullTokenizer
from tokenization2 import BertTokenizer
from tokenization3 import build_tokenizer
from dataset import Tokenizing, AddSpecialTokensWithTruncation, TokenIndexing, dataset_class
from optim import optim4GPU
from params import get_parser
from type import config_dic

def main(params):

    set_seeds(params.seed)

    option = 1
    if option == 1 :
        tokenizer = FullTokenizer(vocab_file=params.vocab_file, do_lower_case=True)
    if option == 2 :
        name="BertWordPieceTokenizer3"
        path="data/BertWordPieceTokenizer"
        train_path="bias_corpus3.txt"
        vocab_size = 10000
        st = special_tokens
        min_frequency=2
        tokenizer = BertTokenizer(params.max_len, path, name, train_path, vocab_size, min_frequency, st)
    if option == 3 :
        tokenizer_path= "data/tfds.SubwordTextEncoder_vocab3.txt"
        with open("bias_corpus3.txt", "r") as f:
            corpus = f.readlines()
        vocab_size = 10000
        st = special_tokens
        tokenizer = build_tokenizer(tokenizer_path, corpus, vocab_size, st)

    TaskDataset = dataset_class(params.task) # task dataset class according to the task
    pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                AddSpecialTokensWithTruncation(params.max_len),
                TokenIndexing(tokenizer.convert_tokens_to_ids,
                              TaskDataset.labels, params.max_len)]

    dataset = TaskDataset(params.data_file, pipeline)
    data_iter = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)

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
    n_labels=len(TaskDataset.labels)
    model = BertClassifier(transformer, n_labels=n_labels, dropout=params.dropout_rate)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim4GPU(model=model, lr=params.lr, warmup=params.warmup, total_steps=params.total_steps)
    trainer = Trainer(params, model, data_iter, optimizer, params.save_dir, get_device())

    if params.mode == 'train':
        def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits, label_id)
            return loss

        trainer.train(get_loss, params.model_file, params.pretrain_file, params.data_parallel)

    elif params.mode == 'eval':
        def evaluate(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            _, label_pred = logits.max(1)
            result = (label_pred == label_id).float() #.cpu().numpy()
            accuracy = result.mean()
            return accuracy, result

        results = trainer.eval(evaluate, params.model_file, params.data_parallel)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy:', total_accuracy)

if __name__ == '__main__':
    
    params = get_parser().parse_args()
        
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if os.path.isfile(params.config_file):
        with open(params.config_file) as json_data:
            data_dict = json.load(json_data)
            for key, value in data_dict.items():
                conf = config_dic[key]   
                if value == "False":
                    value = False
                elif value == "True" :
                    value = True
                
                """
                try :
                    setattr(params, key, conf[0](value))
                except :
                    setattr(params, key, value)
                """
                # Allow to overwrite the parameters of the json configuration file.
                try :
                    value = conf[0](value)
                except :
                    pass
                
                if getattr(params, key, conf[1]) == conf[1] :
                    setattr(params, key, value)

    main(params)