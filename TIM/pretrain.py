""" Pretrain transformer with Masked LM and Sentence Classification """

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
import json
import os
import argparse 

from train import TrainerConfig, Trainer
from models import ModelConfig, Transformer, BertModel4Pretrain, TIM_EncoderLayer
from utils import set_seeds, get_device, bool_flag, special_tokens
from tokenization1 import FullTokenizer
from tokenization2 import BertTokenizer
from tokenization3 import build_tokenizer
from dataset import Preprocess4Pretrain, SentPairDataLoader, SentPairDataLoader
from optim import optim4GPU


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--train_cfg", type=str, default="config/pretrain.json", help="")
    parser.add_argument("--model_cfg", type=str, default="config/bert_base.json", help="")
    parser.add_argument("--vocab_file", type=str, default="", help="")
    parser.add_argument("--data_file", type=str, default="", help="")
    parser.add_argument("--model_file", type=str, default="", help="")
    parser.add_argument("--save_dir", type=str, default="/content/bert_pretrain", help="")
    parser.add_argument("--log_dir", type=str, default="/content/bert_pretrain/runs", help="")
    parser.add_argument("--data_parallel", type=bool_flag, default=False, help="")
    
    # data parameters
    parser.add_argument("--max_len", type=int, default=512, help="maximum length of tokens")
    parser.add_argument("--max_pred", type=int, default=20, help="")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="")

    # model parameters
    parser.add_argument("--d_model", type=int, default=512, help="")
    parser.add_argument("--d_k", type=int, default=512, help="")
    parser.add_argument("--d_v", type=int, default=512, help="")
    parser.add_argument("--num_heads", type=int, default=8, help="")    
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="")
    parser.add_argument("--vocab_size", type=int, default=None, help="")
    parser.add_argument("--n_segments", type=int, default=2, help="")

    # tim model parameters
    parser.add_argument("--n_s", type=int, default=2, help="")
    parser.add_argument("--H", type=int, default=8, help="")
    parser.add_argument("--H_c", type=int, default=8, help="")
    parser.add_argument("--tim_layers_pos", type=str, default="", help="tim layers position : 1,2,...")

    parser.add_argument("--config_file", type=str, default="", help="")

    return parser

def main(params):

    cfg = TrainerConfig.from_json(params.train_cfg)
    model_cfg =  ModelConfig.from_json(params.model_cfg)
    
    set_seeds(cfg.seed)

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

    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    pipeline = [Preprocess4Pretrain(params.max_pred,
                                    params.mask_prob,
                                    list(tokenizer.vocab.keys()),
                                    tokenizer.convert_tokens_to_ids,
                                    params.max_len)]
    data_iter = SentPairDataLoader(params.data_file,
                                   cfg.batch_size,
                                   tokenize,
                                   params.max_len,
                                   pipeline=pipeline)
    #assert len(data_iter) != 0
    vocab_size = max(tokenizer.vocab.values()) 

    tim_layers_pos = None
    tim_encoder_layer = None
    if params.tim_layers_pos != "" :
        tim_layers_pos = [int(pos) for pos in params.tim_layers_pos.split(",")]
        tim_encoder_layer = TIM_EncoderLayer(model_cfg.d_model, model_cfg.dim_feedforward, 
                                            params.n_s, model_cfg.d_k, model_cfg.d_v, params.H, params.H_c)
    
    transformer = Transformer(d_model = model_cfg.d_model, 
                                num_heads = model_cfg.num_heads, 
                                d_k = model_cfg.d_k, d_v = model_cfg.d_k,
                                num_encoder_layers = model_cfg.num_encoder_layers,
                                dim_feedforward = model_cfg.dim_feedforward,
                                dropout = model_cfg.dropout_rate,
                                vocab_size = vocab_size,
                                max_len = model_cfg.max_len, n_segments = model_cfg.n_segments,
                                tim_encoder_layer = tim_encoder_layer, tim_layers_pos = tim_layers_pos
                                )  

    model = BertModel4Pretrain(transformer)

    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()

    optimizer = optim4GPU(model=model, lr=cfg.lr, warmup=cfg.warmup, total_steps=cfg.total_steps)
    trainer = Trainer(cfg, model, data_iter, optimizer, params.save_dir, get_device())
    writer = SummaryWriter(log_dir=params.log_dir) # for tensorboardX


    def get_loss(model, batch, global_step): # make sure loss is tensor
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

        logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
        loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids) # for masked LM
        loss_lm = (loss_lm*masked_weights.float()).mean()
        loss_clsf = criterion2(logits_clsf, is_next) # for sentence classification
        writer.add_scalars('data/scalar_group',
                           {'loss_lm': loss_lm.item(),
                            'loss_clsf': loss_clsf.item(),
                            'loss_total': (loss_lm + loss_clsf).item(),
                            'lr': optimizer.get_lr()[0],
                           },
                           global_step)
        return loss_lm + loss_clsf

    trainer.train(get_loss, params.model_file, None, params.data_parallel)


if __name__ == '__main__':
    
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = get_parser()
    params = parser.parse_args()

    if os.path.isfile(params.config_file):
        with open(params.config_file) as json_data:
            data_dict = json.load(json_data)
            for key, value in data_dict.items():
                conf = types.config_dic[key]   
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
