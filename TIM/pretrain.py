""" Pretrain transformer with Masked LM and Sentence Classification """

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
import os

from train import Trainer
from models import Transformer, BertModel4Pretrain, TIM_EncoderLayer
from utils import set_seeds, get_device, special_tokens
from tokenization1 import FullTokenizer
from tokenization2 import BertTokenizer
from tokenization3 import build_tokenizer
from dataset import Preprocess4Pretrain, SentPairDataLoader, SentPairDataLoader
from optim import optim4GPU
from params import get_parser, from_config_file
 
def main(params):
    
    set_seeds(params.seed)

    option = 1
    if option == 1 :
        tokenizer = FullTokenizer(vocab_file=params.vocab_file, do_lower_case=True)
    if option == 2 :
        name="bias/data/BertWordPieceTokenizer3"
        path="bias/data/BertWordPieceTokenizer"
        train_path="bias/data/bias_corpus3.txt"
        vocab_size = 10000
        st = special_tokens
        min_frequency=2
        tokenizer = BertTokenizer(params.max_len, path, name, train_path, vocab_size, min_frequency, st)
    if option == 3 :
        tokenizer_path= "bias/data/tfds.SubwordTextEncoder_vocab3.txt"
        with open("bias/data/bias_corpus3.txt", "r") as f:
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
                                   params.batch_size,
                                   tokenize,
                                   params.max_len,
                                   pipeline=pipeline)

    i = 0
    for _ in data_iter :
        i += 1
    num_data = i*params.batch_size 
    print("======= num_data : %d ======="%num_data)
    assert num_data != 0
    params.total_steps = params.n_epochs*(num_data/params.batch_size)
    data_iter = SentPairDataLoader(params.data_file,
                                   params.batch_size,
                                   tokenize,
                                   params.max_len,
                                   pipeline=pipeline)
    
    
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

    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()

    optimizer = optim4GPU(model=model, lr=params.lr, warmup=params.warmup, total_steps=params.total_steps)
    trainer = Trainer(params, model, data_iter, optimizer, params.save_dir, get_device())
    writer = SummaryWriter(log_dir=params.log_dir) # for tensorboardX


    def get_loss(model, batch, global_step): # make sure loss is tensor
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

        logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
        loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids) # for masked LM
        loss_lm = (loss_lm*masked_weights.float()).mean()
        loss_clsf = criterion2(logits_clsf, is_next) # for sentence classification

        """
        _, label_pred = logits_lm.max(1)
        result = (label_pred == masked_ids).float() #.cpu().numpy()
        accuracy1 = result.mean()
        print(accuracy1)
        """
        """
        _, label_pred = logits_clsf.max(1)
        result = (label_pred == is_next).float() #.cpu().numpy()
        accuracy2 = result.mean()
        print(accuracy2)
        """

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
    
    params = get_parser().parse_args()
    set_seeds(params.seed)
    params = from_config_file(params)
    main(params)
