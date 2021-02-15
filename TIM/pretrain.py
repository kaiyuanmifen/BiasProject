""" Pretrain transformer with Masked LM and Sentence Classification """

from tensorboardX import SummaryWriter

from train import TrainerConfig, Trainer
from models import ModelConfig
from utils import set_seeds
from tokenization1 import FullTokenizer
from tokenization2 import BertTokenizer
from tokenization3 import build_tokenizer
from dataset import Preprocess4Pretrain, SentPairDataLoader, SentPairDataLoader
from optim import optim4GPU

def main(train_cfg='config/pretrain.json',
         model_cfg='config/bert_base.json',
         data_file='/content/corpus.txt',
         model_file=None,
         data_parallel=True,
         vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='/content/bert_pretrain',
         log_dir='/content/bert_pretrain/runs',
         max_len=512,
         max_pred=20,
         mask_prob=0.15):

    cfg = TrainerConfig.from_json(train_cfg)
    model_cfg =  ModelConfig.from_json(model_cfg)
    
    set_seeds(cfg.seed)

    tokenizer = FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    pipeline = [Preprocess4Pretrain(max_pred,
                                    mask_prob,
                                    list(tokenizer.vocab.keys()),
                                    tokenizer.convert_tokens_to_ids,
                                    max_len)]
    data_iter = SentPairDataLoader(data_file,
                                   cfg.batch_size,
                                   tokenize,
                                   max_len,
                                   pipeline=pipeline)

    vocab_size = model_cfg.vocab_size
    tim_encoder_layer = None
    tim_layers_pos = None
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

    model = BertModel4Pretrain(sample_transformer)

    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()

    optimizer = optim4GPU(model=model, lr=cfg.lr, warmup=cfg.warmup, total_steps=cfg.total_steps)
    trainer = Trainer(cfg, model, data_iter, optimizer, save_dir, get_device())
    writer = SummaryWriter(log_dir=log_dir) # for tensorboardX


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

    trainer.train(get_loss, model_file, None, data_parallel)


if __name__ == '__main__':
    main()
