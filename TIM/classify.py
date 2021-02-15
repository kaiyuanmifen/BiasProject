""" Fine-tuning on A Classification Task with pretrained Transformer """

#pretrain_file='../uncased_L-12_H-768_A-12/bert_model.ckpt',
#pretrain_file='../exp/bert/pretrain_100k/model_epoch_3_steps_9732.pt',

from tensorboardX import SummaryWriter

from train import TrainerConfig, Trainer
from models import ModelConfig
from utils import set_seeds

from tokenization1 import FullTokenizer
from tokenization2 import BertTokenizer
from tokenization3 import build_tokenizer

from dataset import Tokenizing, AddSpecialTokensWithTruncation, TokenIndexing
from optim import optim4GPU

def main(task='sentiment_analysis',
         train_cfg='config/classification.json',
         model_cfg='config/bert_base.json',
         data_file='../glue/MRPC/train.tsv',
         model_file=None,
         pretrain_file='../uncased_L-12_H-768_A-12/bert_model.ckpt',
         data_parallel=True,
         vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='../exp/bert/mrpc',
         max_len=128,
         mode='train'):

    cfg = TrainerConfig.from_json(train_cfg)
    model_cfg =  ModelConfig.from_json(model_cfg)

    set_seeds(cfg.seed)

    tokenizer = FullTokenizer(vocab_file=vocab, do_lower_case=True)
    TaskDataset = dataset_class(task) # task dataset class according to the task
    pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                AddSpecialTokensWithTruncation(max_len),
                TokenIndexing(tokenizer.convert_tokens_to_ids,
                              TaskDataset.labels, max_len)]

    dataset = TaskDataset(data_file, pipeline)
    data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

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
    n_labels=len(TaskDataset.labels)
    model = BertClassifier(transformer, n_labels=n_labels, dropout=model_cfg.dropout_rate)


    criterion = nn.CrossEntropyLoss()

    optimizer = optim4GPU(model=model, lr=cfg.lr, warmup=cfg.warmup, total_steps=cfg.total_steps)
    trainer = train.Trainer(cfg, model, data_iter, optimizer, save_dir, get_device())

    if mode == 'train':
        def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits, label_id)
            return loss

        trainer.train(get_loss, model_file, pretrain_file, data_parallel)

    elif mode == 'eval':
        def evaluate(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            _, label_pred = logits.max(1)
            result = (label_pred == label_id).float() #.cpu().numpy()
            accuracy = result.mean()
            return accuracy, result

        results = trainer.eval(evaluate, model_file, data_parallel)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy:', total_accuracy)


if __name__ == '__main__':
    main()
