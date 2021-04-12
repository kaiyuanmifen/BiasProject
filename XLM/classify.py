import torch
import torch.nn.functional as F
import os

from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import initialize_exp, set_seeds
from src.bias_classification import load_dataset, BertClassifier, Trainer
from src.utils import bool_flag

from params import get_parser, from_config_file
from metrics import get_stats, get_collect, get_score

def get_loss(model, batch, params): 
    (x, lengths, langs), y1, y2 = batch
        
    langs = langs.to(params.device) if params.n_langs > 1 else None
    logits, loss = model(x.to(params.device), lengths.to(params.device), y=y1.to(params.device), positions=None, langs=langs)
        
    stats = {}
    n_words = lengths.sum().item()
    stats['n_words'] = n_words
    stats["avg_loss"] = loss.item()
    stats["loss"] = loss.item()
    stats["y1"] = y1.detach().cpu()#.numpy()
        
    logits = F.softmax(logits, dim = -1)
    
    s = get_stats(logits, y2, params)
    assert (s["label_pred"] == s["top%d_label_pred"%1]).all()
    stats = {**stats, **s}
    stats["logits"] = logits.detach().cpu()
    stats["y2"] = y2.cpu()
    
    if params.version == 2 :
        inv_y2, inv_logits = logits.max(1)[1].view(-1).cpu(), y1.to(params.device)
        s = get_stats(inv_logits, inv_y2, params)
        #assert (s["label_pred"] == s["top%d_label_pred"%1]).all()
        for k in s.keys() :
            stats["inv_%s"%k] = s[k]
        stats["inv_logits"] = inv_logits.detach().cpu()
        stats["inv_y2"] = inv_y2.cpu()
        
        #stats["q_c"] = logits.detach().cpu()#.numpy()
        #stats["p_c"] = y1.detach().cpu()#.numpy()
    
    return loss, stats

def end_of_epoch(stats_list, val_first = True):
    scores = {}
    for prefix, total_stats in zip(["val", "train"] if val_first else ["train","val"], stats_list):
        collect = get_collect(total_stats, params)
        
        l = len(collect['loss'])
        l = l if l != 0 else 1
        scores["%s_loss"%prefix] = sum(collect['loss']) / l
        
        s = get_score(collect, prefix, params, inv="") 
        scores = {**scores, **s}
        
        if params.version == 2 :   
            s = get_score(collect, prefix, params, inv="inv_") 
            scores = {**scores, **s}
 
    return scores

def main(params):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)
    
    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()
            
    model = BertClassifier(n_labels = 6, params = params, logger = logger)
    #model = model.to(params.device)
    logger.info(model)
            
    train_dataset, val_dataset = load_dataset(params, logger, model.dico)
    #logger.info(model.dico.word2id)
    #exit()
    
    # optimizers
    optimizers = model.get_optimizers(params) 
    
    # Trainer
    trainer = Trainer(params, model, optimizers, train_dataset, val_dataset, logger)
    
    logger.info("")
    if not params.eval_only :
        trainer.train(get_loss, end_of_epoch)
    else :
        trainer.eval(get_loss, end_of_epoch)
        
if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    
    parser.add_argument('--version', default=1, const=1, nargs='?',
                        choices=[1, 2], 
                        help=  '1 : averaging the labels with the confidence scores as weights (might be noisy) \
                                2 : computed the coefficient of variation (CV) among annotators for \
                                    each sample in the dataset \
                                    see bias_classification_loss.py for more informations about v2')
    
    #if parser.parse_known_args()[0].version == 2:
    parser.add_argument("--log_softmax", type=bool_flag, default=True, 
                        help="use log_softmax in the loss function instead of log")

    parser.add_argument("--train_data_file", type=str, default="", help="file (.csv) containing the data")
    parser.add_argument("--val_data_file", type=str, default="", help="file (.csv) containing the data")
    
    parser.add_argument("--shuffle", type=bool_flag, default=True, help="shuffle Dataset")
    #parser.add_argument("--group_by_size", type=bool_flag, default=True, help="Sort sentences by size during the training")
    
    parser.add_argument("--codes", type=str, required=True, help="path of bpe code")
    parser.add_argument("--vocab", type=str, default="", help="path of bpe vocab")
    
    parser.add_argument("--min_len", type=int, default=1, 
                        help="minimun sentence length before bpe in training set")
    
    parser.add_argument('--debug_num', default=0, const=0, nargs='?',
                        choices=[0, 1, 2], 
                        help=  '0 : Transformer + Linear \
                                1 : Transformer + Linear + Tanh + Dropout + Linear \
                                2 : Transformer + GRU + Dropout + Linear')
    #if parser.parse_known_args()[0].debug_num in [0, 2] :
    parser.add_argument("--hidden_dim", type=int, default=-1, 
                        help="hidden dimension of classifier")
    
    # GRU
    #if parser.parse_known_args()[0].debug_num == 2 :
    parser.add_argument("--gru_n_layers", type=int, default=1, 
                        help="number of layers, GRU")
    parser.add_argument("--bidirectional", type=bool_flag, default=False, help="bidirectional GRU or not")
    
    parser.add_argument('--topK', default=3, const=3, nargs='?',
                        choices=[1, 2, 3, 4, 5, 6], 
                        help="topK")

    parser.add_argument("--model_path", type=str, default="", 
                        help="Reload transformer model from pretrained model / dico / ...")
    
    parser.add_argument("--finetune_layers", type=str, default='', 
                        help="Layers to finetune. default ==> freeze the transformer encoder part of the model \
                            0:_1 or 0:-1 ===> 0 = embeddings, _1 = last encoder layer \
                            0,1,6 or 0:1,6 ===> embeddings, first encoder layer, 6th encoder layer \
                            0:4,6:8,11 ===> embeddings, 1-2-3-4th encoder layers,  6-7-8th encoder layers, 11 encoder layer \
                            Do not include any symbols other than numbers ([0, n_layers]), comma (,) and two points (:)\
                            This supports negative indexes ( _1 or -1 refers to the last layer for example)")
    parser.add_argument("--weighted_training", type=bool_flag, default=False,
                        help="Use a weighted loss during training")
    #parser.add_argument("--dropout", type=float, default=0,
    #                    help="Fine-tuning dropout")
    parser.add_argument("--optimizer_e", type=str, default="adam,lr=0.0001",
                        help="Embedder (pretrained model) optimizer")
    parser.add_argument("--optimizer_p", type=str, default="adam,lr=0.0001",
                        help="Projection (classifier) optimizer")
    
    params = parser.parse_args()
    params = from_config_file(params)
    
    set_seeds(params.random_seed)

    if params.device not in ["cpu", "cuda"] :
        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else :
        params.device = torch.device(params.device)

    # check parameters
    assert os.path.isfile(params.model_path) #or os.path.isfile(params.reload_checkpoint)
    assert os.path.isfile(params.codes) 
    
    # run experiment
    main(params)