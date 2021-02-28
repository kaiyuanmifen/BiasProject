import argparse 
import json
import os

# Test
default_d_model = 512  # 64*8
default_h = 8
default_num_encoder_layers = 6

# BERT-Base: 12-layer, 768-hidden-nodes, 12-attention-heads, 110M parameters
#default_d_model = 768 # 64*12
#default_h = 12
#default_num_encoder_layers = 12

# BERT-Large: 24-layer, 1024-hidden-nodes, 16-attention-heads, 340M parameters
#default_d_model = 1024 # 64*16
#default_h = 16
#default_num_encoder_layers = 24

default_dim_feedforward = 2048

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # data parameters
    parser.add_argument("--vocab_file", type=str, default="", help="file (.txt, .json) containing the vocabulary.")
    parser.add_argument("--train_data_file", type=str, default="", help="file (.txt, .csv) containing the data")
    parser.add_argument("--val_data_file", type=str, default="", help="file (.txt, .csv) containing the data")
    parser.add_argument("--max_len", type=int, default=100, help="maximum length of tokens")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--n_segments", type=int, default=2, help="Number of Sentence Segments")

    # model parameters
    parser.add_argument("--d_model", type=int, default=default_d_model, help="model dimension : the number of expected features in the encoder")
    parser.add_argument("--d_k", type=int, default=default_d_model//default_h, help="key dimension")
    parser.add_argument("--d_v", type=int, default=default_d_model//default_h, help="value dimension")
    parser.add_argument("--num_heads", type=int, default=default_h, help="# Numher of Heads in Multi-Headed Attention Layers")    
    parser.add_argument("--num_encoder_layers", type=int, default=default_num_encoder_layers, 
                        help=" Numher of Hidden (Encoder) Layers")
    parser.add_argument("--dim_feedforward", type=int, default=default_dim_feedforward, 
                        help="Dimension of Intermediate Layers in Positionwise Feedforward Net")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="residual dropout rate")
    # Transformers with Independent Mechanisms (TIM) model parameters
    parser.add_argument("--n_s", type=int, default=2, help="number of mechanisms")
    parser.add_argument("--H", type=int, default=default_h, help="number of heads for self-attention")
    parser.add_argument("--H_c", type=int, default=default_h, help="number of heads for inter-mechanism attention")
    parser.add_argument("--tim_layers_pos", type=str, default="", help="tim layers position : 1,2,6 fpr example")
    
    # Training parameters
    parser.add_argument("--pretrain", type=bool_flag, default=True, help="pretrain of fine-tune")
    parser.add_argument("--data_parallel", type=bool_flag, default=False, 
                        help="use Data Parallelism with Multi-GPU")
    parser.add_argument("--seed", type=int, default=3431, help="random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=32, help="batch sizes")
    parser.add_argument("--n_epochs", type=int, default=2, help="Maximum epoch size")
    #parser.add_argument("--accumulate_gradients", type=int, default=1,
    #                    help="Accumulate model gradients over N iterations (N times larger batch sizes)")
    ## pretrain : MLM and NSP
    parser.add_argument("--max_pred", type=int, default=20, help="max tokens of prediction")
    parser.add_argument("--mask_prob", type=float, default=0.15, 
                        help="Fraction of words for which we need to make a prediction")
    parser.add_argument("--word_mask_keep_rand", type=str, default="0.8,0.1,0.1",
                        help="Fraction of words to mask out / keep / randomize, among the words to predict")
    #parser.add_argument("--clip_grad_norm", type=float, default=5,
    #                    help="Clip gradients norm (0 to disable)")
    
    ## fine-tune
    ### https://stackoverflow.com/questions/40324356/python-argparse-choices-with-a-default-choice/40324463
    parser.add_argument('--task', default='bias_classification', const='bias_classification', nargs='?',
                                  choices=['bias_classification','sentiment_analysis', 'mrpc', 'mnli'], help='')

    # optimizer
    parser.add_argument("--warmup", type=float, default=0.1, help="warmup")
    if parser.parse_known_args()[0].pretrain:
        parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    else :
        parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")

 
    # configuration file
    parser.add_argument("--config_file", type=str, default="", help=".json file containing all the parameters")

    parser.add_argument("--stopping_criterion", type=str, default="", 
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="", help="Validation metrics")
    parser.add_argument("--train_n_samples", type=int, default=-1, help="Just consider train_n_sample train data")
    parser.add_argument("--val_n_samples", type=int, default=-1, help="ust consider valid_n_sample validation data")
    parser.add_argument("--eval_only", type=bool_flag, default=False, help="Only run evaluations")

    parser.add_argument("--shuffle", type=bool_flag, default=False, help="shuffle Dataset")


    # log parameters
    parser.add_argument("--dump_path", type=str, default="", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")

    parser.add_argument("--reload_transformer", type=str, default="", 
                        help="Reload transformer model from pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="", 
                        help="Reload a checkpoint")
    parser.add_argument("--reload_model", type=str, default="", 
                        help="Reload a pretrained model")

    parser.add_argument("--device", type=str, default="cuda", 
                        help="change to cpu if cuda is not available")

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")
    parser.add_argument("--log_file_prefix", type=str, default="", 
                        help="Log file prefix.")

    return parser

config_dic = {
    # data parameters
    "vocab_file" :[str, ""],
    "train_data_file":[str, ""],
    "val_data_file":[str, ""],
    "max_len":[int, 100],
    "max_vocab":[int, None],
    "n_segments":[int, 2],

    # model parameters
    "d_model":[int, default_d_model],
    "d_k":[int, default_d_model//default_h],
    "d_v":[int, default_d_model//default_h],
    "num_heads":[int, default_h],
    "num_encoder_layers":[int, default_num_encoder_layers],
    "dim_feedforward":[int, default_dim_feedforward],
    "dropout_rate":[float, 0.1],
    # Transformers with Independent Mechanisms (TIM) model parameters
    "n_s":[int, 2],
    "H":[int, default_h],
    "H_c":[int, default_h],
    "tim_layers_pos":[str, ""],

    # Training parameters
    "pretrain":[bool, True],
    "data_parallel":[bool, False],
    "seed": [int, 3431],
    "batch_size":  [int, 32],
    "n_epochs":  [int, 2],
    ## pretrain : MLM and NSP
    "max_pred":[int, 20],
    "mask_prob":[float, 0.15],
    "word_mask_keep_rand":[str, "0.8,0.1,0.1"],
    ## fine-tune : classification...
    #"mode" : [str, "train"],
    "pretrain_file":[str, ""],
    "task" : [str, ""],

    # optimizer
    "lr": [float, 1e-4],
    "warmup": [float, 0.1],

    # configuration file
    "config_file":[str, ""],

    "stopping_criterion":[str, ""], 
    "validation_metrics":[str, ""], 
    "train_n_samples":[int, -1],
    "val_n_samples":[int, -1],
    "eval_only":[bool, False],

    "shuffle":[bool, False],

    "exp_name":[str, ""],
    "exp_id":[str, ""], 
    "dump_path":[str, ""],
    "reload_transformer":[str, ""],
    "reload_checkpoint":[str, ""],
    "reload_model":[str, ""], 

    "device":"",
    "local_rank":[int, -1],
    "master_port":[int, -1],
    "log_file_prefix":[str, ""]
}

def from_config_file(params):
    if os.path.isfile(params.config_file):
        with open(params.config_file) as json_data:
            data_dict = json.load(json_data)
            for key, value in data_dict.items():
                conf = config_dic.get(key, "__key_error__")
                if conf != "__key_error__":   
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

    return params