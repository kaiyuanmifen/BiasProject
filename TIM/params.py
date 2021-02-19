import argparse 
import json
import os

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

default_d_model = 512
default_h = 8
def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # data parameters
    parser.add_argument("--vocab_file", type=str, default="", help="file (.txt, .json) containing the vocabulary.")
    parser.add_argument("--data_file", type=str, default="", help="file (.txt, .csv) containing the data")
    parser.add_argument("--max_len", type=int, default=512, help="maximum length of tokens")
    parser.add_argument("--vocab_size", type=int, default=None, 
                        help="vocabulary size : will be calculated automatically if not specified.")
    parser.add_argument("--n_segments", type=int, default=2, help="Number of Sentence Segments")

    # model parameters

    parser.add_argument("--d_model", type=int, default=default_d_model, help="model dimension : the number of expected features in the encoder")
    parser.add_argument("--d_k", type=int, default=default_d_model//default_h, help="key dimension")
    parser.add_argument("--d_v", type=int, default=default_d_model//default_h, help="value dimension")
    parser.add_argument("--num_heads", type=int, default=default_h, help="# Numher of Heads in Multi-Headed Attention Layers")    
    parser.add_argument("--num_encoder_layers", type=int, default=6, help=" Numher of Hidden (Encoder) Layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048, 
                        help="Dimension of Intermediate Layers in Positionwise Feedforward Net")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="residual dropout rate")
    # Transformers with Independent Mechanisms (TIM) model parameters
    parser.add_argument("--n_s", type=int, default=2, help="number of mechanisms")
    parser.add_argument("--H", type=int, default=default_h, help="number of heads for self-attention")
    parser.add_argument("--H_c", type=int, default=default_h, help="number of heads for inter-mechanism attention")
    parser.add_argument("--tim_layers_pos", type=str, default="", help="tim layers position : 1,2,6 fpr example")

    # log parameters
    parser.add_argument("--model_file", type=str, default="", help="file (.pt, .ckpt) containing the model to be used as a starting point.")
    parser.add_argument("--save_dir", type=str, default="", help="folder in which the model will be saved")
    parser.add_argument("--log_dir", type=str, default="", help="folder in which the log files will be saved")
    
    # Training parameters
    parser.add_argument("--pretrain", type=bool_flag, default=True, help="pretrain of fine-tune")
    parser.add_argument("--data_parallel", type=bool_flag, default=False, 
                        help="use Data Parallelism with Multi-GPU")
    parser.add_argument("--seed", type=int, default=3431, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--n_epochs", type=int, default=2, help="")
    ## pretrain : MLM and NSP
    parser.add_argument("--max_pred", type=int, default=20, help="max tokens of prediction")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="masking probability")
    ## fine-tune : classification...
    parser.add_argument("--mode", type=str, default="train", help="")
    parser.add_argument("--pretrain_file", type=str, default="", help="")
    ### https://stackoverflow.com/questions/40324356/python-argparse-choices-with-a-default-choice/40324463
    parser.add_argument('--task', default='bias_classification', const='sentiment_analysis', nargs='?',
                                  choices=['bias_classification','sentiment_analysis', 'mrpc', 'mnli'], help='')

    # optimizer
    # todo : total_steps = n_epochs*(num_data/batch_size)
    known_args = parser.parse_known_args()[0]
    if known_args.pretrain:
        default_lr = 1e-4
        default_warmup = 0.1
        default_save_steps = 10000
        default_total_steps = 1000000
    else :
        default_lr = 2e-5
        default_warmup = 0.1
        default_save_steps = 100
        default_total_steps = 345

    parser.add_argument("--lr", type=float, default=default_lr, help="")
    parser.add_argument("--warmup", type=float, default=default_warmup, help="")
    parser.add_argument("--save_steps", type=int, default=default_save_steps, help="")
    parser.add_argument("--total_steps", type=int, default=default_total_steps, help="")
 
    # configuration file
    parser.add_argument("--config_file", type=str, default="", help=".json file containing all the parameters")

    return parser

config_dic = {
    # data parameters
    "vocab_file" :[str, ""],
    "data_file":[str, ""],
    "max_len":[int, 512],
    "vocab_size":[int, None],
    "n_segments":[int, 2],

    # model parameters
    "d_model":[int, default_d_model],
    "d_k":[int, default_d_model//default_h],
    "d_v":[int, default_d_model//default_h],
    "num_heads":[int, default_h],
    "num_encoder_layers":[int, 6],
    "dim_feedforward":[int, 2048],
    "dropout_rate":[float, 0.1],
    # Transformers with Independent Mechanisms (TIM) model parameters
    "n_s":[int, 2],
    "H":[int, default_h],
    "H_c":[int, default_h],
    "tim_layers_pos":[str, ""],

    # log parameters
    "model_file" : [str, ""],
    "save_dir":[str, ""],
    "log_dir":[str, ""],

    # Training parameters
    "pretrain":[bool, True],
    "data_parallel":[bool, False],
    "seed": [int, 3431],
    "batch_size":  [int, 32],
    "n_epochs":  [int, 2],
    ## pretrain : MLM and NSP
    "max_pred":[int, 20],
    "mask_prob":[float, 0.15],
    ## fine-tune : classification...
    "mode" : [str, "train"],
    "pretrain_file":[str, ""],
    "task" : [str, ""],

    # optimizer
    "lr": [float, 1e-4],
    "warmup": [float, 0.1],
    "save_steps":  [int, 10000],
    "total_steps":  [int, 1000000],

    # configuration file
    "config_file":[str, ""]
}


def from_config_file(params):
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

    return params