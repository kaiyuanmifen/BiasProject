# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import json
import random
import argparse 
import copy
import gc
import os

from src.data import check_data_params, load_data
from src.model import check_model_params, build_model

import configs.types as types

def get_parser():
    r"""
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    parser.add_argument("--use_lang_emb", type=bool_flag, default=True,
                        help="Use language embedding")

    parser.add_argument("--random_seed", type=int, default=0,
                        help="random seed for reproductibility")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="change to cpu if cuda is not available")
    parser.add_argument("--config_file", type=str, default="", 
                        help="")

def main(params) :
    # load data
    data = load_data(params)

    # build model
    if params.encoder_only:
        model = build_model(params = params, dico = data['dico'])
    else:
        encoder, decoder = build_model(params = params, dico = data['dico'])

    # build trainer, reload potential checkpoints / build evaluator
    if params.encoder_only:
        #trainer = SingleTrainer(model, data, params)
        #evaluator = SingleEvaluator(trainer, data, params)
    else:
        trainer = EncDecTrainer(encoder, decoder, data, params)
        evaluator = EncDecEvaluator(trainer, data, params)

    for _ in range(params.max_epoch):
        one_epoch(trainer, params)
        end_of_epoch(trainer = trainer, evaluator = evaluator, params = params)

if __name__ == '__main__':

   	# generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    torch.manual_seed(params.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if params.device not in ["cpu", "cuda"] :
        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else :
        params.device = torch.device(params.device)

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
    
    check_data_params(params)
    check_model_params(params)  

    main(params)