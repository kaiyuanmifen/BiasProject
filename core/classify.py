# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Test : 6-layer, 512-hidden-nodes (64*8), 8-attention-heads (google bert : ___M parameters)
# BERT-Base: 12-layer, 768-hidden-nodes (64*12), 12-attention-heads (google bert : 110M parameters)
# BERT-Large: 24-layer, 1024-hidden-nodes (64*16), 16-attention-heads, (google bert :340M parameters)

import torch
import os
import copy

from src import get_trainer_evaluator, __main__
from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import initialize_exp, set_seeds

from src.classification import build_model, load_dataset, get_loss, end_of_epoch
from src.classification.trainer import Trainer
from src.classification.params import add_argument, check_parameters

from params import get_parser, from_config_file
    
def main(params, params_pretrain):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)
    
    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()
        
    # Model
    if params.pretrain :
        for attr_name in ['n_gpu_per_node', 'multi_gpu', 'is_master']:
            setattr(params_pretrain, attr_name, getattr(params, attr_name))
        pre_trainer, evaluator, _ = get_trainer_evaluator(params_pretrain, logger)

    else :
        pre_trainer, evaluator = None, None
        
    model = build_model(params, logger, pre_trainer = pre_trainer)
    
    # Data 
    train_dataset, val_dataset = load_dataset(params, logger, model)
        
    # optimizers
    optimizers = model.get_optimizers(params) if not params.eval_only else []
        
    # Trainer
    trainer = Trainer(params, model, optimizers, train_dataset, val_dataset, logger, pre_trainer, evaluator)
    
    if params.pretrain :
        assert id(trainer.model.embedder.model) == id(pre_trainer.model)
    
    # Run train/evaluation
    logger.info("")
    if not params.eval_only :
        trainer.train(get_loss, end_of_epoch)
    else :
        trainer.eval(get_loss, end_of_epoch)
        
if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    parser = add_argument(parser)
    params = parser.parse_args()
    params = from_config_file(params)
    
    if params.device not in ["cpu", "cuda"] :
        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else :
        params.device = torch.device(params.device)
    
    set_seeds(params.random_seed)
    params.pretrain = os.path.isfile(params.pretrain_config)
    if params.pretrain :
        params_pretrain = from_config_file(copy.deepcopy(params), config_file = params.pretrain_config)
        __main__(params_pretrain)
    else :
        params_pretrain = None
    
    # check parameters    
    check_parameters(params)
    
    # run experiment
    main(params, params_pretrain)
    