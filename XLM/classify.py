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

from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import initialize_exp, set_seeds

from src.classification import build_model, load_dataset, get_loss, end_of_epoch
from src.classification.trainer import Trainer
from src.classification.params import add_argument, check_parameters

from params import get_parser, from_config_file
    
def main(params):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)
    
    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()
    
    # Model
    model = build_model(params, logger)
    
    # Data 
    train_dataset, val_dataset = load_dataset(params, logger, model)
        
    # optimizers
    optimizers = model.get_optimizers(params) if not params.eval_only else []
    
    # Trainer
    trainer = Trainer(params, model, optimizers, train_dataset, val_dataset, logger)
    
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
    
    set_seeds(params.random_seed)

    if params.device not in ["cpu", "cuda"] :
        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else :
        params.device = torch.device(params.device)

    # check parameters    
    check_parameters(params)
    
    # run experiment
    main(params)
    