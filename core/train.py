# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

import copy
import gc

from src import __main__, get_trainer_evaluator, one_epoch, end_of_epoch
from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import initialize_exp, set_seeds

from params import get_parser, from_config_file

def main(params):
    
    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)
    
    # initialize the experiment
    meta_params = copy.deepcopy(params).meta_params
    params.meta_params = "..." # to long to be log
    logger = initialize_exp(params)
    params.meta_params = meta_params

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()
    trainer, evaluator, eval_trainers = get_trainer_evaluator(params, logger) 

    # evaluation
    if params.eval_only:
        end_of_epoch(params = params, logger = logger, trainer = trainer, evaluator = evaluator)
        if params.eval_tasks:
            logger.info("============ Evaluation task ============")
            for eval_task in params.eval_tasks :
                logger.info("============ %s ============" % eval_task)
                end_of_epoch(
                            params = eval_trainers[eval_task]["params"], 
                            logger = logger, 
                            trainer = eval_trainers[eval_task]['trainer'], 
                            evaluator = eval_trainers[eval_task]['evaluator'], 
                            eval_task = eval_task 
                )
        exit()
    
    # language model training
    for _ in range(params.max_epoch):
        
        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        one_epoch(trainer, params)

        if params.eval_tasks:
            logger.info("============ Evaluation task ============")
            for eval_task in params.eval_tasks :
                logger.info("============ %s ============" % eval_task)
                if params.encoder_only:
                    eval_trainers[eval_task]['trainer'].model = copy.deepcopy(trainer.model)
                else :
                    eval_trainers[eval_task]['trainer'].encoder = copy.deepcopy(trainer.encoder)
                    eval_trainers[eval_task]['trainer'].decoder = copy.deepcopy(trainer.decoder)
                one_epoch(eval_trainers[eval_task]['trainer'], eval_trainers[eval_task]["params"], eval_task = eval_task)
            
        logger.info("============ End of epoch %i ============" % trainer.epoch)

        end_of_epoch(params = params, logger = logger, trainer = trainer, evaluator = evaluator)
        
        if params.eval_tasks:
            logger.info("============ Evaluation task ============")
            for eval_task in params.eval_tasks :
                end_of_epoch(
                            params = eval_trainers[eval_task]["params"], 
                            logger = logger, 
                            trainer = eval_trainers[eval_task]['trainer'], 
                            evaluator = eval_trainers[eval_task]['evaluator'], 
                            eval_task = eval_task 
                )

        # our
        logger.info("============ garbage collector collecting %d ..." % gc.collect())
        

if __name__ == '__main__':

    # generate parser / parse parameters
    params = get_parser().parse_args()

    params = from_config_file(params)

    set_seeds(params.random_seed)

    if params.device not in ["cpu", "cuda"] :
        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else :
        params.device = torch.device(params.device)
    
    __main__(params)
    
    # run experiment
    main(params)
