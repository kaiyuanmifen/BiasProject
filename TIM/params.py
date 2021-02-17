import argparse 

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--vocab_file", type=str, default="", help="")
    parser.add_argument("--data_file", type=str, default="", help="")
    parser.add_argument("--model_file", type=str, default="", help="")
    parser.add_argument("--save_dir", type=str, default="", help="")
    parser.add_argument("--log_dir", type=str, default="", help="")
    parser.add_argument("--data_parallel", type=bool_flag, default=False, help="")
    
    # data parameters
    parser.add_argument("--max_len", type=int, default=512, help="maximum length of tokens")
    parser.add_argument("--max_pred", type=int, default=20, help="")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="")

    # model parameters
    parser.add_argument("--d_model", type=int, default=512, help="")
    parser.add_argument("--d_k", type=int, default=512, help="")
    parser.add_argument("--d_v", type=int, default=512, help="")
    parser.add_argument("--num_heads", type=int, default=8, help="")    
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="")
    parser.add_argument("--vocab_size", type=int, default=None, help="")
    parser.add_argument("--n_segments", type=int, default=2, help="")

    # classif
    parser.add_argument("--mode", type=str, default="train", help="")
    parser.add_argument("--pretrain_file", type=str, default="", help="")
    # https://stackoverflow.com/questions/40324356/python-argparse-choices-with-a-default-choice/40324463
    parser.add_argument('--task', default='bias_classification', const='sentiment_analysis', nargs='?',
                                  choices=['bias_classification','sentiment_analysis', 'mrpc', 'mnli'], help='')

    # tim model parameters
    parser.add_argument("--n_s", type=int, default=2, help="")
    parser.add_argument("--H", type=int, default=8, help="")
    parser.add_argument("--H_c", type=int, default=8, help="")
    parser.add_argument("--tim_layers_pos", type=str, default="", help="tim layers position : 1,2,...")

    # pretrain
    parser.add_argument("--seed", type=int, default=3431, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--lr", type=float, default=1e-4, help="")
    parser.add_argument("--n_epochs", type=int, default=2, help="")
    
    # optim
    parser.add_argument("--warmup", type=float, default=0.1, help="")
    parser.add_argument("--save_steps", type=int, default=10000, help="")
    parser.add_argument("--total_steps", type=int, default=1000000, help="")
 

    parser.add_argument("--config_file", type=str, default="", help="")

    return parser