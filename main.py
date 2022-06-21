import argparse
import sys
import torch
import numpy as np
import time
from trainer import Trainer, Evaluator

parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, default="sst")
parser.add_argument("--data_dir", type=str, default=".")
parser.add_argument("--output_dir", type=str,default="test_outputs/")
parser.add_argument('--encoder', type=str, choices=['bilstm', 'linear', 'lstm'], default="bilstm")
parser.add_argument('--attention', type=str, choices=['tanh', 'frozen', 'pre-loaded'], default="tanh")
parser.add_argument('--n_iters', type=int, required=False, default=2)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--gold_label_dir', type=str, required=False)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--lmbda', type=float, required=False)
parser.add_argument("--mode", type=str, choices=['adv', 'std', 'adv-ours'], default='std')
parser.add_argument('--bert_encoder', type=str, choices=['bert-base-uncased'],default='bert-base-uncased')

# parser.add_argument('--adversarial', action='store_const', required=False, const=True)
parser.add_argument('--pgd_radius', type=float,default=0.2)
parser.add_argument('--pgd_step', type=float,default=10)
parser.add_argument('--pgd_step_size', type=float,default=0.04)
parser.add_argument('--pgd_norm_type', type=str,default="l-infty")
parser.add_argument('--lambda_1', type=float, default=1e-2)
parser.add_argument('--lambda_2', type=float, default=1e-2)
parser.add_argument('--exp_name', type=str, default="debug")
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--topk_prox_metric', type=str, choices=['l1', 'l2',"kl-full", 'jsd-full',"kl-topk", 'jsd-topk'], default='l1')

parser.add_argument(
        '-f',
        '--file',
        help='Path for input file. First line should contain number of lines to search in'
    )

args, extras = parser.parse_known_args()
args.extras = extras
args.command = ' '.join(['python'] + sys.argv)

# wandb.init(project="XAI-NLP", entity="yixin",config=args)
# wandb.log(vars(args))

# # check that have provided a data directory to load attentions/predictions from
# if (args.attention == 'pre-loaded' or args.adversarial) and not args.gold_label_dir :
#     raise Exception("You must specify a gold-label directory for attention distributions")
#
# #check that have provided the correct dir:
# if args.gold_label_dir and args.dataset.lower() not in args.gold_label_dir and args.dataset not in args.gold_label_dir :
#     raise Exception("Gold-attention labels directory does not match specified dataset")
#
# # add check for lmbda value if adversarial model
# if args.adversarial and not args.lmbda :
#     raise Exception("Must specify a lambda value for the adversarial model")


# frozen_attn means uniform
# if args.adversarial or args.ours :
#     args.frozen_attn = False
#     args.pre_loaded_attn = False
# elif args.attention == 'frozen' :
#     args.frozen_attn = True
#     args.pre_loaded_attn = False
# elif args.attention == 'tanh' :
#     args.frozen_attn = False
#     args.pre_loaded_attn = False
# elif args.attention == 'pre-loaded': # not an adversarial model
#     args.frozen_attn = False
#     args.pre_loaded_attn = True
# else :
#     raise LookupError("Attention not found ...")

if __name__ == '__main__':

    ENCODER="lstm"
    N_iter = 2
    MODE = 'std'
    ATT = 'tanh'
    args.n_iters = N_iter
    args.mode = MODE
    args.attention = ATT
    args.encoder = ENCODER

    if args.attention == 'frozen':
        args.frozen_attn = True
    else:
        args.frozen_attn = False
    if args.attention == 'pre-loaded':
        args.pre_loaded_attn = True
    else:
        args.pre_loaded_attn = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == 'adv':
        exp_name = '+'.join((args.encoder, 'adversarial'))
    else:
        exp_name = '+'.join((args.encoder, args.attention))

    from dataset_load import datasets
    # load data
    dataset = datasets[args.dataset](args)
    if args.output_dir is not None:
        dataset.output_dir = args.output_dir

    from config import generate_config
    config_setting = generate_config(dataset, args, exp_name)
    start = time.time()
    trainer = Trainer(dataset, args, config=config_setting)
    dirname = trainer.model.save_values(save_model=False) # mk log dir and save config
    print("DIRECTORY:", dirname)
    if args.mode == "adv" :
        trainer.train_adversarial(dataset.train_data, dataset.test_data, args)
    # elif args.ours:
    #     from attention.attack import  PGDAttacker
    #     PGDer = PGDAttacker(
    #         radius=args.pgd_radius, steps=args.pgd_step, step_size=args.pgd_step_size, random_start= \
    #         True, norm_type=args.pgd_norm_type, ascending=True
    #     )
    #     trainer.PGDer = PGDer
    #     trainer.train_ours(dataset.train_data, dataset.test_data, args)
    elif args.mode == "std":
        trainer.train_standard(dataset.train_data, dataset.test_data, args, save_on_metric=dataset.save_on_metric)
    print('####################################')
    print("TEST RESULTS FROM BEST MODEL")
    evaluator = Evaluator(dataset, trainer.model.dirname, args)
    _ = evaluator.evaluate(dataset.test_data, save_results=True)
    print("TOTAL ELAPSED TIME: %f HOURS OR %f MINUTES" % (((time.time() - start)/60/60), ((time.time() - start)/60)))



