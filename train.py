import argparse
import os.path
import time
import sys
# import wandb
# from visualdl import LogWriter


parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument("--baseline", action="store_true", help="Run baseline model training")
parser.add_argument('--dataset', type=str)
parser.add_argument("--data_dir", type=str, default=".")
parser.add_argument("--output_dir", type=str,default="test_outputs/")
parser.add_argument('--encoder', type=str, choices=[ 'average', 'lstm','simple-rnn','bert'], default="lstm")
parser.add_argument('--attention', type=str, choices=['tanh', 'frozen', 'pre-loaded'], required=False)
parser.add_argument('--n_iters', type=int, required=False, default=40)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--gold_label_dir', type=str, required=False)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--lmbda', type=float, required=False)
parser.add_argument('--adversarial', action='store_const', required=False, const=True)
parser.add_argument('--ours', action='store_true')

# parser.add_argument('--pgd_random_start', action='store_true')
parser.add_argument('--pgd_radius', type=float,default=0.1)
parser.add_argument('--pgd_step', type=float,default=10)
parser.add_argument('--pgd_step_size', type=float,default=0.02)
parser.add_argument('--pgd_norm_type', type=str,default="l-infty")

parser.add_argument('--x_pgd_radius', type=float,default=0.05)
parser.add_argument('--x_pgd_step', type=float,default=10)
parser.add_argument('--x_pgd_step_size', type=float,default=0.01)
parser.add_argument('--x_pgd_norm_type', type=str,default="l-infty")

parser.add_argument('--lambda_1', type=float, default=1e-2)
parser.add_argument('--lambda_2', type=float, default=1e-2)
parser.add_argument('--exp_name', type=str, default="debug")
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--topk_prox_metric', type=str, choices=['l1', 'l2',"kl-full", 'jsd-full',"kl-topk", 'jsd-topk'], default='l1')

parser.add_argument("--auto_hyperparameter", action="store_true", help="auto set the hyperparameters")

parser.add_argument("--wandb_entity", type=str, default="yixin")

args, extras = parser.parse_known_args()
args.extras = extras
args.command = ' '.join(['python'] + sys.argv)


from configurations import lambda_config
if args.auto_hyperparameter:
    args.lambda_1 =lambda_config[args.dataset][args.encoder]['lambda_1']
    args.lambda_2 =lambda_config[args.dataset][args.encoder]['lambda_2']
    args.x_pgd_radius =lambda_config[args.dataset][args.encoder]['x_pgd_radius']
    args.pgd_radius =lambda_config[args.dataset][args.encoder]['pgd_radius']


# auto set
args.pgd_step_size = args.pgd_radius / args.pgd_step * 2
args.x_pgd_step_size = args.x_pgd_radius / args.x_pgd_step * 2

from Trainers.DatasetBC import auto_load_dataset

import torch
import numpy as np

if args.adversarial :
    exp_name = '+'.join((args.encoder, 'adversarial'))
elif args.ours:
    exp_name = '+'.join((args.encoder, 'ours'))
else :
    exp_name = '+'.join((args.encoder, args.attention))

from attention.common_code.common import get_latest_model
base = "./test_outputs"
att = '+'.join((args.encoder, 'tanh'))

print(args.baseline)
if not args.baseline:
    # get the least recent baseline model
    print(f'{os.path.join(base,args.dataset,att)}')
    args.gold_label_dir = get_latest_model(f'{os.path.join(base,args.dataset,att)}')
    print(
        args.gold_label_dir
    )


import wandb
wandb.init(project="XAI-NLP-NEW", entity=args.wandb_entity,config=args)
wandb.log(vars(args))


# check that have provided a data directory to load attentions/predictions from
if (args.attention == 'pre-loaded' or args.adversarial) and not args.gold_label_dir :
    raise Exception("You must specify a gold-label directory for attention distributions") 

#check that have provided the correct dir:
if args.gold_label_dir and args.dataset.lower() not in args.gold_label_dir and args.dataset not in args.gold_label_dir :
    raise Exception("Gold-attention labels directory does not match specified dataset")

# add check for lmbda value if adversarial model
if args.adversarial and not args.lmbda :
    raise Exception("Must specify a lambda value for the adversarial model")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.adversarial or args.ours :
    args.frozen_attn = False
    args.pre_loaded_attn = False
elif args.attention == 'frozen' :
    args.frozen_attn = True
    args.pre_loaded_attn = False
elif args.attention == 'tanh' :
    args.frozen_attn = False
    args.pre_loaded_attn = False
elif args.attention == 'pre-loaded':
    args.frozen_attn = False
    args.pre_loaded_attn = True
else :
    raise LookupError("Attention not found ...")


dataset = auto_load_dataset(dataset_name=args.dataset, args=args)

if args.output_dir is not None :
    dataset.output_dir = args.output_dir

from attention.configurations import generate_config
from attention.Trainers.TrainerBC import Trainer, Evaluator
config = generate_config(dataset, args, exp_name)
trainer = Trainer(dataset, args, config=config)

dirname = trainer.model.save_values(save_model=False)
print("DIRECTORY:", dirname)
if args.ours:
    from attention.attack import PGDAttacker
    PGDer = PGDAttacker(
        radius=args.pgd_radius, steps=args.pgd_step, step_size=args.pgd_step_size, random_start= \
        True, norm_type=args.pgd_norm_type, ascending=True
    )
    trainer.PGDer = PGDer
    X_PGDer = PGDAttacker(
        radius=args.x_pgd_radius, steps=args.x_pgd_step, step_size=args.x_pgd_step_size, random_start= \
            True, norm_type=args.x_pgd_norm_type, ascending=True
    )
    trainer.PGDer = PGDer
    trainer.X_PGDer = X_PGDer
    trainer.train_ours(dataset.train_data, dataset.test_data, args,dataset)
else:
    trainer.train_standard(dataset.train_data, dataset.test_data, args, save_on_metric=dataset.save_on_metric)
print('####################################')
# print("TEST RESULTS FROM BEST MODEL")
evaluator = Evaluator(dataset, trainer.model.dirname, args)
final_metric,_,_ = evaluator.evaluate(dataset.test_data, save_results=True)
wandb.log({
    "final_metric":final_metric
})
wandb.finish()
# os.kill(os.getpid(), signal.SIGINT)
import os
import signal
os.kill(os.getpid(), signal.SIGKILL)