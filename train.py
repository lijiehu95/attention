import argparse
import time
import sys
import wandb


parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str)
parser.add_argument("--data_dir", type=str, default=".")
parser.add_argument("--output_dir", type=str,default="test_adv_outputs/")
parser.add_argument('--encoder', type=str, choices=[ 'average', 'lstm'], default="lstm")
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

args, extras = parser.parse_known_args()
args.extras = extras
args.command = ' '.join(['python'] + sys.argv)


# auto set
args.pgd_step_size = args.pgd_radius / args.pgd_step * 2
args.x_pgd_step_size = args.x_pgd_radius / args.x_pgd_step * 2


wandb.init(project="XAI-NLP", entity="yixin",config=args)
wandb.log(vars(args))

from attention.Trainers.DatasetBC import datasets
from attention.ExperimentsBC import train_dataset_on_encoders

import torch
import numpy as np

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
elif args.attention == 'pre-loaded': # not an adversarial model
    args.frozen_attn = False
    args.pre_loaded_attn = True
else :
    raise LookupError("Attention not found ...")

dataset = datasets[args.dataset](args)

if args.output_dir is not None :
    dataset.output_dir = args.output_dir

if args.adversarial :
    exp_name = '+'.join((args.encoder, 'adversarial'))
elif args.ours:
    exp_name = '+'.join((args.encoder, 'ours'))
else :
    exp_name = '+'.join((args.encoder, args.attention))

start = time.time()
train_dataset_on_encoders(dataset, args, exp_name)
print("TOTAL ELAPSED TIME: %f HOURS OR %f MINUTES" % (((time.time() - start)/60/60), ((time.time() - start)/60)))
# _python_exit()
# sys.exit()
wandb.log({
    "finish":'True'
})
import os
import signal
os.kill(os.getpid(), signal.SIGKILL)


