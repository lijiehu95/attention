#!/usr/bin/env python
# coding: utf-8

# ## 训练参数

# In[59]:


import argparse
import time
import sys
# from Trainers.DatasetBC import datasets
from ExperimentsBC import train_dataset_on_encoders
import torch
import numpy as np

parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, default="sst")
parser.add_argument("--data_dir", type=str, default=".")
parser.add_argument("--output_dir", type=str,default="test_outputs/")
parser.add_argument('--encoder', type=str, choices=['lstm', 'average'], default="lstm")
parser.add_argument('--attention', type=str, choices=['tanh', 'frozen', 'pre-loaded'], default="tanh")
parser.add_argument('--n_iters', type=int, required=False, default=80)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--gold_label_dir', type=str, required=False)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--lmbda', type=float, required=False)
parser.add_argument('--adversarial', action='store_const', required=False, const=True)
parser.add_argument('--ours', action='store_true')

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
#
# wandb.init(project="XAI-NLP", entity="yixin",config=args)
# wandb.log(vars(args))

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

# frozen_attn means uniform

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


if args.adversarial or args.ours :
    exp_name = '+'.join((args.encoder, 'adversarial'))
else :
    exp_name = '+'.join((args.encoder, args.attention))


# In[60]:


# exp_name


# ## 数据读入

# In[61]:


import os
import pickle
import numpy as np
import json
from attention.preprocess import vectorizer

def sortbylength(X, y) :
    len_t = np.argsort([len(x) for x in X])
    X1 = [X[i] for i in len_t]
    y1 = [y[i] for i in len_t]
    return X1, y1

def filterbylength(X, y, min_length = None, max_length = None) :
    lens = [len(x)-2 for x in X]
    min_l = min(lens) if min_length is None else min_length
    max_l = max(lens) if max_length is None else max_length

    idx = [i for i in range(len(X)) if len(X[i]) > min_l+2 and len(X[i]) < max_l+2]
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    return X, y

def set_balanced_pos_weight(dataset) :
    y = np.array(dataset.train_data.y)
    dataset.pos_weight = [len(y) / sum(y) - 1] # N/P

class DataHolder() :
    def __init__(self, X, y, y_attn=None, true_pred=None) :
        self.X = X
        self.y = y
        self.gold_attns = y_attn
        self.true_pred = true_pred
        self.attributes = ['X', 'y', 'gold_attns', 'true_pred']


class Dataset() :
    def __init__(self, name, path, min_length=None, max_length=None, args=None) :
        self.name = name
        if args is not None and hasattr(args, 'data_dir') :
            path = os.path.join(args.data_dir, path)

        self.vec = pickle.load(open(path, 'rb'))

        X, Xt = self.vec.seq_text['train'], self.vec.seq_text['test'] # these are lists (of lists) of num. insts-length (NOT PADDED)
        y, yt = self.vec.label['train'], self.vec.label['test']

        X, y = filterbylength(X, y, min_length=min_length, max_length=max_length)
        Xt, yt = filterbylength(Xt, yt, min_length=min_length, max_length=max_length)
        Xt, yt = sortbylength(Xt, yt)

        if args.pre_loaded_attn or args.adversarial or args.ours :
            # these are lists of lists, with some residual padding
            y_attn = json.load(open(os.path.join(args.gold_label_dir, 'train_attentions_best_epoch.json'), 'r'))
            yt_attn = json.load(open(os.path.join(args.gold_label_dir, 'test_attentions_best_epoch.json'), 'r'))

            true_pred = json.load(open(os.path.join(args.gold_label_dir, 'train_predictions_best_epoch.json'), 'r'))
            true_pred_t = json.load(open(os.path.join(args.gold_label_dir, 'test_predictions_best_epoch.json'), 'r'))
            true_pred = [e[0] for e in true_pred]
            true_pred_t = [e[0] for e in true_pred_t] #these are lists of num. insts-length

            #trim padding from static attentions
            new_attns = []
            for e, a in zip(X, y_attn):
                tmp = [0] + [el for el in a if el != 0] + [0]
                assert len(tmp) == len(e)
                new_attns.append(tmp)
            y_attn = new_attns

            #do the same for test
            new_attns = []
            for e, a in zip(Xt, yt_attn):
                tmp = [0] + [el for el in a if el != 0] + [0]
                assert len(tmp) == len(e)
                new_attns.append(tmp)
            yt_attn = new_attns

            self.train_data = DataHolder(X, y, y_attn, true_pred)
            self.test_data = DataHolder(Xt, yt, yt_attn, true_pred_t)

        else :
            self.train_data = DataHolder(X, y)
            self.test_data = DataHolder(Xt, yt)

        if args is not None and hasattr(args, 'hidden_size') :
            self.hidden_size = args.hidden_size

        self.output_size = 1
        self.save_on_metric = 'roc_auc'
        self.keys_to_use = {
            'roc_auc' : 'roc_auc',
            'pr_auc' : 'pr_auc'
        }

        self.bsize = 32
        if args is not None and hasattr(args, 'output_dir') :
            self.basepath = args.output_dir


########################################## Dataset Loaders ################################################################################

def SST_dataset(args=None) :
    dataset = Dataset(name='sst', path='preprocess/SST/vec_sst.p', min_length=5, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def IMDB_dataset(args=None) :
    dataset = Dataset(name='imdb', path='preprocess/IMDB/vec_imdb.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def News20_dataset(args=None) :
    dataset = Dataset(name='20News_sports', path='preprocess/20News/vec_20news_sports.p', min_length=6, max_length=500, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def ADR_dataset(args=None) :
    dataset = Dataset(name='tweet', path='preprocess/Tweets/vec_adr.p', min_length=5, max_length=100, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def Anemia_dataset(args=None) :
    dataset = Dataset(name='anemia', path='preprocess/MIMIC/vec_anemia.p', max_length=4000, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def Diabetes_dataset(args=None) :
    dataset = Dataset(name='diabetes', path='preprocess/MIMIC/vec_diabetes.p', min_length=6, max_length=4000, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def AGNews_dataset(args=None) :
    dataset = Dataset(name='agnews', path='preprocess/ag_news/vec_agnews.p', args=args)
    set_balanced_pos_weight(dataset)
    return dataset

datasets = {
    "sst" : SST_dataset,
    "imdb" : IMDB_dataset,
    "20News_sports" : News20_dataset,
    "tweet" : ADR_dataset ,
    "Anemia" : Anemia_dataset,
    "Diabetes" : Diabetes_dataset,
    "AgNews" : AGNews_dataset
}


# In[62]:


dataset = datasets[args.dataset](args)

if args.output_dir is not None :
    dataset.output_dir = args.output_dir


# In[67]:


# max([len(dataset.train_data.X[i]) for i in range(len(dataset.train_data.X))])
y_attn = json.load(open(os.path.join("/Users/apple/Desktop/workspace/research_project/attention/test_outputs/sst/lstm+tanh/Mon_Jun__6_13:13:30_2022", 'train_attentions_best_epoch.json'), 'r'))


# In[75]:


# len(y_attn[6])
# y_attn[6][-1]


# ## 日志相关

# In[22]:


import os
import git

def generate_config(dataset, args, exp_name) :

    repo = git.Repo(search_parent_directories=True)

    if args.encoder == 'lstm' :
        enc_type = 'rnn'
    elif args.encoder == 'average' :
        enc_type = args.encoder
    else :
        raise Exception("unknown encoder type")

    config = {
        'model' :{
            'encoder' : {
                'vocab_size' : dataset.vec.vocab_size,
                'embed_size' : dataset.vec.word_dim,
		'type' : enc_type,
		'hidden_size' : args.hidden_size
            },
            'decoder' : {
                'attention' : {
                    'type' : 'tanh'
                },
                'output_size' : dataset.output_size
            }
        },
        'training' : {
            'bsize' : dataset.bsize if hasattr(dataset, 'bsize') else 32,
            'weight_decay' : 1e-5,
            'pos_weight' : dataset.pos_weight if hasattr(dataset, 'pos_weight') else None,
            'basepath' : dataset.basepath if hasattr(dataset, 'basepath') else 'outputs',
            'exp_dirname' : os.path.join(dataset.name, exp_name)
        },
        'git_info' : {
            'branch' : repo.active_branch.name,
            'sha' : repo.head.object.hexsha
        },
        'command' : args.command
    }

    if args.encoder == 'average' :
    	config['model']['encoder'].update({'projection' : True, 'activation' : 'tanh'})

    return config


# ## 模型定义

# In[23]:


#### Attention
from torch import nn
from allennlp.common import Registrable

def masked_softmax(attn_odds, masks) :
    attn_odds.masked_fill_(masks.bool(), -float('inf'))
    attn = nn.Softmax(dim=-1)(attn_odds)
    return attn

class Attention(nn.Module, Registrable) :
    def forward(self, **kwargs) :
        raise NotImplementedError("Implement forward Model")

@Attention.register('tanh')
class TanhAttention(Attention) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.attn1 = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)

    def forward(self, input_seq, hidden, masks) :
        #input_seq = (B, L), hidden : (B, L, H), masks : (B, L)

        attn1 = nn.Tanh()(self.attn1(hidden)) #(B, L, H//2) #
        attn2 = self.attn2(attn1).squeeze(-1) #(B, L)
        attn = masked_softmax(attn2, masks) #(B, L)

        return attn


# In[24]:


###### encoder

import torch
import torch.nn as nn

from attention.model.modelUtils import isTrue
from allennlp.common import Registrable
from allennlp.nn.activations import Activation

class Encoder(nn.Module, Registrable) :
    def forward(self, **kwargs) :
        raise NotImplementedError("Implement forward Model")

@Encoder.register('rnn')
class EncoderRNN(Encoder) :
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None) :
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.output_size = self.hidden_size * 2

    def forward(self, data) :
        seq = data.seq
        lengths = data.lengths
        embedding = self.embedding(seq) #(B, L, E)
        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (h, c) = self.rnn(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0) # [batch_size, seq_len, feature]

        data.hidden = output
        data.last_hidden = torch.cat([h[0], h[1]], dim=-1)

        if isTrue(data, 'keep_grads') :
            data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()


@Encoder.register("average")
class EncoderAverage(Encoder) :
    def __init__(self,  vocab_size, embed_size, projection, hidden_size=None, activation:Activation=Activation.by_name('linear'), pre_embed=None) :
        super(EncoderAverage, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        if projection :
            self.projection = nn.Linear(embed_size, hidden_size)
            self.output_size = hidden_size
        else :
            self.projection = lambda s : s
            self.output_size = embed_size

        self.activation = activation

    def forward(self, data) :
        seq = data.seq
        lengths = data.lengths
        embedding = self.embedding(seq) #(B, L, E)

        output = self.activation(self.projection(embedding)) #(B, L, H)
        h = output.mean(1) #(B, H)

        data.hidden = output
        data.last_hidden = h

        if isTrue(data, 'keep_grads') :
            data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()


# In[25]:


####### decoder
from allennlp.common.from_params import FromParams
import torch
import torch.nn as nn
from typing import Dict
from allennlp.common import Params

# from attention.model.modules.Attention import Attention
from attention.model.modelUtils import isTrue, BatchHolder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AttnDecoder(nn.Module, FromParams) :
    def __init__(self, hidden_size:int,
                       attention:Dict,
                       output_size:int = 1,
                       use_attention:bool = True) :
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_1 = nn.Linear(hidden_size, output_size)

        attention['hidden_size'] = self.hidden_size
        self.attention = Attention.from_params(Params(attention))

        self.use_attention = use_attention

    def decode(self, predict) :
        predict = self.linear_1(predict)
        return predict

    def forward(self, data:BatchHolder, revise_att=None) :
        if self.use_attention :
            output = data.hidden
            mask = data.masks

            attn = self.attention(data.seq, output, mask) # (B, L)

            if revise_att is not None:
                attn = revise_att

            context = (attn.unsqueeze(-1) * output).sum(1) # (B, H)
            data.attn = attn
        else :
            context = data.last_hidden

        predict = self.decode(context)
        data.predict = predict


class FrozenAttnDecoder(AttnDecoder) :

    def forward(self, data:BatchHolder) :
        if self.use_attention :
            output = data.hidden
            attn = data.generate_frozen_uniform_attn()

            context = (attn.unsqueeze(-1) * output).sum(1)
            data.attn = attn
        else :
            context = data.last_hidden

        predict = self.decode(context)
        data.predict = predict


class PretrainedWeightsDecoder(AttnDecoder) :

    def forward(self, data:BatchHolder) :
        if self.use_attention :
            output = data.hidden
            attn = data.target_attn

            context = (attn.unsqueeze(-1) * output).sum(1)
            data.attn = attn
        else :
            context = data.last_hidden

        predict = self.decode(context)
        data.predict = predict


# ## 训练controller

# In[26]:


from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score
import numpy as np
from pandas.io.json.normalize import nested_to_record
from collections import defaultdict
import pandas as pd
import torch
from IPython.display import display

def tvd(predictions, targets): #accepts two numpy arrays of dimension: (num. instances, )
    return (0.5 * np.abs(predictions - targets)).sum()

def batch_tvd(predictions, targets): #accepts two Torch tensors... " "
    return (0.5 * torch.abs(predictions - targets)).sum()

def calc_metrics_classification(target, predictions, target_scores=None, jsd_score=None) :

    if target_scores is not None :
        assert predictions.squeeze(1).shape == target_scores.shape
        tvdist = tvd(predictions.squeeze(1), target_scores)

    if predictions.shape[-1] == 1 :
        predictions = predictions[:, 0]
        predictions = np.array([1 - predictions, predictions]).T

    predict_classes = np.argmax(predictions, axis=-1)

    if len(np.unique(target)) < 4 :
        rep = nested_to_record(classification_report(target, predict_classes, output_dict=True), sep='/')
    else :
        rep = {}
    rep.update({'accuracy' : accuracy_score(target, predict_classes)})

    if jsd_score :
        rep.update({'js_divergence' : jsd_score})
    if target_scores is not None :
        rep.update({'TVD' : tvdist})

    if predictions.shape[-1] == 2 :
        rep.update({'roc_auc' : roc_auc_score(target, predictions[:, 1])})
        rep.update({"pr_auc" : average_precision_score(target, predictions[:, 1])})
    return rep

def print_metrics(metrics, adv=False) :
    tabular = {k:v for k, v in metrics.items() if '/' in k}
    non_tabular = {k:v for k, v in metrics.items() if '/' not in k}
    print(non_tabular)

    d = defaultdict(dict)
    for k, v in tabular.items() :
        if not k.startswith('label_') :
            d[k.split('/', 1)[0]][k.split('/', 1)[1]] = v
        if '/1/' in k or 'auc' in k:
            d[k.split('/', 1)[0]][k.split('/', 1)[1]] = v

    df = pd.DataFrame(d)
    with pd.option_context('display.max_columns', 30):
        display(df.round(3))

    if adv :
        print("TVD:", metrics['TVD'])
        print("JS:", metrics['js_divergence'])


# In[27]:


import json
import os
import shutil
from copy import deepcopy

import torch
import torch.nn as nn
from allennlp.common import Params
from sklearn.utils import shuffle
from tqdm import tqdm
import time

# from attention.model.modules.Decoder import AttnDecoder, FrozenAttnDecoder, PretrainedWeightsDecoder
# from attention.model.modules.Encoder import Encoder
# from attention.common_code.metrics import batch_tvd

# from attention.model.modelUtils import BatchHolder, get_sorting_index_with_noise_from_lengths
# from attention.model.modelUtils import jsd as js_divergence


def isTrue(obj, attr):
    return hasattr(obj, attr) and getattr(obj, attr)


import numpy as np
import torch


def get_sorting_index_with_noise_from_lengths(lengths, noise_frac):
    if noise_frac > 0:
        noisy_lengths = [x + np.random.randint(np.floor(-x * noise_frac), np.ceil(x * noise_frac)) for x in lengths]
    else:
        noisy_lengths = lengths
    return np.argsort(noisy_lengths)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BatchHolder():
    def __init__(self, data, target_attn=None):
        maxlen = max([len(x) for x in data])
        self.maxlen = maxlen
        self.B = len(data)

        lengths = []
        expanded = []
        masks = []
        expanded_attn = []

        if target_attn:
            for d, ta in zip(data, target_attn):
                assert len(d) == len(ta)
                rem = maxlen - len(d)
                expanded.append(d + [0] * rem)
                lengths.append(len(d))
                masks.append([1] + [0] * (len(d) - 2) + [1] * (rem + 1))
                assert len([1] + [0] * (len(d) - 2) + [1] * (rem + 1)) == len(ta + [0] * rem) == len(d + [0] * rem)
                # also pad target attention:
                expanded_attn.append(ta + [0] * rem)
        else:
            for _, d in enumerate(data):
                rem = maxlen - len(d)
                expanded.append(d + [0] * rem)
                lengths.append(len(d))
                masks.append([1] + [0] * (len(d) - 2) + [1] * (rem + 1))

        self.lengths = torch.LongTensor(np.array(lengths)).to(device)
        self.seq = torch.LongTensor(np.array(expanded, dtype='int64')).to(device)
        self.masks = torch.ByteTensor(np.array(masks)).to(device)

        self.hidden = None
        self.predict = None
        self.attn = None

        if target_attn:
            self.target_attn = torch.FloatTensor(expanded_attn).to(device)
        self.inv_masks = ~self.masks

    def generate_frozen_uniform_attn(self):
        attn = np.zeros((self.B, self.maxlen))
        inv_l = 1. / (self.lengths.cpu().data.numpy() - 2)
        attn += inv_l[:, None]
        attn = torch.Tensor(attn).to(device)
        attn.masked_fill_(self.masks.bool(), 0)
        return attn


def kld(a1, a2):
    # (B, *, A), #(B, *, A)
    a1 = torch.clamp(a1, 0, 1)
    a2 = torch.clamp(a2, 0, 1)
    log_a1 = torch.log(a1 + 1e-10)
    log_a2 = torch.log(a2 + 1e-10)

    kld = a1 * (log_a1 - log_a2)
    kld = kld.sum(-1)

    return kld


def jsd(p, q):
    m = 0.5 * (p + q)
    jsd = 0.5 * (kld(p, m) + kld(q, m))  # for each instance in the batch

    return jsd.unsqueeze(-1)  # jsd.squeeze(1).sum()


# file_name = os.path.abspath(__file__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BC_Model():
    def __init__(self, configuration, args, pre_embed=None):
        configuration = deepcopy(configuration)
        self.configuration = deepcopy(configuration)

        configuration['model']['encoder']['pre_embed'] = pre_embed
        self.encoder = Encoder.from_params(
            Params(configuration['model']['encoder'])).to(device)

        self.frozen_attn = args.frozen_attn
        self.adversarial = args.adversarial
        self.pre_loaded_attn = args.pre_loaded_attn

        configuration['model']['decoder'][
            'hidden_size'] = self.encoder.output_size
        if self.frozen_attn:
            self.decoder = FrozenAttnDecoder.from_params(
                Params(configuration['model']['decoder'])).to(device)
        elif self.pre_loaded_attn:
            self.decoder = PretrainedWeightsDecoder.from_params(
                Params(configuration['model']['decoder'])).to(device)
        else:
            self.decoder = AttnDecoder.from_params(
                Params(configuration['model']['decoder'])).to(device)

        self.encoder_params = list(self.encoder.parameters())
        if not self.frozen_attn:
            self.attn_params = list([
                v for k, v in self.decoder.named_parameters()
                if 'attention' in k
            ])
        self.decoder_params = list([
            v for k, v in self.decoder.named_parameters()
            if 'attention' not in k
        ])

        self.bsize = configuration['training']['bsize']

        weight_decay = configuration['training'].get('weight_decay', 1e-5)
        self.encoder_optim = torch.optim.Adam(
            self.encoder_params,
            lr=0.001,
            weight_decay=weight_decay,
            amsgrad=True)
        if not self.frozen_attn:
            self.attn_optim = torch.optim.Adam(
                self.attn_params, lr=0.001, weight_decay=0, amsgrad=True)
        self.decoder_optim = torch.optim.Adam(
            self.decoder_params,
            lr=0.001,
            weight_decay=weight_decay,
            amsgrad=True)

        pos_weight = configuration['training'].get(
            'pos_weight', [1.0] * self.decoder.output_size)
        self.pos_weight = torch.Tensor(pos_weight).to(device)

        # setup either adversarial or std binary cross-entropy loss
        if self.adversarial:
            self.criterion = nn.KLDivLoss(
                size_average=None, reduce=None, reduction='sum').to(device)
            self.lmbda = args.lmbda
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)

        dirname = configuration['training']['exp_dirname']
        basepath = configuration['training'].get('basepath', 'outputs')
        self.time_str = time.ctime().replace(' ', '_')
        self.dirname = os.path.join(basepath, dirname, self.time_str)

        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2


        self.K = args.K
        self.topk_prox_metric = args.topk_prox_metric

    @classmethod
    def init_from_config(cls, dirname, args, **kwargs):
        config = json.load(open(dirname + '/config.json', 'r'))
        config.update(kwargs)
        obj = cls(config, args)
        obj.load_values(dirname)
        return obj

    # def train_ours(self,
    #           data_in,
    #           target_in,
    #           target_pred,
    #           target_attn_in,
    #           PGDer,train=True):
    #     sorting_idx = get_sorting_index_with_noise_from_lengths(
    #         [len(x) for x in data_in], noise_frac=0.1)
    #     data = [data_in[i] for i in sorting_idx]
    #     target = [target_in[i] for i in sorting_idx]
    #
    #     # print(target_pred)
    #
    #     target_pred = [target_pred[i] for i in sorting_idx]
    #     target_attn = [target_attn_in[i] for i in sorting_idx]
    #
    #     self.encoder.train()
    #     self.decoder.train()
    #
    #     from attention.utlis import AverageMeter
    #
    #     bsize = self.bsize
    #     N = len(data)
    #     loss_total = 0
    #     loss_orig_total = 0
    #     tvd_loss_total = 0
    #     topk_loss_total = 0
    #     pgd_tvd_loss_total = 0
    #     true_topk_loss_counter = AverageMeter()
    #
    #     batches = list(range(0, N, bsize))
    #     batches = shuffle(batches)
    #
    #     for n in tqdm(batches):
    #         batch_doc = data[n:n + bsize]
    #
    #         batch_target_attn = target_attn[n:n + bsize]
    #         batch_data = BatchHolder(batch_doc, batch_target_attn)
    #
    #         batch_target_pred = target_pred[n:n + bsize]
    #         batch_target_pred = torch.Tensor(batch_target_pred).to(device)
    #
    #         if len(batch_target_pred.shape) == 1:  # (B, )
    #             batch_target_pred = batch_target_pred.unsqueeze(
    #                 -1)  # (B, 1)
    #
    #         self.encoder(batch_data)
    #         self.decoder(batch_data)
    #
    #         batch_target = target[n:n + bsize]
    #         batch_target = torch.Tensor(batch_target).to(device)
    #
    #         if len(batch_target.shape) == 1:  #(B, )
    #             batch_target = batch_target.unsqueeze(-1)  #(B, 1)
    #
    #         # calculate adversarial loss (Section 4) if adversarial model
    #
    #         from attention.utlis import topk_overlap_loss,topK_overlap_true_loss
    #         topk_loss = topk_overlap_loss(batch_data.target_attn,
    #                                       batch_data.attn,K=self.K, metric=self.topk_prox_metric)
    #         topk_true_loss = topK_overlap_true_loss(batch_data.target_attn,
    #                                       batch_data.attn,K=self.K)
    #         true_topk_loss_counter.update(
    #             topk_true_loss,len(batch_doc)
    #         )
    #
    #         tvd_loss = batch_tvd(
    #             torch.sigmoid(batch_data.predict), batch_target_pred)
    #
    #         ### pgd loss
    #         def target_model(w, data, decoder):
    #             decoder(revise_att=w, data=data)
    #             return data.predict
    #
    #         def crit(gt, pred):
    #             return batch_tvd(torch.sigmoid(pred), gt)
    #
    #         # PGD generate the new weight
    #         new_att = PGDer.perturb(criterion=crit, att=batch_data.attn, data=batch_data                                     , decoder=self.decoder, batch_target_pred=batch_target_pred,
    #                                 target_model=target_model)
    #
    #         # output the prediction tvd of new weight and old weight
    #         self.decoder(batch_data, revise_att=new_att)
    #         new_out = batch_data.predict
    #         att_pgd_pred_tvd = batch_tvd(
    #             torch.sigmoid(new_out), batch_target_pred)
    #
    #         loss_orig = tvd_loss + self.lambda_1 * att_pgd_pred_tvd + self.lambda_2 * topk_loss
    #
    #
    #         weight = batch_target * self.pos_weight + (1 - batch_target)
    #         loss = (loss_orig * weight).mean(1).sum()
    #
    #         if hasattr(batch_data, 'reg_loss'):
    #             loss += batch_data.reg_loss
    #
    #         if train:
    #             self.encoder_optim.zero_grad()
    #             self.decoder_optim.zero_grad()
    #             if not self.frozen_attn:
    #                 self.attn_optim.zero_grad()
    #             loss.backward()
    #             self.encoder_optim.step()
    #             self.decoder_optim.step()
    #             if not self.frozen_attn:
    #                 self.attn_optim.step()
    #
    #         loss_total += float(loss.data.cpu().item())
    #         loss_orig_total += float(loss_orig.data.cpu().item())
    #         tvd_loss_total += float(tvd_loss.data.cpu().item())
    #         topk_loss_total += float(topk_loss.data.cpu().item())
    #         pgd_tvd_loss_total += float(
    #             att_pgd_pred_tvd.data.cpu().item())
    #
    #     return  loss_total, loss_orig_total, tvd_loss_total, topk_loss_total, pgd_tvd_loss_total, true_topk_loss_counter.average()

    def train(self,
              data_in,
              target_in,
              target_pred=None,
              target_attn_in=None,
              train=True):
        sorting_idx = get_sorting_index_with_noise_from_lengths(
            [len(x) for x in data_in], noise_frac=0.1)
        data = [data_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]

        if target_pred:
            target_pred = [target_pred[i] for i in sorting_idx]
            target_attn = [target_attn_in[i] for i in sorting_idx]

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)
        loss_total = 0
        loss_orig_total = 0
        tvd_loss_total = 0
        kl_loss_total = 0
        topk_loss_total = 0
        pgd_tvd_loss_total = 0

        batches = list(range(0, N, bsize))
        batches = shuffle(batches)

        for n in tqdm(batches):
            batch_doc = data[n:n + bsize]
            if target_pred:
                batch_target_attn = target_attn[n:n + bsize]
                batch_data = BatchHolder(batch_doc, batch_target_attn)

                batch_target_pred = target_pred[n:n + bsize]
                batch_target_pred = torch.Tensor(batch_target_pred).to(device)

                if len(batch_target_pred.shape) == 1:  #(B, )
                    batch_target_pred = batch_target_pred.unsqueeze(
                        -1)  #(B, 1)
            else:
                batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_target = target[n:n + bsize]
            batch_target = torch.Tensor(batch_target).to(device)

            if len(batch_target.shape) == 1:  #(B, )
                batch_target = batch_target.unsqueeze(-1)  #(B, 1)

            # calculate adversarial loss (Section 4) if adversarial model
            if target_pred:
                # if ours:
                #     from attention.utlis import topk_overlap_loss
                #     topk_loss = topk_overlap_loss(batch_data.target_attn.log(),
                #                                   batch_data.attn)
                #     tvd_loss = batch_tvd(
                #         torch.sigmoid(batch_data.predict), batch_target_pred)
                #
                #     ### pgd loss
                #     def target_model(w, data, decoder):
                #         decoder(revise_att=w, data=data)
                #         return data.predict
                #
                #     def crit(gt, pred):
                #         return batch_tvd(torch.sigmoid(pred), gt)
                #
                #     # PGD generate the new weight
                #     new_att = PDGer.perturb(criterion=crit, att=batch_data.attn, data=batch_data \
                #                           , decoder=self.decoder,batch_target_pred=batch_target_pred, target_model=target_model)
                #
                #     # output the prediction tvd of new weight and old weight
                #     self.decoder(batch_data, revise_att=new_att)
                #     new_out = batch_data.predict
                #     att_pgd_pred_tvd = batch_tvd(
                #         torch.sigmoid(new_out), batch_target_pred)
                #
                #     loss_orig = tvd_loss + self.lambda_1 * att_pgd_pred_tvd + self.lambda_2 * topk_loss
                #
                # else:
                kl_loss = self.criterion(batch_data.target_attn.log(),
                                         batch_data.attn)
                tvd_loss = batch_tvd(
                    torch.sigmoid(batch_data.predict), batch_target_pred)
                loss_orig = tvd_loss - self.lmbda * kl_loss

            # else calculate standard BCE loss
            else:
                loss_orig = self.criterion(batch_data.predict, batch_target)

            weight = batch_target * self.pos_weight + (1 - batch_target)
            loss = (loss_orig * weight).mean(1).sum()

            if hasattr(batch_data, 'reg_loss'):
                loss += batch_data.reg_loss

            if train:
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                if not self.frozen_attn:
                    self.attn_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
                if not self.frozen_attn:
                    self.attn_optim.step()

            loss_total += float(loss.data.cpu().item())

            if target_attn_in:
                # if ours:
                #     loss_orig_total += float(loss_orig.data.cpu().item())
                #     tvd_loss_total += float(tvd_loss.data.cpu().item())
                #     # kl_loss_total += float(kl_loss.data.cpu().item())
                #     topk_loss_total += float(topk_loss.data.cpu().item())
                #     pgd_tvd_loss_total += float(
                #         att_pgd_pred_tvd.data.cpu().item())
                # else:

                loss_orig_total += float(loss_orig.data.cpu().item())
                tvd_loss_total += float(tvd_loss.data.cpu().item())
                kl_loss_total += float(kl_loss.data.cpu().item())
        # if ours:
        #     return loss_total * bsize / N, loss_total, loss_orig_total, tvd_loss_total, topk_loss_total, pgd_tvd_loss_total
        # else:
        #     return loss_total * bsize / N, loss_total, loss_orig_total, tvd_loss_total, kl_loss_total
        return loss_total * bsize / N, loss_total, loss_orig_total, tvd_loss_total, kl_loss_total

    def evaluate(self, data, target_attn=None):
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []
        attns = []
        js_scores = []

        for n in tqdm(range(0, N, bsize)):
            batch_doc = data[n:n + bsize]
            if target_attn:
                batch_target_attn = target_attn[n:n + bsize]
                batch_data = BatchHolder(batch_doc, batch_target_attn)
            else:
                batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_data.predict = torch.sigmoid(batch_data.predict)
            if self.decoder.use_attention:  #and n == 0:
                attn = batch_data.attn.cpu().data.numpy()  #.astype('float16')
                attns.append(attn)

            predict = batch_data.predict.cpu().data.numpy(
            )  #.astype('float16')
            outputs.append(predict)

            if target_attn:
                #compute JS-divergence for batched attentions
                batch_jsdscores = js_divergence(
                    batch_data.target_attn, batch_data.attn).squeeze(
                        1).cpu().data.numpy()  #.astype('float16')
                js_scores.append(batch_jsdscores)

        outputs = [x for y in outputs for x in y]
        if self.decoder.use_attention:
            attns = [x for y in attns for x in y]
        if target_attn:
            js_score = sum([x for y in js_scores for x in y]).item()
        else:
            js_score = None

        return outputs, attns, js_score

    def save_values(self, use_dirname=None, save_model=True):
        if use_dirname is not None:
            dirname = use_dirname
        else:
            dirname = self.dirname
        os.makedirs(dirname, exist_ok=True)
        # shutil.copy2(file_name, dirname + '/')
        json.dump(self.configuration, open(dirname + '/config.json', 'w'))

        if save_model:
            torch.save(self.encoder.state_dict(), dirname + '/enc.th')
            torch.save(self.decoder.state_dict(), dirname + '/dec.th')

        return dirname

    def load_values(self, dirname):
        self.encoder.load_state_dict(
            torch.load(dirname + '/enc.th', map_location={'cuda:1': 'cuda:0'}))
        self.decoder.load_state_dict(
            torch.load(dirname + '/dec.th', map_location={'cuda:1': 'cuda:0'}))


# In[78]:


# a = [[1,2,3,5],[4,5,6]]
# [x for y in a for x in y]


# In[35]:


from attention.common_code.metrics import calc_metrics_classification, print_metrics
# import attention.model.Binary_Classification as BC
import codecs, json
from tqdm import tqdm
import numpy as np
from attention.preprocess import vectorizer
# import wandb

class Trainer() :
    def __init__(self, dataset, args, config) :
        Model = BC_Model
        self.model = Model(config, args, pre_embed=dataset.vec.embeddings)
        self.metrics = calc_metrics_classification
        self.display_metrics = True
        self.PGDer = None

    def train_standard(self, train_data, test_data, args, save_on_metric='roc_auc') :

        best_metric = 0.0
        for i in tqdm(range(args.n_iters)) :

            _, loss_tr, loss_tr_orig, _, _ = self.model.train(train_data.X, train_data.y)
            predictions_tr, attentions_tr, _ = self.model.evaluate(train_data.X)
            predictions_tr = np.array(predictions_tr)
            train_metrics = self.metrics(np.array(train_data.y), predictions_tr)
            print_str = "FULL (WEIGHTED) LOSS: %f | ORIG (UNWEIGHTED) LOSS: %f" % (loss_tr, loss_tr_orig)
            print(print_str)

            print("TRAIN METRICS:")
            if self.display_metrics:
                print_metrics(train_metrics, adv=False)

            predictions_te, attentions_te, _ = self.model.evaluate(test_data.X)
            predictions_te = np.array(predictions_te)
            test_metrics = self.metrics(np.array(test_data.y), predictions_te)

            print("TEST METRICS:")
            if self.display_metrics:
                print_metrics(test_metrics, adv=False)

            metric = test_metrics[save_on_metric]
            if metric > best_metric :
                best_metric = metric
                save_model = True
                print("Model Saved on ", save_on_metric, metric)
            else :
                save_model = False
                print("Model not saved on ", save_on_metric, metric)

            dirname = self.model.save_values(save_model=save_model)
            if save_model:
                attentions_tr = [el.tolist() for el in attentions_tr]
                attentions_te = [el.tolist() for el in attentions_te]
                print("SAVING PREDICTIONS AND ATTENTIONS")
                json.dump(predictions_tr.tolist(), codecs.open(dirname + '/train_predictions_best_epoch.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(predictions_te.tolist(), codecs.open(dirname + '/test_predictions_best_epoch.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(attentions_tr, codecs.open(dirname + '/train_attentions_best_epoch.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(attentions_te, codecs.open(dirname + '/test_attentions_best_epoch.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

            print("DIRECTORY:", dirname)

            f = open(dirname + '/epoch.txt', 'a')
            f.write(str(test_metrics) + '\n')
            f.close()

    def train_ours(self, train_data, test_data, args):
        br = False
        n_fail = 0
        best_loss = 10000000000
        for i in tqdm(range(args.n_iters)):

            loss_tr, loss_tr_orig, tvd_loss_tr, topk_loss_tr, pgd_tvd_loss_tr,true_topk_loss_tr = self.model.train_ours(train_data.X, train_data.y,
                                                                                 train_data.true_pred,
                                                                                 train_data.gold_attns,PGDer=self.PGDer)
            # wandb.log({
            #     "loss_tr":loss_tr,
            #     "loss_tr_orig":loss_tr_orig,
            #     "tvd_loss_tr":tvd_loss_tr,
            #     "topk_loss_tr":topk_loss_tr,
            #     "pgd_tvd_loss_tr":pgd_tvd_loss_tr,
            #     "true_topk_loss_tr":true_topk_loss_tr
            # })

            loss_te, loss_te_orig, tvd_loss_te, topk_loss_te, pgd_tvd_loss_te, true_topk_loss_te = self.model.train_ours(test_data.X,
                                                                                                test_data.y,
                                                                                                test_data.true_pred,
                                                                                                test_data.gold_attns,
                                                                                                PGDer=self.PGDer, train=False)
            # wandb.log({
            #     "loss_te": loss_te,
            #     "loss_te_orig": loss_te_orig,
            #     "tvd_loss_te": tvd_loss_te,
            #     "topk_loss_te": topk_loss_te,
            #     "pgd_tvd_loss_te": pgd_tvd_loss_te,
            #     "true_topk_loss_te":true_topk_loss_te
            # })

            predictions_tr, attentions_tr, jsd_score_tr = self.model.evaluate(train_data.X,
                                                                              target_attn=train_data.gold_attns)

            # wandb.log({
            #     "predictions_tr": predictions_tr,
            #     "attentions_tr": attentions_tr,
            #     "jsd_score_tr": jsd_score_tr,
            # })

            predictions_tr = np.array(predictions_tr)
            train_metrics = self.metrics(np.array(train_data.y), predictions_tr, np.array(train_data.true_pred),
                                         jsd_score_tr)
            print_str = "FULL (WEIGHTED) LOSS: %f | ORIG (UNWEIGHTED) LOSS: %f | TOPK-LOSS: %f | TVD-OUT: %f | TVD-PGD: %f" % (
            loss_tr, loss_tr_orig, topk_loss_tr, tvd_loss_tr, pgd_tvd_loss_tr)
            # print(print_str)
            #
            # print("TRAIN METRICS:")
            # if self.display_metrics:
            #     print_metrics(train_metrics, adv=True)
            #
            predictions_te, attentions_te, jsd_score_te = self.model.evaluate(test_data.X,
                                                                              target_attn=test_data.gold_attns)
            # wandb.log({
            #     "predictions_te": predictions_te,
            #     "attentions_te": attentions_te,
            #     "jsd_score_te": jsd_score_te,
            # })

            predictions_te = np.array(predictions_te)
            test_metrics = self.metrics(np.array(test_data.y), predictions_te, np.array(test_data.true_pred),
                                        jsd_score_te)

            # print("TEST METRICS:")
            # if self.display_metrics:
            #     print_metrics(test_metrics, adv=True)

            if loss_tr < best_loss:
                best_loss = loss_tr
                n_fail = 0
                save_model = True
                # print("Model Saved on Training Loss: ", loss_tr)
                # wandb.log({
                #     "best_loss": best_loss,
                # })

            else:
                n_fail += 1
                save_model = False
                # print("Model not saved on Training Loss: ", loss_tr)
                if n_fail >= 10:
                    br = True
                    # print("Loss hasn't decreased for 10 epochs...EARLY STOPPING TRIGGERED")

            dirname = self.model.save_values(save_model=save_model)
            if save_model:
                attentions_tr = [el.tolist() for el in attentions_tr]
                attentions_te = [el.tolist() for el in attentions_te]
                # print("SAVING PREDICTIONS AND ATTENTIONS")
                json.dump(predictions_tr.tolist(),
                          codecs.open(dirname + '/train_predictions_best_epoch.json', 'w', encoding='utf-8'),
                          separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(predictions_te.tolist(),
                          codecs.open(dirname + '/test_predictions_best_epoch.json', 'w', encoding='utf-8'),
                          separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(attentions_tr,
                          codecs.open(dirname + '/train_attentions_best_epoch.json', 'w', encoding='utf-8'),
                          separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(attentions_te,
                          codecs.open(dirname + '/test_attentions_best_epoch.json', 'w', encoding='utf-8'),
                          separators=(',', ':'), sort_keys=True, indent=4)

            print("DIRECTORY:", dirname)

            f = open(dirname + '/epoch.txt', 'a')
            f.write(str(test_metrics) + '\n')
            f.close()

            f = open(dirname + '/losses.txt', 'a')
            f.write("EPOCH %d: " % i + print_str + '\n')
            f.close()

            if br:
                break

    def train_adversarial(self, train_data, test_data, args) :

        br = False
        n_fail = 0
        best_loss = 10000000000
        for i in tqdm(range(args.n_iters)) :

            _, loss_tr, loss_tr_orig, tvd_loss_tr, kl_loss_tr = self.model.train(train_data.X, train_data.y, train_data.true_pred, train_data.gold_attns)
            predictions_tr, attentions_tr, jsd_score_tr = self.model.evaluate(train_data.X, target_attn=train_data.gold_attns)
            predictions_tr = np.array(predictions_tr)
            train_metrics = self.metrics(np.array(train_data.y), predictions_tr, np.array(train_data.true_pred), jsd_score_tr)
            print_str = "FULL (WEIGHTED) LOSS: %f | ORIG (UNWEIGHTED) LOSS: %f | KL: %f | TVD: %f" % (loss_tr, loss_tr_orig, kl_loss_tr, tvd_loss_tr)
            print(print_str)

            print("TRAIN METRICS:")
            if self.display_metrics:
                print_metrics(train_metrics, adv=True)

            predictions_te, attentions_te, jsd_score_te = self.model.evaluate(test_data.X, target_attn=test_data.gold_attns)
            predictions_te = np.array(predictions_te)
            test_metrics = self.metrics(np.array(test_data.y), predictions_te, np.array(test_data.true_pred), jsd_score_te)

            print("TEST METRICS:")
            if self.display_metrics:
                print_metrics(test_metrics, adv=True)

            if loss_tr < best_loss:
                best_loss = loss_tr
                n_fail = 0
                save_model = True
                print("Model Saved on Training Loss: ", loss_tr)

            else :
                n_fail += 1
                save_model = False
                print("Model not saved on Training Loss: ", loss_tr)
                if n_fail >= 10:
                    br = True
                    print("Loss hasn't decreased for 10 epochs...EARLY STOPPING TRIGGERED")

            dirname = self.model.save_values(save_model=save_model)
            if save_model:
                attentions_tr = [el.tolist() for el in attentions_tr]
                attentions_te = [el.tolist() for el in attentions_te]
                print("SAVING PREDICTIONS AND ATTENTIONS")
                json.dump(predictions_tr.tolist(), codecs.open(dirname + '/train_predictions_best_epoch.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(predictions_te.tolist(), codecs.open(dirname + '/test_predictions_best_epoch.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(attentions_tr, codecs.open(dirname + '/train_attentions_best_epoch.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
                json.dump(attentions_te, codecs.open(dirname + '/test_attentions_best_epoch.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

            print("DIRECTORY:", dirname)

            f = open(dirname + '/epoch.txt', 'a')
            f.write(str(test_metrics) + '\n')
            f.close()

            f = open(dirname + '/losses.txt', 'a')
            f.write("EPOCH %d: " % i + print_str + '\n')
            f.close()

            if br:
                break

class Evaluator() :
    def __init__(self, dataset, dirname, args) :
        Model = BC_Model
        self.model = Model.init_from_config(dirname, args)
        self.model.dirname = dirname
        self.metrics = calc_metrics_classification
        self.display_metrics = True

    def evaluate(self, test_data, save_results=False) :
        if self.model.adversarial :
            predictions, attentions, jsd_score = self.model.evaluate(test_data.X, target_attn=test_data.gold_attns)
            predictions = np.array(predictions)
            test_metrics = self.metrics(np.array(test_data.y), predictions, np.array(test_data.true_pred), jsd_score)
        else :
            predictions, attentions, _ = self.model.evaluate(test_data.X)
            predictions = np.array(predictions)
            test_metrics = self.metrics(test_data.y, predictions)

        # if self.display_metrics :
        #     print_metrics(test_metrics, adv=self.model.adversarial)

        if save_results :
            f = open(self.model.dirname + '/evaluate.json', 'w')
            json.dump(test_metrics, f)
            f.close()

        test_data.yt_hat = predictions
        test_data.attn_hat = attentions
        return predictions, attentions


# ## 训练流程

# In[37]:


# # from attention.configurations import generate_config
# # from attention.Trainers.TrainerBC import Trainer, Evaluator
#
# def train_dataset(dataset, args, config='lstm') :
#         config = generate_config(dataset, args, config)
#         trainer = Trainer(dataset, args, config=config)
#         #go ahead and save model
#         dirname = trainer.model.save_values(save_model=False)
#         print("DIRECTORY:", dirname)
#         if args.adversarial :
#             trainer.train_adversarial(dataset.train_data, dataset.test_data, args)
#         # elif args.ours:
#         #     from attention.attack import  PGDAttacker
#         #     PGDer = PGDAttacker(
#         #         radius=args.pgd_radius, steps=args.pgd_step, step_size=args.pgd_step_size, random_start= \
#         #         True, norm_type=args.pgd_norm_type, ascending=True
#         #     )
#         #     trainer.PGDer = PGDer
#         #     trainer.train_ours(dataset.train_data, dataset.test_data, args)
#         else:
#             trainer.train_standard(dataset.train_data, dataset.test_data, args, save_on_metric=dataset.save_on_metric)
#         print('####################################')
#         print("TEST RESULTS FROM BEST MODEL")
#         evaluator = Evaluator(dataset, trainer.model.dirname, args)
#         _ = evaluator.evaluate(dataset.test_data, save_results=True)
#         return trainer, evaluator


start = time.time()
# train_dataset_on_encoders(dataset, args, exp_name)
# train_dataset(dataset, args, exp_name)
# args, exp_name
# exp_name =
config = exp_name
config = generate_config(dataset, args, config)
trainer = Trainer(dataset, args, config=config)
#go ahead and save model
dirname = trainer.model.save_values(save_model=False)
print("DIRECTORY:", dirname)
if args.adversarial :
    trainer.train_adversarial(dataset.train_data, dataset.test_data, args)
# elif args.ours:
#     from attention.attack import  PGDAttacker
#     PGDer = PGDAttacker(
#         radius=args.pgd_radius, steps=args.pgd_step, step_size=args.pgd_step_size, random_start= \
#         True, norm_type=args.pgd_norm_type, ascending=True
#     )
#     trainer.PGDer = PGDer
#     trainer.train_ours(dataset.train_data, dataset.test_data, args)
else:
    trainer.train_standard(dataset.train_data, dataset.test_data, args, save_on_metric=dataset.save_on_metric)
print('####################################')
print("TEST RESULTS FROM BEST MODEL")
evaluator = Evaluator(dataset, trainer.model.dirname, args)
_ = evaluator.evaluate(dataset.test_data, save_results=True)


print("TOTAL ELAPSED TIME: %f HOURS OR %f MINUTES" % (((time.time() - start)/60/60), ((time.time() - start)/60)))
# _python_exit()
# sys.exit()
# wandb.log({
#     "finish":'True'
# })
# import os
# import signal
# os.kill(os.getpid(), signal.SIGKILL)



# ## 结果分析

# In[ ]:




