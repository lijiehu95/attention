import json
import os
import shutil
from copy import deepcopy
from decoder import *
from encoder import Encoder
import torch
import torch.nn as nn
from allennlp.common import Params
from sklearn.utils import shuffle
# from tqdm import tqdm
import time
from utlis import *
from tqdm import tqdm


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
        self.mode = args.mode
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
        if self.mode == "adv":
            self.criterion = nn.KLDivLoss(
                size_average=None, reduce=None, reduction='sum').to(device)
            self.lmbda = args.lmbda
        elif self.mode == 'std':
            self.criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)
        else:
            assert NotImplementedError

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
    #         new_att = PGDer.perturb(criterion=crit, att=batch_data.attn, data=batch_data \
    #                                 , decoder=self.decoder, batch_target_pred=batch_target_pred,
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

                loss_orig_total += float(loss_orig.data.cpu().item())
                tvd_loss_total += float(tvd_loss.data.cpu().item())
                kl_loss_total += float(kl_loss.data.cpu().item())

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
                batch_jsdscores = jsd(
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