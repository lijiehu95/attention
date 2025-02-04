import json
import os
import shutil
from copy import deepcopy

import torch
import torch.nn as nn
import wandb
from allennlp.common import Params
from sklearn.utils import shuffle
from tqdm import tqdm
import time

from attention.model.modules.Decoder import AttnDecoder, FrozenAttnDecoder, PretrainedWeightsDecoder
from attention.model.modules.Encoder import Encoder
from attention.common_code.metrics import batch_tvd

from .modelUtils import BatchHolder, get_sorting_index_with_noise_from_lengths
from .modelUtils import jsd as js_divergence

file_name = os.path.abspath(__file__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model():
    def __init__(self, configuration, args, pre_embed=None):
        configuration = deepcopy(configuration)
        self.configuration = deepcopy(configuration)

        configuration['model']['encoder']['pre_embed'] = pre_embed
        self.encoder = Encoder.from_params(
            Params(configuration['model']['encoder'])).to(device)

        self.frozen_attn = args.frozen_attn
        self.adversarial = args.adversarial
        self.ours = args.ours
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

    # def related_score(self,
    #           data_in,
    #           target_in,
    #           target_pred,
    #           target_attn_in,train=False,preturb_x=True,X_PGDer=None):
    #     sorting_idx = get_sorting_index_with_noise_from_lengths(
    #         [len(x) for x in data_in], noise_frac=0.1)
    #     data = [data_in[i] for i in sorting_idx]
    #     target = [target_in[i] for i in sorting_idx]
    #
    #     self.encoder.train()
    #     self.decoder.train()
    #
    #
    #     target_pred = [target_pred[i] for i in sorting_idx]
    #     target_attn = [target_attn_in[i] for i in sorting_idx]
    #
    #     N = len(data)
    #     bsize = self.bsize
    #     batches = list(range(0, N, bsize))
    #     batches = shuffle(batches)
    #
    #     spearman_score=0
    #     spearman_score_trained_att=0
    #     num=0
    #     kendalltau_score=0
    #     kendalltau_score_trained_att=0
    #
    #     print("related score comp")
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
    #         if preturb_x:
    #             # get the hidden
    #             def target_model(embedd, data, decoder, encoder):
    #                 encoder(data, revise_embedding=embedd)
    #                 decoder(data=data)
    #                 return torch.sigmoid(data.predict)
    #
    #             def crit(gt, pred):
    #                 return batch_tvd(gt,pred)
    #
    #             # old prediction
    #             self.encoder(batch_data)
    #             self.decoder(batch_data)
    #
    #             # PGD generate the new hidden
    #             new_embedd = X_PGDer.perturb(criterion=crit, x=batch_data.embedding, data=batch_data \
    #                                          , decoder=self.decoder, encoder=self.encoder,
    #                                          batch_target_pred=batch_target_pred,
    #                                          target_model=target_model)
    #             # pgd perturb the hidden
    #             self.encoder(batch_data, revise_embedding=new_embedd)
    #             self.decoder(data=batch_data)
    #             new_att = batch_data.attn
    #         else:
    #             assert False
    #
    #         # this is the unpreturbed embedding
    #         batch_data.keep_grads = True
    #         self.encoder(batch_data)
    #         self.decoder(batch_data)
    #         batch_data.predict.sum().backward()
    #
    #         grad = batch_data.embedding.grad
    #         grad = grad.mean(dim=-1)
    #         from torchmetrics import SpearmanCorrCoef
    #         from scipy.stats import kendalltau
    #         spearman = SpearmanCorrCoef()
    #         for i in range(grad.shape[0]):
    #             spearman.reset()
    #             target = batch_data.target_attn[i]
    #             # print(target.shape)
    #             pred = grad[i]
    #             # print(pred.shape)
    #             num+=bsize
    #             spearman_score += spearman(pred, new_att[i]).item()
    #             spearman_score_trained_att += spearman(pred,target).item()
    #             kendalltau_score += kendalltau(pred.detach().cpu().numpy(), new_att[i].detach().cpu().numpy())[0]
    #             kendalltau_score_trained_att += kendalltau(pred.detach().cpu().numpy(), target.detach().cpu().numpy())[0]
    #
    #     kendalltau_score /= num
    #     kendalltau_score_trained_att /=num
    #     spearman_score /= num
    #     spearman_score_trained_att /= num
    #
    #     wandb.log({
    #         "spearman_score":spearman_score,
    #         "spearman_score_trained_att":spearman_score_trained_att,
    #         "kendalltau_score":kendalltau_score,
    #         "kendalltau_score_trained_att":kendalltau_score_trained_att,
    #     })
    #

    def preterub_x_eval(self,
              data_in,
              target_in,
              target_pred,
              target_attn_in,X_PGDer=None):
        sorting_idx = get_sorting_index_with_noise_from_lengths(
            [len(x) for x in data_in], noise_frac=0.1)
        data = [data_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]

        target_pred = [target_pred[i] for i in sorting_idx]
        target_attn = [target_attn_in[i] for i in sorting_idx]

        self.encoder.train()
        self.decoder.train()

        from attention.utlis import AverageMeter

        N = len(data)
        bsize = self.bsize
        batches = list(range(0, N, bsize))
        batches = shuffle(batches)

        # px_jsd_att_diff_original = []
        # px_tvd_pred_diff = []
        from attention.utlis import AverageMeter
        px_l1_att_diff = AverageMeter()
        px_l2_att_diff = AverageMeter()
        px_tvd_pred_diff = AverageMeter()
        px_jsd_att_diff = AverageMeter()


        for n in tqdm(batches):
            batch_doc = data[n:n + bsize]

            batch_target_attn = target_attn[n:n + bsize]
            batch_data = BatchHolder(batch_doc, batch_target_attn)

            batch_target_pred = target_pred[n:n + bsize]
            batch_target_pred = torch.Tensor(batch_target_pred).to(device)

            if len(batch_target_pred.shape) == 1:  # (B, )
                batch_target_pred = batch_target_pred.unsqueeze(
                    -1)  # (B, 1)

            # get the hidden
            def target_model(embedd, data, decoder, encoder):
                encoder(data, revise_embedding=embedd)
                decoder(data=data)
                return torch.sigmoid(data.predict)

            def crit(gt, pred):
                return batch_tvd(gt, pred)

            # old prediction
            self.encoder(batch_data)
            self.decoder(batch_data)

            # old att pred
            old_pred = torch.sigmoid(batch_data.predict)
            old_att = batch_data.attn

            # PGD generate the new hidden
            new_embedd = X_PGDer.perturb(criterion=crit, x=batch_data.embedding, data=batch_data \
                                         , decoder=self.decoder, encoder=self.encoder,
                                         batch_target_pred=batch_target_pred,
                                         target_model=target_model)

            # pgd perturb the hidden
            self.encoder(batch_data, revise_embedding=new_embedd)
            self.decoder(data=batch_data)

            # diff of att
            new_att = torch.sigmoid(batch_data.attn)
            px_l1_att_diff.update(torch.abs(old_att - new_att).mean().item(), len(batch_doc))
            px_l2_att_diff.update(torch.pow(old_att - new_att, 2).mean().item(), len(batch_doc))
            # jsd between att
            # px_jsd_att_diff_original.append(js_divergence(
            #     old_att, new_att).squeeze(
            #     1).cpu().data.numpy().mean())
            # px_l1_att_diff = (old_att - new_att).abs().mean().item()
            # px_l2_att_diff = (old_att - new_att).pow(2).mean().item()
            # px_jsd_att_diff =
            new_pred = torch.sigmoid(batch_data.predict)
            px_tvd_pred_diff.update(batch_tvd(old_pred, new_pred).item(),len(batch_doc))
            px_jsd_att_diff.update(js_divergence(torch.softmax(old_att,dim=-1), torch.softmax(new_att,dim=-1)).squeeze(1).cpu().data.numpy().mean(), len(batch_doc))

            # px_tvd_pred_diff.append(batch_tvd(new_pred, old_pred).item())

        import numpy as np
        res = {
            "px_l1_att_diff":px_l1_att_diff.average(),
            "px_l2_att_diff":px_l2_att_diff.average(),
            "px_tvd_pred_diff":px_tvd_pred_diff.average(),
            "px_jsd_att_diff":px_jsd_att_diff.average(),
        }
        # if baseline:
        #     for k,v in res.items():
        #         # rename the key of dict res
        #         res[k+"_baseline"] = v
        #         # remove the original key
        #         del res[k]
        # if trian:
        #     for k,v in res.items():
        #         # rename the key of dict res
        #         res[k+"_tr"] = v
        #         # remove the original key
        #         del res[k]
        # else:
        #     for k,v in res.items():
        #         # rename the key of dict res
        #         res[k+"_te"] = v
        #         # remove the original key
        #         del res[k]
        return res



    def train_ours(self,
              data_in,
              target_in,
              target_pred,
              target_attn_in,
              PGDer,train=True,preturb_x=False,X_PGDer=None):
        # sorting_idx = get_sorting_index_with_noise_from_lengths(
        #     [len(x) for x in data_in], noise_frac=0.1)
        # data = [data_in[i] for i in sorting_idx]
        # target = [target_in[i] for i in sorting_idx]
        data= data_in
        target = target_in

        # print(target_pred)

        target_pred = [target_pred[i] for i in sorting_idx]
        target_attn = [target_attn_in[i] for i in sorting_idx]

        self.encoder.train()
        self.decoder.train()

        from attention.utlis import AverageMeter
        loss_weighted = 0
        loss_unweighted = 0
        tvd_loss_total = 0
        topk_loss_total = 0
        pgd_tvd_loss_total = 0
        true_topk_loss = AverageMeter()

        N = len(data)
        bsize = self.bsize
        batches = list(range(0, N, bsize))
        batches = shuffle(batches)


        from attention.utlis import AverageMeter
        px_l1_att_diff = AverageMeter()
        px_l2_att_diff = AverageMeter()
        px_tvd_pred_diff = AverageMeter()
        px_jsd_att_diff = AverageMeter()


        for n in tqdm(batches):
            batch_doc = data[n:n + bsize]

            batch_target_attn = target_attn[n:n + bsize]
            batch_data = BatchHolder(batch_doc, batch_target_attn)

            batch_target_pred = target_pred[n:n + bsize]
            batch_target_pred = torch.Tensor(batch_target_pred).to(device)

            if len(batch_target_pred.shape) == 1:  # (B, )
                batch_target_pred = batch_target_pred.unsqueeze(
                    -1)  # (B, 1)

            if preturb_x:
                # get the hidden
                def target_model(embedd, data, decoder,encoder):
                    encoder(data,revise_embedding=embedd)
                    decoder(data=data)
                    return torch.sigmoid(data.predict)

                def crit(gt, pred):
                    return batch_tvd(gt,pred)

                # old prediction
                self.encoder(batch_data)
                self.decoder(batch_data)

                # old att pred
                old_pred = torch.sigmoid(batch_data.predict)
                old_att = batch_data.attn

                # PGD generate the new hidden
                new_embedd = X_PGDer.perturb(criterion=crit, x=batch_data.embedding, data=batch_data \
                                        , decoder=self.decoder,encoder=self.encoder, batch_target_pred=batch_target_pred,
                                        target_model=target_model)

                # pgd perturb the hidden
                self.encoder(batch_data,revise_embedding=new_embedd)
                self.decoder(data=batch_data)

                # diff of att
                new_att = torch.sigmoid(batch_data.attn)
                px_l1_att_diff.update(torch.abs(old_att - new_att).mean().item(), len(batch_doc))
                px_l2_att_diff.update(torch.pow(old_att - new_att, 2).mean().item(), len(batch_doc))
                # jsd between att
                # px_jsd_att_diff_original.append(js_divergence(
                #     old_att, new_att).squeeze(
                #     1).cpu().data.numpy().mean())
                # px_l1_att_diff = (old_att - new_att).abs().mean().item()
                # px_l2_att_diff = (old_att - new_att).pow(2).mean().item()
                # px_jsd_att_diff =
                new_pred = torch.sigmoid(batch_data.predict)
                px_tvd_pred_diff.update(batch_tvd(old_pred, new_pred).item(), len(batch_doc))
                px_jsd_att_diff.update(
                    js_divergence(torch.softmax(old_att, dim=-1), torch.softmax(new_att, dim=-1)).squeeze(
                        1).cpu().data.numpy().mean(), len(batch_doc))

                # px_tvd_pred_diff.append(batch_tvd(new_pred, old_pred).item())

                import numpy as np
                # res = {
                #     "px_l1_att_diff": px_l1_att_diff.average(),
                #     "px_l2_att_diff": px_l2_att_diff.average(),
                #     "px_tvd_pred_diff": px_tvd_pred_diff.average(),
                #     "px_jsd_att_diff": px_jsd_att_diff.average(),
                # }

            # else:

            # res = self.preterub_x_eval(
            #   data_in,
            #   target_in,
            #   target_pred,
            #   target_attn_in,X_PGDer=X_PGDer)
            #
            # # if train:
            #     px_l1_att_diff, px_l2_att_diff, px_tvd_pred_diff = res["px_l1_att_diff_tr"], res["px_l2_att_diff_tr"], res["px_tvd_pred_diff_tr"]
            # else:
            #     px_l1_att_diff, px_l2_att_diff, px_tvd_pred_diff = res["px_l1_att_diff_te"], res["px_l2_att_diff_te"], res[
            #     "px_tvd_pred_diff_te"]
            #     px_l1_att_diff, px_l2_att_diff, px_tvd_pred_diff,px_jsd_att_diff = res["px_l1_att_diff"], res["px_l2_att_diff"], res["px_tvd_pred_diff"],res["px_jsd_att_diff"]

            # to the true att and embedding
            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_target = target[n:n + bsize]
            batch_target = torch.Tensor(batch_target).to(device)

            if len(batch_target.shape) == 1:  #(B, )
                batch_target = batch_target.unsqueeze(-1)  #(B, 1)

            # calculate adversarial loss (Section 4) if adversarial model

            from attention.utlis import topk_overlap_loss,topK_overlap_true_loss

            topk_loss = topk_overlap_loss(batch_data.target_attn,
                                          batch_data.attn,K=self.K, metric=self.topk_prox_metric)
            topk_true_loss = topK_overlap_true_loss(batch_data.target_attn,
                                          batch_data.attn,K=self.K)

            true_topk_loss.update(
                topk_true_loss,len(batch_doc)
            )

            tvd_loss = batch_tvd(
                torch.sigmoid(batch_data.predict), batch_target_pred)

            ### pgd loss
            def target_model(w, data, decoder, encoder):
                decoder(revise_att=w, data=data)
                return data.predict

            def crit(gt, pred):
                return batch_tvd(torch.sigmoid(pred), gt)

            # PGD generate the new weight
            new_att = PGDer.perturb(criterion=crit, x=batch_data.attn, data=batch_data \
                                    , decoder=self.decoder, batch_target_pred=batch_target_pred,
                                    target_model=target_model)

            # output the prediction tvd of new weight and old weight
            self.decoder(batch_data, revise_att=new_att)

            new_out = batch_data.predict
            att_pgd_pred_tvd = batch_tvd(
                torch.sigmoid(new_out), batch_target_pred)

            loss_orig = tvd_loss + self.lambda_1 * att_pgd_pred_tvd + self.lambda_2 * topk_loss


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

            loss_weighted += float(loss.data.cpu().item())
            loss_unweighted += float(loss_orig.data.cpu().item())
            tvd_loss_total += float(tvd_loss.data.cpu().item())
            topk_loss_total += float(topk_loss.data.cpu().item())
            pgd_tvd_loss_total += float(
                att_pgd_pred_tvd.data.cpu().item())


        import numpy as np
        # assert len(px_jsd_att_diff) !=0 and len(px_tvd_pred_diff)!=0
        # px_tvd_pred_diff = np.mean(px_tvd_pred_diff)
        # px_jsd_att_diff = np.mean(px_jsd_att_diff)

        true_topk_loss = true_topk_loss.average()

        # return  loss_total, loss_orig_total, tvd_loss_total, topk_loss_total, pgd_tvd_loss_total, true_topk_loss, px_tvd_pred_diff ,px_l1_att_diff ,px_l2_att_diff
        res = {
            "loss_weighted": loss_weighted,
            "loss_unweighted": loss_unweighted,
            "tvd_loss": tvd_loss_total,
            "topk_loss": topk_loss_total,
            "pgd_tvd_loss": pgd_tvd_loss_total,
            "true_topk_loss": true_topk_loss,}
        if preturb_x:
            res["px_tvd_pred_diff"] = px_tvd_pred_diff.average()
            res["px_jsd_att_diff"] = px_jsd_att_diff.average()
            res["px_l1_att_diff"] = px_l1_att_diff.average()
            res["px_l2_att_diff"] = px_l2_att_diff.average()
        return res
        # return {
        #     "loss_weighted": loss_weighted,
        #     "loss_unweighted": loss_unweighted,
        #     "tvd_loss": tvd_loss_total,
        #     "topk_loss": topk_loss_total,
        #     "pgd_tvd_loss": pgd_tvd_loss_total,
        #     "true_topk_loss": true_topk_loss,
        #     "px_l1_att_diff": px_l1_att_diff.average(),
        #     "px_l2_att_diff": px_l2_att_diff.average(),
        #     "px_tvd_pred_diff": px_tvd_pred_diff.average(),
        #     "px_jsd_att_diff": px_jsd_att_diff.average(),
        # }

    def train(self,
              data_in,
              target_in,
              target_pred=None,
              target_attn_in=None,
              train=True,
              ours=False,
              PDGer=None):
        # sorting_idx = get_sorting_index_with_noise_from_lengths(
        #     [len(x) for x in data_in], noise_frac=0.1)
        # data = [data_in[i] for i in sorting_idx]
        # target = [target_in[i] for i in sorting_idx]

        data= data_in
        target = target_in

        if target_pred:
            # target_pred = [target_pred[i] for i in sorting_idx]
            # target_attn = [target_attn_in[i] for i in sorting_idx]
            target_attn = target_attn_in

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)
        loss_total = 0
        loss_unweighted = 0
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
                if ours:
                    from attention.utlis import topk_overlap_loss
                    topk_loss = topk_overlap_loss(batch_data.target_attn.log(),
                                                  batch_data.attn)
                    tvd_loss = batch_tvd(
                        torch.sigmoid(batch_data.predict), batch_target_pred)

                    ### pgd loss
                    def target_model(w, data, decoder):
                        decoder(revise_att=w, data=data)
                        return data.predict

                    def crit(gt, pred):
                        return batch_tvd(torch.sigmoid(pred), gt)

                    # PGD generate the new weight
                    new_att = PDGer.perturb(criterion=crit, att=batch_data.attn, data=batch_data \
                                          , decoder=self.decoder,batch_target_pred=batch_target_pred, target_model=target_model)

                    # output the prediction tvd of new weight and old weight
                    self.decoder(batch_data, revise_att=new_att)
                    new_out = batch_data.predict
                    att_pgd_pred_tvd = batch_tvd(
                        torch.sigmoid(new_out), batch_target_pred)

                    loss_orig = tvd_loss + self.lambda_1 * att_pgd_pred_tvd + self.lambda_2 * topk_loss

                else:
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
                if ours:
                    loss_unweighted += float(loss_orig.data.cpu().item())
                    tvd_loss_total += float(tvd_loss.data.cpu().item())
                    # kl_loss_total += float(kl_loss.data.cpu().item())
                    topk_loss_total += float(topk_loss.data.cpu().item())
                    pgd_tvd_loss_total += float(
                        att_pgd_pred_tvd.data.cpu().item())
                else:

                    loss_unweighted += float(loss_orig.data.cpu().item())
                    tvd_loss_total += float(tvd_loss.data.cpu().item())
                    kl_loss_total += float(kl_loss.data.cpu().item())
        if ours:
            return loss_total * bsize / N, loss_total, loss_unweighted, tvd_loss_total, topk_loss_total, pgd_tvd_loss_total
        else:
            return loss_total * bsize / N, loss_total, loss_unweighted, tvd_loss_total, kl_loss_total

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
        shutil.copy2(file_name, dirname + '/')
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
