from attention.common_code.metrics import calc_metrics_classification, print_metrics
import attention.model.Binary_Classification as BC
import codecs, json
from tqdm import tqdm
import numpy as np
from attention.preprocess import vectorizer
import wandb

class Trainer() :
    def __init__(self, dataset, args, config) :
        Model = BC.Model
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

    def train_ours(self, train_data, test_data, args,dataset=None):
        br = False
        n_fail = 0
        best_loss = 10000000000

        # for eval the original model
        evaluator = Evaluator(dataset, args.gold_label_dir, args)
        original_metric,_,_ = evaluator.evaluate(dataset.test_data, save_results=True)

        wandb.log({
            "original_metric": original_metric,
        })

        # log original performance of defense x preturb
        original_px_att_diff, original_px_tvd_pred_diff = evaluator.model.preterub_x_testing(
            test_data.X,
            test_data.y,
            test_data.true_pred,
            test_data.gold_attns,X_PGDer=self.X_PGDer)

        wandb.log({
            "original_px_att_diff": original_px_att_diff,
            "original_px_tvd_pred_diff": original_px_tvd_pred_diff
        })

        # predictions_te, attentions_te, jsd_score_te = self.model.evaluate(test_data.X,
        #                                                                   target_attn=test_data.gold_attns)
        # wandb.log({
        #     "original_predictions_te": predictions_te,
        #     "original_attentions_te": attentions_te,
        #     "original_jsd_score_te": jsd_score_te,
        # })

        # predictions_te = np.array(predictions_te)
        # test_metrics = self.metrics(np.array(test_data.y), predictions_te, np.array(test_data.true_pred),
        #                             jsd_score_te)
        #
        # wandb.log({
        #     "original_test_metrics":test_metrics
        # })

        for i in tqdm(range(args.n_iters)):

            loss_tr, loss_tr_orig, tvd_loss_tr, topk_loss_tr, pgd_tvd_loss_tr,true_topk_loss_tr = self.model.train_ours(train_data.X, train_data.y,
                                                                                 train_data.true_pred,
                                                                                 train_data.gold_attns,PGDer=self.PGDer)
            wandb.log({
                "loss_tr":loss_tr,
                "loss_tr_orig":loss_tr_orig,
                "tvd_loss_tr":tvd_loss_tr,
                "topk_loss_tr":topk_loss_tr,
                "pgd_tvd_loss_tr":pgd_tvd_loss_tr,
                "true_topk_loss_tr":true_topk_loss_tr
            })

            loss_te, loss_te_orig, tvd_loss_te, topk_loss_te, pgd_tvd_loss_te, true_topk_loss_te = self.model.train_ours(test_data.X,
                                                                                                test_data.y,
                                                                                                test_data.true_pred,
                                                                                                test_data.gold_attns,
                                                                                                PGDer=self.PGDer, train=False,preturb_x=True,X_PGDer=self.X_PGDer)

            # loss_te_px, loss_te_orig_px, tvd_loss_te_px, topk_loss_te_px, pgd_tvd_loss_te_px, true_topk_loss_te_px = self.model.train_ours(
            #     test_data.X,
            #     test_data.y,
            #     test_data.true_pred,
            #     test_data.gold_attns,
            #     PGDer=self.PGDer, train=False,preturb_x=True,X_PGDer=self.X_PGDer)


            wandb.log({
                "loss_te": loss_te,
                "loss_te_orig": loss_te_orig,
                "tvd_loss_te": tvd_loss_te,
                "topk_loss_te": topk_loss_te,
                "pgd_tvd_loss_te": pgd_tvd_loss_te,
                "true_topk_loss_te":true_topk_loss_te,


                # "loss_te_px": loss_te_px,
                # "loss_te_orig_px": loss_te_orig_px,
                # "tvd_loss_te_px": tvd_loss_te_px,
                # "topk_loss_te_px": topk_loss_te_px,
                # "pgd_tvd_loss_te_px": pgd_tvd_loss_te_px,
                # "true_topk_loss_te_px":true_topk_loss_te_px,
            })

            predictions_tr, attentions_tr, jsd_score_tr = self.model.evaluate(train_data.X,
                                                                              target_attn=train_data.gold_attns)

            wandb.log({
                "predictions_tr": predictions_tr,
                "attentions_tr": attentions_tr,
                "jsd_score_tr": jsd_score_tr,
            })

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
            wandb.log({
                "predictions_te": predictions_te,
                "attentions_te": attentions_te,
                "jsd_score_te": jsd_score_te,
            })

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
                wandb.log({
                    "best_loss": best_loss,
                })

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

        self.model.related_score(test_data.X,test_data.y,test_data.true_pred,test_data.gold_attns,train=False,preturb_x=True,X_PGDer=self.X_PGDer)

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
        Model = BC.Model
        self.model = Model.init_from_config(dirname, args)
        self.model.dirname = dirname
        self.metrics = calc_metrics_classification
        self.display_metrics = True

    def evaluate(self, test_data, save_results=False) :
        if self.model.adversarial or self.model.ours:
            predictions, attentions, jsd_score = self.model.evaluate(test_data.X, target_attn=test_data.gold_attns)
            predictions = np.array(predictions)
            test_metrics = self.metrics(np.array(test_data.y), predictions, np.array(test_data.true_pred), jsd_score)
        else :
            predictions, attentions, _ = self.model.evaluate(test_data.X)
            predictions = np.array(predictions)
            test_metrics = self.metrics(test_data.y, predictions)

        # wandb.log({
        #     "final_test_metrics": test_metrics
        # })

        # if self.display_metrics :
        #     print_metrics(test_metrics, adv=self.model.adversarial or self.model.ours)

        if save_results :
            f = open(self.model.dirname + '/evaluate.json', 'w')
            json.dump(test_metrics, f)
            f.close()

        test_data.yt_hat = predictions
        test_data.attn_hat = attentions
        return test_metrics,predictions, attentions
