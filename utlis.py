import torch
import numpy as np


def intersection_of_two_tensor(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection

def topK_overlap_true_loss(a,b,K=2):
    t1 = torch.argsort(a, descending=True)
    t2 = torch.argsort(b, descending=True)
    t1 = t1.detach().cpu().numpy()
    t2 = t2.detach().cpu().numpy()
    N = t1.shape[0]
    loss = []
    for i in range(N):
        inset = np.intersect1d(t1[i,:K],t2[i,:K])
        overlap = len(inset)/K
        # print(overlap)
        loss.append(overlap)
    return np.mean(loss)


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean

    def total(self):
        return self.sum



def topk_overlap_loss(gt,pred,K=2,metric='l1'):
    idx = torch.argsort(gt,dim=1,descending=True)
    # print(idx)
    idx = idx[:,:K]
    pred_TopK_1 = pred.gather(1, idx)
    gt_Topk_1 = gt.gather(1,idx)

    idx_pred = torch.argsort(pred,dim=1,descending=True)
    idx_pred = idx_pred[:,:K]
    gt_TopK_2 = gt.gather(1, idx_pred)
    pred_TopK_2 = pred.gather(1, idx_pred)

    if metric == 'l1':
        loss = torch.abs((pred_TopK_1 - gt_Topk_1)) + torch.abs(gt_TopK_2 - pred_TopK_2)
        loss = loss.sum()/(2*K)
    elif metric == "l2":
        loss = torch.norm(pred_TopK_1 - gt_Topk_1, p=2) + torch.norm(gt_TopK_2 - pred_TopK_2, p=2)
        loss = loss.sum()/(2*K)
    elif metric == "kl":
        loss = torch.nn.functional.kl_div(gt,pred)
    elif metric == "jsd":
        loss = torch.nn.functional.kl_div(gt,pred) + torch.nn.functional.kl_div(pred,gt)
        loss /= 2
    return  loss

if __name__ == '__main__':

    # print(
    #     intersection_of_two_tensor(a,b)
    # )
    # combined = torch.cat((t1, t2), dim=1)
    # print(combined)
    # uniques, counts = combined.unique(return_counts=True, dim=1)
    # # intersection = uniques[counts > 1]
    # print(uniques,counts)
    from torch.autograd import gradcheck
    import torch
    import torch.nn as nn

    # intersection_of_two_tensor(t1[i], t2[i])

    t1 = torch.tensor(
        np.array([[100, 2, 3, 4],
                  [2, 1, 3, 7]],),requires_grad=True, dtype=torch.double
    )
    print(t1.shape)
    t2 = torch.tensor(
        np.array([[1, 2, 3, 4],
                  [2, 4, 6, 7]]),requires_grad=True, dtype=torch.double
    )
    print(t2.shape)



    # test = gradcheck(lambda t1,t2: topk_overlap_loss(t1,t2), (t1,t2))
    # print("Are the gradients correct: ", test)

    # N = 2
    # for i in range(N):
    #     inset = intersection_of_two_tensor(t1[i],t2[i])
    #     print(inset.size())
    # inputs = torch.randn((10, 5), requires_grad=True, dtype=torch.double)
    # linear = nn.Linear(5, 3)
    # linear = linear.double()


    print(topK_overlap_true_loss(torch.argsort(t1,descending=True),torch.argsort(t2,descending=True),K=2))
