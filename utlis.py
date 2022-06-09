import torch
import numpy as np


def intersection_of_two_tensor(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection

def topk_overlap_loss(gt,pred,K=2):
    idx = torch.argsort(gt,dim=1,descending=True)
    print(idx)
    idx = idx[:,:K]
    pred_TopK_1 = pred.gather(1, idx)
    gt_Topk_1 = gt.gather(1,idx)

    idx_pred = torch.argsort(pred,dim=1,descending=True)
    idx_pred = idx_pred[:,:K]
    gt_TopK_2 = gt.gather(1, idx_pred)
    pred_TopK_2 = pred.gather(1, idx_pred)

    loss = torch.abs((pred_TopK_1 - gt_Topk_1)) + torch.abs(gt_TopK_2 - pred_TopK_2)
    loss = loss.sum()/(2*K)
    print(loss)
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
        np.array([[1, 2, 3, 4],
                  [2, 1, 3, 7]],),requires_grad=True, dtype=torch.double
    )
    print(t1.shape)
    t2 = torch.tensor(
        np.array([[1, 2, 3, 4],
                  [2, 1, 6, 7]]),requires_grad=True, dtype=torch.double
    )
    print(t2.shape)



    test = gradcheck(lambda t1,t2: topk_overlap_loss(t1,t2), (t1,t2))
    print("Are the gradients correct: ", test)

    # N = 2
    # for i in range(N):
    #     inset = intersection_of_two_tensor(t1[i],t2[i])
    #     print(inset.size())
    # inputs = torch.randn((10, 5), requires_grad=True, dtype=torch.double)
    # linear = nn.Linear(5, 3)
    # linear = linear.double()
    # def overlap_sum(t1,t2):
    #     # x = t1,t2
    #     N = t1.shape[0]
    #     loss = 0
    #     for i in range(N):
    #         inset = intersection_of_two_tensor(t1[i],t2[i])
    #         # print(inset.size())
    #         inset[inset != 0] = 1
    #         loss += inset.sum()
    #         print(loss)
    #     return loss
