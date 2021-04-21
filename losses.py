import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class KDLoss(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, cls_num_list, T, weight=None):
        super(KDLoss, self).__init__()
        self.T = T
        # self.T = torch.cuda.FloatTensor([1, 2, 3, 4, 5, 6, 7, 4.5, 5, 5.5])
        self.weight = weight
        self.class_freq = torch.cuda.FloatTensor(cls_num_list / np.sum(cls_num_list))
        self.CELoss = nn.CrossEntropyLoss(weight=self.weight).cuda()

    def forward(self, out_s, out_t, target, alpha):
        kd = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                      F.softmax(out_t / self.T, dim=1),
                      reduction='none').mean(dim=0)
        kd_loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T
        ce_loss = self.CELoss(out_s, target)
        loss = alpha * kd_loss + ce_loss

        return loss, kd


class BKDLoss(nn.Module):

    def __init__(self, cls_num_list, T, weight=None):
        super(BKDLoss, self).__init__()
        self.T = T
        self.weight = weight
        self.class_freq = torch.cuda.FloatTensor(cls_num_list / np.sum(cls_num_list))
        self.CELoss = nn.CrossEntropyLoss().cuda()

    def forward(self, out_s, out_t, target, alpha):
        pred_t = F.softmax(out_t/self.T, dim=1)
        if self.weight is not None:
            pred_t = pred_t * self.weight
            pred_t = pred_t / pred_t.sum(1)[:, None]
        kd = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        pred_t,
                        reduction='none').mean(dim=0)
        kd_loss = kd.sum() * self.T * self.T
        ce_loss = self.CELoss(out_s, target)
        loss = alpha * kd_loss + ce_loss

        return loss, kd

