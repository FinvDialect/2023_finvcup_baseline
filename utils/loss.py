'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import *

import numpy as np
import time
DISTANCE_MATRIX = np.array([[ 0, 32, 23, 34, 69, 68, 57, 79, 52],
                            [32,  0, 38, 25, 66, 61, 50, 77, 44],
                            [23, 38,  0, 40, 69, 71, 62, 79, 56],
                            [34, 25, 40,  0, 66, 63, 54, 77, 34],
                            [69, 66, 69, 66,  0, 67, 68, 71, 68],
                            [68, 61, 71, 63, 67,  0, 41, 78, 64],
                            [57, 50, 62, 54, 68, 41,  0, 76, 57],
                            [79, 77, 79, 77, 71, 78, 76,  0, 77],
                            [52, 44, 56, 34, 68, 64, 57, 77,  0]])
class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s):
        
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss

class StandardSimilarityLoss(nn.Module):
    def __init__(self, n_class, n_model=192, loss_function=nn.MSELoss, distance_matrix=DISTANCE_MATRIX):
        
        super(StandardSimilarityLoss, self).__init__()
        self.linear = nn.Linear(n_model, n_class)
        self.loss_function = loss_function()
        self.distance_matrix = distance_matrix
    def forward(self, x, label):
        n = x.shape[0]
        y_true = torch.zeros((n, 9))
        for i in range(n):
            y_true[i] = torch.from_numpy(100 - self.distance_matrix[label[i]][:9])
        y_true = y_true.to(x.device)
        loss = self.loss_function(self.linear(x), y_true)
        return loss

class PairDistanceLoss(nn.Module):
    def __init__(self, loss_function=nn.MSELoss, distance_matrix=DISTANCE_MATRIX):
        super(PairDistanceLoss, self).__init__()
        self.loss_function = loss_function()
        self.distance_matrix = DISTANCE_MATRIX
    def forward(self, y_pred, labels):
        n = y_pred.shape[0]
        y_true = torch.zeros((n,n))
        for i in range(n):
            for j in range(n):
                y_true[i][j] = self.distance_matrix[labels[i]][labels[j]]  
        y_true = y_true.to(y_pred.device)
        dist = torch.cdist(y_pred, y_pred, p=2)
        loss = self.loss_function(dist, y_true)
        return loss

