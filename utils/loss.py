'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import *

import numpy as np
import time
DISTANCE_MATRIX = np.array([[ 0. , 32.1, 23.4, 34.4, 68.7, 67.7, 57.1, 79.2, 51.7],
                            [32.1,  0. , 38.2, 24.5, 65.8, 60.7, 49.9, 76.6, 43.7],
                            [23.4, 38.2,  0. , 40.1, 68.6, 70.5, 61.8, 79.1, 55.9],
                            [34.4, 24.5, 40.1,  0. , 66. , 62.6, 54.2, 77.2, 34. ],
                            [68.7, 65.8, 68.6, 66. ,  0. , 67.1, 68.3, 71. , 68.4],
                            [67.7, 60.7, 70.5, 62.6, 67.1,  0. , 40.7, 78. , 63.9],
                            [57.1, 49.9, 61.8, 54.2, 68.3, 40.7,  0. , 76.2, 57.3],
                            [79.2, 76.6, 79.1, 77.2, 71. , 78. , 76.2,  0. , 77. ],
                            [51.7, 43.7, 55.9, 34. , 68.4, 63.9, 57.3, 77. ,  0. ]])
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

