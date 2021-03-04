# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000, weight=None): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold] 
        return pixel_losses.mean()

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse

eps = 1e-6


def soft_dice_loss(outputs, targets, per_image=False, per_channel=False, reduction = "mean"):
    batch_size, n_channels = outputs.size(0), outputs.size(1)
    
    eps = 1e-6
    n_parts = 1
    if per_image:
        n_parts = batch_size
    if per_channel:
        n_parts = batch_size * n_channels
    
    dice_target = targets.contiguous().view(n_parts, -1).float()
    dice_output = outputs.contiguous().view(n_parts, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union)
    if reduction == "mean":
        loss = loss.mean()
    return loss

def dice_metric(preds, trues, per_image=False, per_channel=False, reduction = "mean"):
    preds = preds.float()
    return 1 - soft_dice_loss(preds, trues, per_image, per_channel, reduction)

class DiceMetric(nn.Module):
    def __init__(self, threshold, num_classes = 1):
        super(DiceMetric, self).__init__()
        self.threshold = threshold
        self.per_image = True
        self.per_channel = False
        self.num_classes = num_classes
    def forward(self, outputs, labels):
        ph, pw = outputs.size(2), outputs.size(3)
        h, w = labels.size(1), labels.size(2)
        # print(score.shape, target.shape, ph, pw, h, w)
        if ph != h or pw != w:
            outputs = F.upsample(
                    input=outputs, size=(h, w), mode='bilinear')

        if self.num_classes == 1:
            outputs = torch.sigmoid(outputs)
        if self.num_classes  == 2:
            outputs = torch.softmax(outputs)
            outputs = outputs[:,1:,:,:]
        predicts = (outputs > self.threshold).float()
        return dice_metric(predicts, labels, self.per_image, self.per_channel, reduction= None)
