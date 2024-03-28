# -*- coding: utf-8 -*-

"""
The loss functions that MRI-SAM uses.
"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

def dice_similarity(y_true, y_pred, smooth=1e-10):
    """
    Calculate the dice similarity of the two images, dice similarity = (2 * 
    intersection + smooth) / (sum of squares of prediction + sum of squares 
    of ground truth + smooth)

    Parameters:
        y_true (np.array): the gound truth of the output
        y_pred (np.array): the predicted output
        smooth (float): a small number to avoid zero denominator.
    """
    intersection = np.sum(y_true * y_pred)
    sum_of_squares_pred = np.sum(np.square(y_pred))
    sum_of_squares_true = np.sum(np.square(y_true))
    dice = (2 * intersection + smooth) / (sum_of_squares_pred + 
                                          sum_of_squares_true + smooth)
    return dice


def iou(y_true, y_pred, smooth=1e-10):
    intersection = np.sum(np.bitwise_and(y_true, y_pred))
    union = np.sum(np.bitwise_or(y_true, y_pred))
    return (intersection) /  (union + smooth)


def bce_dice_loss(y_true, y_pred):
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    dicescore = 1 - dice_similarity(y_true, y_pred)
    bcescore = nn.BCELoss()
    bceloss = bcescore(y_true, y_pred)

    return bceloss + dicescore

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, target):
        y_pred = F.softmax(y_pred, dim=1).float()

        smooth = smooth=1e-5
        intersection = (target * y_pred).sum()
        union_a = y_pred.sum()
        union_b = target.sum()
        dice_coef = (2 * intersection) / (union_a + 
                                            union_b + smooth)
        return 1 - dice_coef
    
class CeDiceLoss(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self,y_pred, target):
        y_pred = y_pred.float()
        target = target.float()

        dicescore = MultiClassDiceLoss(self.num_classes)
        diceloss = dicescore(y_pred, target)
        cescore = nn.CrossEntropyLoss()
        celoss = cescore(y_pred, target)

        return celoss + diceloss
