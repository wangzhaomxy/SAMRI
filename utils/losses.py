# -*- coding: utf-8 -*-

"""
The loss functions that MRI-SAM uses.
"""
import numpy as np
import torch.nn as nn
import torch

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

class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_true, y_pred):
        y_true = y_true.float()
        y_pred = y_pred.float()

        smooth = smooth=1e-10
        intersection = torch.sum(y_true * y_pred)
        sum_of_squares_pred = torch.sum(torch.square(y_pred))
        sum_of_squares_true = torch.sum(torch.square(y_true))
        dice = 1 - (2 * intersection + smooth) / (sum_of_squares_pred + 
                                            sum_of_squares_true + smooth)
        return dice
    
class BceDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_true, y_pred):
        y_true = y_true.float()
        y_pred = y_pred.float()

        dicescore = DiceLoss()
        diceloss = dicescore(y_true, y_pred)
        bcescore = nn.BCELoss()
        bceloss = bcescore(y_true, y_pred)

        return bceloss + dicescore
