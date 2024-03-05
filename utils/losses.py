# -*- coding: utf-8 -*-

"""
The loss functions that MRI-SAM uses.
"""
import numpy as np

def dice_similarity(y_true, y_pred, smooth=1e-10):
    """
    Calculate the dice similarity of the two images, dice similarity = (2 * 
    intersection + smooth) / (sum of squares of prediction + sum of squares 
    of ground truth + smooth)

    Parameters:
        y_true (np.array): the gound truth of the output
        y_pred (np.array): the predicted output
        smooth ()
    """
    intersection = np.sum(y_true * y_pred)
    sum_of_squares_pred = np.sum(np.square(y_pred))
    sum_of_squares_true = np.sum(np.square(y_true))
    dice = (2 * intersection + smooth) / (sum_of_squares_pred + 
                                          sum_of_squares_true + smooth)
    return dice