# -*- coding: utf-8 -*-

"""
The loss functions that SAMRI uses.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from collections.abc import Callable, Sequence


class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        weight = torch.as_tensor(weight) if weight is not None else None
        self.register_buffer("class_weight", weight)
        self.class_weight: None | torch.Tensor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> from monai.losses.dice import *  # NOQA
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        num_of_classes = target.shape[1]
        if self.class_weight is not None and num_of_classes != 1:
            # make sure the lengths of weights are equal to the number of classes
            if self.class_weight.ndim == 0:
                self.class_weight = torch.as_tensor([self.class_weight] * num_of_classes)
            else:
                if self.class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            if self.class_weight.min() < 0:
                raise ValueError("the value/values of the `weight` should be no less than 0.")
            # apply class_weight to loss
            f = f * self.class_weight.to(f)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f



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
    def __init__(self, num_classes=3) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, target):
        y_pred = F.softmax(y_pred, dim=1).float()
        smooth = smooth=1e-5
        intersection = (target * y_pred).sum(axis=(-4,-2,-1))
        union_a = intersection
        union_b = target.sum(axis=(-4,-2,-1))
        dice_coef = (2 * intersection) / (union_a + 
                                            union_b + smooth)
        return (1- dice_coef).mean()
    
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

class BiDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,y_pred, target, smooth=1e-10):
        y_pred = F.sigmoid(y_pred).float()
        intersection = (y_pred * target).sum()
        sum_of_pred = y_pred.sum()
        sum_of_target = target.sum()
        dice_coef = (2 * intersection + smooth) / (sum_of_pred + 
                                            sum_of_target + smooth)
        return 1 - dice_coef
    
class BatchDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,y_pred, target, smooth=1e-10):
        y_pred = F.sigmoid(y_pred).float()
        intersection = (y_pred * target).sum(axis=(-3,-2,-1))
        sum_of_pred = y_pred.pow(2).sum(axis=(-3,-2,-1))
        sum_of_target = target.pow(2).sum(axis=(-3,-2,-1))
        dice_coef = (2 * intersection + smooth) / (sum_of_pred + 
                                            sum_of_target + smooth)
        return 1 - dice_coef.mean()

