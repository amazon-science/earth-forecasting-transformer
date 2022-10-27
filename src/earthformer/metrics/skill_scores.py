"""Code is adapted from https://github.com/MIT-AI-Accelerator/neurips-2020-sevir. Their license is MIT License."""

import numpy as np
import torch
from torchmetrics import Metric
from torch import nn
from torch.nn import init
import torch.nn.functional as F


def _threshold(target, pred ,T):
    """
    Returns binary tensors t,p the same shape as target & pred.  t = 1 wherever
    target > t.  p =1 wherever pred > t.  p and t are set to 0 wherever EITHER
    t or p are nan.
    This is useful for counts that don't involve correct rejections.

    Parameters
    ----------
    target
        torch.Tensor
    pred
        torch.Tensor
    T
        numeric_type:   threshold
    Returns
    -------
    t
    p
    """
    t = (target >= T).float()
    p = (pred >= T).float()
    is_nan = torch.logical_or(torch.isnan(target),
                              torch.isnan(pred))
    t[is_nan] = 0
    p[is_nan] = 0
    return t, p

def _calc_hits_misses_fas(t, p):
    hits = torch.sum(t * p)
    misses = torch.sum(t * (1 - p))
    fas = torch.sum((1 - t) * p)
    return hits, misses, fas

def _pod(target, pred ,T, eps=1e-6):
    """
    Single channel version of probability_of_detection
    """
    t, p = _threshold(target, pred ,T)
    hits, misses, fas = _calc_hits_misses_fas(t, p)
    # return (hits + eps) / (hits + misses + eps)
    return hits / (hits + misses + eps)


def _sucr(target, pred, T, eps=1e-6):
    """
    Single channel version of success_rate
    """
    t, p = _threshold(target, pred, T)
    hits, misses, fas = _calc_hits_misses_fas(t, p)
    # return (hits + eps) / (hits + fas + eps)
    return hits / (hits + fas + eps)

def _csi(target, pred, T, eps=1e-6):
    """
    Single channel version of csi
    """
    t, p = _threshold(target, pred, T)
    hits, misses, fas = _calc_hits_misses_fas(t, p)
    # return (hits + eps) / (hits + misses + fas + eps)
    return hits / (hits + misses + fas + eps)

def _bias(target, pred, T, eps=1e-6):
    """
    Single channel version of csi
    """
    t, p = _threshold(target, pred, T)
    hits, misses, fas = _calc_hits_misses_fas(t, p)
    # return (hits + fas + eps) / (hits + misses + eps)
    return (hits + fas) / (hits + misses + eps)
