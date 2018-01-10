
import math

import torch


def dynamic_ktop(l, L, s, min_ktop):
    """
    Computes the ktop parameter for layer l in a stack of L layers
    across a sequence of length s.
    """
    return max(min_ktop, math.ceil(((L - l) / L) * s))


def global_kmax_pool(t, ktop, dim=3):
    """
    Return the ktop max elements for each feature across dimension 3
    respecting the original order.
    """
    _, indices = t.topk(ktop, dim=dim)
    indices, _ = indices.sort(dim=dim)
    return t.gather(dim, indices)


def folding(t, factor=2, dim=2):
    """
    Applies folding across the height (embedding features) of the feature
    maps of a given convolution. Folding can be seen as applying local
    sum-pooling across the feature dimension (instead of the seq_len dim).
    """
    rows = [fold.sum(2) for fold in t.split(factor, dim=dim)]
    return torch.stack(rows, dim)


def get_padding(filter_size, mode='wide'):
    """
    Get padding for the current convolutional layer according to different
    schemes.

    Parameters:
    -----------
    filter_size: int
    mode: str, one of 'wide', 'narrow'
    """
    pad = 0
    if mode == 'wide':
        pad = math.floor(filter_size / 2) * 2

    return pad
