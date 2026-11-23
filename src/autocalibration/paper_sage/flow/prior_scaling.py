"""Parameter scaling fixed by transformed prior bounds, independent of data."""
import math
import torch

def fixed_theta_standardizer(transform):
    from crqsf_data import Standardizer
    lower=transform.lower.detach().clone().float();upper=transform.upper.detach().clone().float()
    for i,kind in enumerate(transform.kinds):
        if kind=='log10':lower[i]=lower[i].log10();upper[i]=upper[i].log10()
        elif kind!='linear':raise ValueError(kind)
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (upper<=lower).any():raise ValueError('Invalid transformed prior bounds')
    return Standardizer((lower+upper)/2.,(upper-lower)/math.sqrt(12.))
