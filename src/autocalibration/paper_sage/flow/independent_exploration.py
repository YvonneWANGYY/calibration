"""Independent prior candidates without consuming the flow sampler RNG."""
import math
import torch

def replace_with_independent_prior(samples,theta_std,theta_transform,fraction,generator):
    if not math.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('Independent prior fraction must be between0 and1')
    count=round(len(samples)*fraction)
    mask=torch.zeros(len(samples),dtype=torch.bool,device=samples.device)
    if count==0:return samples,mask
    if theta_transform is None:raise ValueError('Independent exploration requires explicit coordinate bounds')
    low=theta_transform.lower.detach().cpu().clone();high=theta_transform.upper.detach().cpu().clone()
    for i,kind in enumerate(theta_transform.kinds):
        if kind=='log10':low[i]=low[i].log10();high[i]=high[i].log10()
    indices=torch.randperm(len(samples),generator=generator)[:count]
    coordinates=low+torch.rand(count,len(low),generator=generator)*(high-low)
    replacements=theta_std.transform(coordinates).to(samples)
    mixed=samples.clone();mixed[indices.to(samples.device)]=replacements;mask[indices.to(samples.device)]=True
    return mixed,mask
