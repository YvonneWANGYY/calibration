"""Balanced member-specific folds using an isolated CPU RNG generator.

The production experiment keeps 20 seeds and shared all-data theta scaling;
these are cross-fitted neural networks, not fully held-out preprocessing folds.
"""
import torch


def member_splits(n_records, n_members, n_folds, seed):
    if any(isinstance(v,bool) or not isinstance(v,int) for v in (n_records,n_members,n_folds,seed)):
        raise ValueError('Record/member/fold counts and seed must be integers')
    if not 2 <= n_folds <= n_records or n_members <= 0 or n_members % n_folds:
        raise ValueError('Need at least two folds and positive equal members per fold')
    permutation=torch.randperm(n_records,generator=torch.Generator().manual_seed(seed))
    folds=torch.tensor_split(permutation,n_folds)
    return [(torch.cat([f for j,f in enumerate(folds) if j!=i%n_folds]),folds[i%n_folds]) for i in range(n_members)]


def validate_configuration(n_records, ensemble_seeds, split_seed, context_mode, test_fraction):
    if not 25 <= n_records <= 50:
        raise ValueError('Production five-fold run requires 25..50 records')
    if list(ensemble_seeds)!=list(range(41,61)) or split_seed!=41:
        raise ValueError('Five-fold experiment requires exactly 20 member seeds 41..60 and split seed 41')
    if context_mode!='scalar' or test_fraction!=0.:
        raise ValueError('Five-fold experiment supports scalar context and no test split')
