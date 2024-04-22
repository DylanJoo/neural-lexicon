import os
import glob
import torch
import random
import numpy as np
from collections import defaultdict

def randomcrop(x, ratio_min, ratio_max):

    ratio = random.uniform(ratio_min, ratio_max)
    length = int(len(x) * ratio)
    start = random.randint(0, len(x) - length)
    end = start + length
    crop = x[start:end]
    return crop

def span_randomcrop(x, y, ratio_min, ratio_max, x_old):

    ratio = random.uniform(ratio_min, ratio_max)
    length = int(len(x_old) * ratio)

    start = random.randint(0, len(x) - length)
    end = start + length

    start_bound = x.index(y[0])
    end_bound = x.index(y[-1])

    if start <= start_bound:
        if end >= end_bound:
            pass
        else:
            end = end_bound
        crop = x[start:end][:length]
    else:
        start = start_bound
        crop = x[start:end][:length]

    return crop


def build_mask(tensors):
    shapes = [x.shape for x in tensors]
    maxlength = max([len(x) for x in tensors])
    returnmasks = []
    ids = []
    for k, x in enumerate(tensors):
        returnmasks.append(torch.tensor([1] * len(x) + [0] * (maxlength - len(x))))
        ids.append(torch.cat((x, torch.tensor([0] * (maxlength - len(x))))))
    ids = torch.stack(ids, dim=0).long()
    returnmasks = torch.stack(returnmasks, dim=0).bool()
    return ids, returnmasks


def add_token(x, token):
    x = torch.cat((torch.tensor([token]), x))
    return x


def deleteword(x, p=0.1):
    mask = np.random.rand(len(x))
    x = [e for e, m in zip(x, mask) if m > p]
    return x


def replaceword(x, min_random, max_random, p=0.1):
    mask = np.random.rand(len(x))
    x = [e if m > p else random.randint(min_random, max_random) for e, m in zip(x, mask)]
    return x


def maskword(x, mask_id, p=0.1):
    mask = np.random.rand(len(x))
    x = [e if m > p else mask_id for e, m in zip(x, mask)]
    return x

def maskword_from_span(x, mask_id, span):
    x = [e if e not in span else mask_id for e in x]
    return x

def shuffleword(x, p=0.1):
    count = (np.random.rand(len(x)) < p).sum()
    """Shuffles any n number of values in a list"""
    indices_to_shuffle = random.sample(range(len(x)), k=count)
    to_shuffle = [x[i] for i in indices_to_shuffle]
    random.shuffle(to_shuffle)
    for index, value in enumerate(to_shuffle):
        old_index = indices_to_shuffle[index]
        x[old_index] = value
    return x


def apply_augmentation(x, opt, span=None):
    if opt.augmentation == "mask":
        return torch.tensor(maskword(x, mask_id=opt.mask_id, p=opt.prob_augmentation))
    elif opt.augmentation == "mask_from_span":
        return torch.tensor(maskword_from_span(x, mask_id=opt.mask_id, span=span))
    elif opt.augmentation == "replace":
        return torch.tensor(
            replaceword(x, min_random=opt.start_id, max_random=opt.vocab_size - 1, p=opt.prob_augmentation)
        )
    elif opt.augmentation == "delete":
        return torch.tensor(deleteword(x, p=opt.prob_augmentation))
    elif opt.augmentation == "shuffle":
        return torch.tensor(shuffleword(x, p=opt.prob_augmentation))
    else:
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        return x


def add_bos_eos(x, bos_token_id, eos_token_id):
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    if x.dim() == 1:
        x = torch.cat([torch.Tensor([bos_token_id]), x.clone().detach(), torch.Tensor([eos_token_id])])
    else:
        x = torch.cat([
            torch.Tensor([[bos_token_id]] * x.size(0)),
            x.clone().detach(), 
            torch.Tensor([[eos_token_id]] * x.size(0)),
        ], dim=-1)
    return x

