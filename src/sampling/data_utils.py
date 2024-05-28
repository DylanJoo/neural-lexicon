import os
import glob
import torch
import random
import numpy as np
from collections import defaultdict
from nltk import ngrams
from .utils import get_candidate_spans, batch_iterator

def compute_span_embeds(
    inputs,
    mask,
    ngram_range,
    stride=1,
    token_embeds=None,
    span_pooling='max',
):
    """ compute similarity of contextualized n-gram
    Params
    ------
    token_input
    ngram_range 
    token_embeds
    doc_pooling
    span_pooling

    Returns
    -------

    """
    candidate_inputs = []
    candidate_embeds = []

    unigram = list(range(sum(mask) - 1))
    # add stride # skip the first one include cls token
    for n in range(ngram_range[0], ngram_range[1]+1):
        ngrams_set = [ngram for i, ngram in enumerate(ngrams(unigram, n)) if i % stride == 0][1:]
        for indices in ngrams_set:
            candidate_inputs.append( inputs[list(indices)].tolist() )
            candidate_embeds.append( token_embeds[list(indices), :].mean(axis=0).reshape(1, -1) )

    candidate_embeds = np.concatenate(candidate_embeds)
    return candidate_inputs, candidate_embeds

def batch_compute_span_embeds(
    encoder, 
    documents,
    bos, eos,
    batch_size,
    ngram_range,
    stride
):
    """ compute similarity of de-contextualized n-gram

    Params
    ------
    encoder
    documents 
    batch_size 
    ngram_range

    Returns
    -------
    span_embes (torch.tensor)
    doc_span_mapping (scipy.sparse_matrix)
    ngram_mapping (np.numpy)

    """
    # prepare big matrix to collect all span candidates
    X, mapping = get_candidate_spans(documents, ngram_range, stride)

    ## compute span embedding BxN H
    span_tokens = list(mapping.values())
    # tokens = [add_bos_eos(s, bos, eos) for s in span_tokens]
    tokens = [torch.Tensor([bos] + s + [eos]) for s in span_tokens]
    tokens, mask = build_mask(tokens)

    list_span_embed = []
    for start, end in batch_iterator(tokens, batch_size, True):
        tokens, mask = tokens.to(encoder.device), mask.to(encoder.device)
        outputs = encoder.encode(tokens[start:end], mask[start:end])
        span_embeds = outputs[0].detach().cpu()
        list_span_embed.append(span_embeds)

    span_embeds = torch.cat(list_span_embed).numpy()
    return span_embeds, X, mapping

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

def maskword_from_span(x, mask_id, span, p=0.5):
    applied = (np.random.uniform(0, 1, 1)[0] < p)
    if applied:
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
        return torch.tensor(maskword(x, mask_id=opt.mask_token_id, p=opt.prob_augmentation))
    elif opt.augmentation == "mask_from_span":
        return torch.tensor(maskword_from_span(x, mask_id=opt.mask_token_id, span=span, p=opt.prob_augmentation))
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

