import os
import torch
import random
import numpy as np
import collections 
from nltk import ngrams
from scipy.sparse import csr_matrix

from .data_utils import build_mask
from .utils import batch_iterator

def compute_span_embeds(
    inputs,
    mask,
    ngram_range,
    stride=1,
    token_embeds=None,
    span_pooling='mean',
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


def get_candidate_spans(docs, ngram_range, stride):
    bag_of_features = collections.defaultdict()
    bag_of_features.default_factory = bag_of_features.__len__
    j_indices, indptr = [], [0]

    # unigram = list(range(sum(mask)))
    # # add stride # skip the first one include cls token
    # for n in range(ngram_range[0], ngram_range[1]+1):
    #     ngrams_set = [ngram for i, ngram in enumerate(ngrams(unigram, n)) if i % stride == 0][1:]

    for doc in docs:
        # remove the redundant ngrams
        ngram_set = []
        for n in range(ngram_range[0], ngram_range[1]+1):
            ngram_set += [ngram for i, ngram in enumerate(ngrams(doc, n)) if i % stride == 0]

        ngram_set = set(ngram_set)
        # create a map to collect feature (ngram token indices)
        feature_indices = []
        for feature in ngram_set:
            idx = bag_of_features[feature]
            feature_indices += [idx]

        j_indices.extend(feature_indices)
        indptr.append(len(j_indices))

    # map to sparse array
    j_indices = np.asarray(j_indices, dtype=np.int64)
    indptr = np.asarray(indptr, dtype=np.int64)
    values = [1] * len(j_indices)
    X = csr_matrix((values, j_indices, indptr), shape=(len(indptr) - 1, len(bag_of_features)))

    # reverse the key value of bof
    feature_mapping = {v: list(k) for k, v in bag_of_features.items()}
    return X, feature_mapping

