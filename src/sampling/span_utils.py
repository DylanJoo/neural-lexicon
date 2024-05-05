import torch
from tqdm import tqdm
import collections
from scipy.sparse import csr_matrix
import numpy as np
from nltk.util import ngrams
from sklearn.metrics.pairwise import cosine_similarity

from .data_utils import add_bos_eos, build_mask # integrate
# from data_utils import add_bos_eos, build_mask # unit

def add_extracted_spans(
    encoder, 
    documents, 
    batch_size=64, 
    max_doc_length=384,
    ngram_range=(2,2),
    top_k_spans=5,
    bos_id=101,
    eos_id=102,
    return_doc_embeddings=False, 
    by_spans=False
):
    with torch.no_grad():
        all_doc_embeddings = []
        extracted_spans = []

        for batch_docs in tqdm(batch_iterator(documents, batch_size), \
		total=len(documents)//batch_size+1):
            ## document encoding
            tokens = [torch.Tensor([bos_id] + d[:(max_doc_length-2)] + [eos_id]) for d in batch_docs]
            tokens, mask = build_mask(tokens)
            tokens, mask = tokens.to(encoder.device), mask.to(encoder.device)

            batch_doc_embeddings = encoder.encode(tokens, mask)
            batch_doc_embeddings = batch_doc_embeddings.detach().cpu().numpy()

            ### additional record doc embeddings
            if return_doc_embeddings and (by_spans is False):
                all_doc_embeddings.append(batch_doc_embeddings)

            ## build the ngram candidate set
            X, candidate_span_mapping = get_candidate_spans(batch_docs, ngram_range)
            span_embeddings = calculate_span_embeddings(candidate_span_mapping, encoder, batch_size)

            ## calculate the document-candidate similarity
            for i, doc_embedding in enumerate(batch_doc_embeddings):
                candidate_indices = X[i].nonzero()[1]
                candidates = [candidate_span_mapping[j] for j in candidate_indices]
                candidate_embeddings = span_embeddings[candidate_indices]

                scores = cosine_similarity(
                        doc_embedding.reshape(1, -1), candidate_embeddings
                )
                key_spans = [(candidates[i_ngram], round(float(scores[0][i_ngram]), 4)) for i_ngram in scores.argsort()[0][-top_k_spans:]][::-1]     
                extracted_spans.append(key_spans)

                ### add average span embeddings
                if return_doc_embeddings and (by_spans is True):
                    avg_candidate_embeddings = candidate_embeddings[list(i for i in scores.argsort()[0][-top_k_spans:])]
                    all_doc_embeddings.append(avg_candidate_embeddings.mean(0)[None, ...])

    if return_doc_embeddings:
        return extracted_spans, np.vstack(all_doc_embeddings)
    else:
        return (extracted_spans, )

def calculate_span_embeddings(
    ngram_mapping, 
    encoder=None,
    batch_size=64,
    bos_id=101,
    eos_id=102,
):
    ## compute span embedding BxN H
    span_tokens = list(ngram_mapping.values())
    tokens = [ torch.Tensor([bos_id] + s + [eos_id]) for s in span_tokens ]
    tokens, mask = build_mask(tokens)

    ret = []
    for start, end in batch_iterator(tokens, batch_size, True):
        tokens, mask = tokens.to(encoder.device), mask.to(encoder.device)
        span_embeddings = encoder.encode(tokens[start:end], mask[start:end])
        span_embeddings = span_embeddings.detach().cpu()
        ret.append(span_embeddings)

    span_embeddings = torch.cat(ret).numpy()
    return span_embeddings

def get_candidate_spans(docs, ngram_range):
    bag_of_features = collections.defaultdict()
    bag_of_features.default_factory = bag_of_features.__len__
    j_indices, indptr = [], [0]

    for doc in docs:
        # remove the redundant ngrams
        feature_set = set([ngram_tuple for n in \
                range(ngram_range[0], ngram_range[1]+1) for ngram_tuple in \
                ngrams(doc, n)])

        # create a map to collect feature (ngram token indices)
        feature_indices = []
        for feature in feature_set:
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

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

