"""
move this to span.py
and move the one in utils.py into a new span_utils.py
"""
import torch
from tqdm import tqdm
import collections
from scipy.sparse import csr_matrix
import numpy as np
from nltk.util import ngrams

from .data_utils import add_bos_eos, build_mask # integrate
from .utils import cosine_top_k, mmr_top_k

from pyserini.encode import FaissRepresentationWriter

def add_extracted_spans(
    encoder, 
    documents, 
    batch_size=64, 
    max_doc_length=384,
    ngram_range=(2,2),
    top_k_spans=5,
    bos_id=101,
    eos_id=102,
    faiss_output_dir=None, 
    span_selection='cosine'
):
    writer = None
    with torch.no_grad():
        extracted_spans = []

        doc_i = 0
        for batch_docs in tqdm(batch_iterator(documents, batch_size), \
		total=len(documents)//batch_size+1):
            ## document encoding
            tokens = [torch.Tensor([bos_id] + d[:(max_doc_length-2)] + [eos_id]) for d in batch_docs]
            tokens, mask = build_mask(tokens)
            tokens, mask = tokens.to(encoder.device), mask.to(encoder.device)

            batch_doc_embeddings = encoder.encode(tokens, mask)
            batch_doc_embeddings = batch_doc_embeddings.detach().cpu().numpy()

            if faiss_output_dir is not None:
                n, d = batch_doc_embeddings.shape
                docids = [j+doc_i for j in range(n)]

                if writer is None:
                    writer = FaissRepresentationWriter(faiss_output_dir, d)
                with writer:
                    writer.write({"id": docids, "vector": batch_doc_embeddings})
                doc_i += n
            
            ## build the ngram candidate set
            X, candidate_span_mapping = get_candidate_spans(batch_docs, ngram_range)
            span_embeddings = calculate_span_embeddings(candidate_span_mapping, encoder, batch_size)

            ## calculate the document-candidate similarity
            for i, doc_embedding in enumerate(batch_doc_embeddings):
                candidate_indices = X[i].nonzero()[1]
                candidates = [candidate_span_mapping[j] for j in candidate_indices]
                candidate_embeddings = span_embeddings[candidate_indices]

                if 'cosine' in span_selection:
                    key_spans = cosine_top_k(
                            doc_embedding, 
                            candidates,
                            candidate_embeddings,
                            top_k_spans
                    )
                elif 'mmr' in span_selection:
                    key_spans = mmr_top_k(
                            doc_embedding, 
                            candidates,
                            candidate_embeddings,
                            top_k_spans,
                            diversity=0.1
                    )

                extracted_spans.append(key_spans)

    return extracted_spans

# orginal independent calculation
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


def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

