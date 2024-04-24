"""
Baseline: 
    X. Independant cropping: positive sampling. length-wise, dataset-wise
    1. Document-wise independent cropping: positive sampling, document-wise
    2. Document-wise independent cropping: plus in-document negative samples
"""
import os
import glob
import sys
import logging
import random
import datetime
import torch
import faiss

from .utils import batch_iterator
from .data_utils import *
from .span_utils import add_extracted_spans
from .cluster_utils import FaissKMeans

logger = logging.getLogger(__name__)

# [todo] add the `doc.by.spans` type of data as new condition.
def load_dataset(opt, tokenizer):
    if opt.loading_mode == "from_scratch": 
        files = glob.glob(os.path.join(opt.train_data_dir, "*corpus*.pkl"))
        assert len(files) == 1, 'more than one files'
        list_of_token_ids = torch.load(files[0], map_location="cpu")
        dataset = ClusteredIndCropping(opt, list_of_token_ids, tokenizer)
        return dataset

    elif opt.loading_mode == "from_precomputed": 
        files = glob.glob(os.path.join(opt.train_data_dir, "*.pt"))
        assert len(files) <= 1, 'more than one files'
        if len(files) == 0: # means precomputed one is not there, run one.
            sys.exit('run the `precompute.py` first')
        else:
            return torch.load(files[0], map_location="cpu")

    elif opt.loading_mode == "from_strong_precomputed": 
        files = glob.glob(os.path.join(opt.train_data_dir, "*.pt.strong"))
        assert len(files) <= 1, 'more than one files'
        if len(files) == 0: # means precomputed one is not there, run one.
            sys.exit('run the `precompute.py` first')
        else:
            return torch.load(files[0], map_location="cpu")

class ClusteredIndCropping(torch.utils.data.Dataset):

    def __init__(self, opt, documents, tokenizer):
        super().__init__()
        self.documents = [d for d in documents if len(d) > opt.chunk_length]
        self.chunk_length = opt.chunk_length
        self.tokenizer = tokenizer
        self.opt = opt
        self.opt.mask_id = tokenizer.mask_token_id 

        # span attrs
        self.spans = None
        self.spans_msim = 0 # the larger the better
        self.select_span_mode = opt.select_span_mode
        self.get_from_span = False

        # cluster attrs
        self.clusters = None
        self.clusters_sse = 999 # the small the better

        # search attrs
        self.nminer = None

    def init_spans(
        self,
        encoder, 
        batch_size=64, 
        max_doc_length=256, 
        ngram_range=(2,3), 
        top_k_spans=5,
        return_doc_embeddings=False,
        doc_embeddings_by_spans=False
    ):
        """ 
        This is for precomputing process, 
        The document for spannign is entire corpus,
        but in fact doing them in batch.
        """
        outputs = add_extracted_spans(
                encoder=encoder,
                documents=self.documents,
                batch_size=batch_size,
                max_doc_length=max_doc_length,
                ngram_range=ngram_range,
                top_k_spans=top_k_spans,
                bos_id=self.tokenizer.bos_token_id,
                eos_id=self.tokenizer.eos_token_id,
                return_doc_embeddings=return_doc_embeddings,
                by_spans=doc_embeddings_by_spans
        )
        self.spans = outputs[0]
        return outputs[1] if return_doc_embeddings else 0

    # [TODO] some alternatives: minibatch
    def init_clusters(
        self,
        embeddings,
        embeddings_for_kmeans=None,
        n_clusters=0.05,
        min_points_per_centroid=32,
        device='cpu',
        **cluster_args
    ):
        """ 
        Should work with constructing span embeddings 
        """
        # this can be the subset of the entire doc embeddings
        if embeddings_for_kmeans is None:
            embeddings_for_kmeans = embeddings

        # TAS exps on MARCO was set default to 5%
        if isinstance(n_clusters, float):
            n_clusters = embeddings_for_kmeans.shape[0] * n_clusters

        n_clusters_used = min(embeddings_for_kmeans.shape[0] // min_points_per_centroid, n_clusters)

        kmeans = FaissKMeans(
                n_clusters=n_clusters_used,
                min_points_per_centroid=min_points_per_centroid,
                device=device,
                **cluster_args
        )
        kmeans.fit(embeddings_for_kmeans)

        self.clusters = kmeans.assign(embeddings).flatten()
        self.clusters_sse = kmeans.inertia_

        return n_clusters_used

    def _select_spans(self, index):
        candidates, scores = list(zip(*self.spans[index]))
        if self.select_span_mode == 'weighted':
            span_tokens = random.choices(candidates, weights=scores, k=1)[0] # sample by the cosine
        elif self.select_span_mode == 'max':
            span_tokens = candidates[0]
        elif self.select_span_mode == 'random':
            span_tokens = random.choices(candidates, k=1)[0] 
        else:
            span_tokens = None
        return span_tokens

    def __getitem__(self, index):
        document = self.documents[index] # the crop is belong to one doc
        span_tokens = self._select_spans(index)

        start_idx = random.randint(0, len(document) - self.chunk_length)
        end_idx = start_idx + self.chunk_length 
        tokens = document[start_idx:end_idx] 

        # normal chunking
        bos, eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        q_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        q_tokens = apply_augmentation(q_tokens, self.opt, span_tokens)
        q_tokens = add_bos_eos(q_tokens, bos, eos)

        c_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        c_tokens = apply_augmentation(c_tokens, self.opt, span_tokens)
        c_tokens = add_bos_eos(c_tokens, bos, eos)

        if span_tokens is not None:
            span_tokens = add_bos_eos(span_tokens, bos, eos)

        return {"q_tokens": q_tokens, "c_tokens": c_tokens, "span_tokens": span_tokens, "data_index": index}

    def __len__(self):
        return len(self.documents)

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as fout:
            torch.save(self, fout)

