from collections import defaultdict
import os
import torch
import random
import faiss
import json
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from pyserini.search import FaissSearcher
from .utils import batch_iterator, argdiff
from .data_utils import add_bos_eos, build_mask

@dataclass
class PRFDenseSearchResult:
    docid: str
    score: float
    vectors: [float]

class NegativeSpanMiner:
    """ Index spans for every documents, """

    def __init__(self, opt, dataset, tokenizer):
        self.opt = opt

        self.dataset = dataset
        self.additional_log = {}

        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id

        ## precomputed negative
        self.negative_jsonl = ""
        self.negatives = []

        ## precomupted index. 
        ## It can be used for (1) mining static negative or (2) mining dynamic negative
        self.index_dir = ""
        self.index, self.docids = None, None

        ## Negative 
        if opt.prebuilt_negative_jsonl is not None:
            self.negatives = self._load_negatives()

        ### Index (if need to rebuild)
        if opt.prebuilt_faiss_dir is not None:
            self._load_index()

    def _load_index(self, faiss_dir=None):
        dir = (faiss_dir or self.opt.prebuilt_faiss_dir)
        self.index_dir = dir

        index_path = os.path.join(dir, 'index')
        self.index = faiss.read_index(index_path)
        # use dummy docids, which suit the getitem of dataset
        self.docids = list(range(self.index.ntotal)) 

    def _load_negatives(self, negative_jsonl=None):
        file = (negative_jsonl or self.opt.prebuilt_negative_jsonl)
        self.negative_jsonl = file

        with open(file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                try:
                    idx = item['docidx']
                except:
                    idx = len(self.negatives)

                negatives = item['negatives']
                self.negatives[int(idx)] = negatives

    ## [static mining]
    def batch_get_negative_inputs(self, indices, field='span_tokens'):
        indices_to_return = []
        tokens_to_return = []

        # prepare negative docs for batch (prevent redundnat neg)
        for idx in indices:
            candidates = self.negatives[int(idx)]
            if len(candidates) > 0:
                negative_idx = random.sample(candidates, 1)
                if negative_idx not in indices + indices_to_return:
                    indices_to_return.append(negative_idx[0])

        return self.prepare_input(indices_to_return)

    ## [static mining]
    def precompute_prior_negatives(self, encoder, batch_size=64, n_samples=1, top_k=100):

        with torch.no_grad():
            for s, e in tqdm(
                    batch_iterator(self.dataset, batch_size, True), 
                    total=len(self.dataset)//batch_size+1
            ):

                batch = defaultdict(list)
                for i in list(range(s, e)):
                    example = self.dataset[i]
                    batch['q_tokens'].append(example['q_tokens'])
                    batch['c_tokens'].append(example['c_tokens'])
                    batch['data_index'].append(example['data_index'])

                q_tokens, q_mask = build_mask(batch['q_tokens'])
                c_tokens, c_mask = build_mask(batch['c_tokens'])

                q_tokens, c_tokens = q_tokens.to(encoder.device), c_tokens.to(encoder.device)
                q_mask, c_mask = q_mask.to(encoder.device), c_mask.to(encoder.device)

                embeds_1 = encoder.encode(q_tokens, q_mask).detach().cpu()
                embeds_2 = encoder.encode(c_tokens, c_mask).detach().cpu()

                for i, (e1, e2) in enumerate(zip(embeds_1, embeds_2)):
                    negative = self.crop_depedent_from_docs(
                            e1, e2, batch['data_index'][i],
                            n=n_samples, k0=0, k=top_k,
                            exclude_overlap=False, 
                            start_from_hit=False, 
                            to_return='span', 
                            return_index=True
                    )
                    self.negatives.append(negative)

    def crop_depedent_from_docs(
        self, 
        embeds_1, 
        embeds_2,
        indices,
        n=1, k0=0, k=100, 
        exclude_overlap=True,
        start_from_hit=False,
        to_return='span',
    ):
        """
        param
        -----
        embeds: search negative documents via doc embedding 
        n: int, number negative samples for each embeds
        k0: int, the threshold of positive samples (not used)
        k: int, the threshold of negative samples
        exclude_overlap: bool, remove the docs from overlap, preventing false neg.

        return
        ------
        vectors of the negatives with size of (batch_size x n). 
        the negatives are from span embeddings
        """
        if embeds_1.dim() == 1: # singe vetor
            embeds_1 = embeds_1.unsqueeze(0)
            embeds_2 = embeds_2.unsqueeze(0)
            indices = [indices]

        ## search topK for each crops
        S1, I1 = self.index.search(embeds_1, k)
        S2, I2 = self.index.search(embeds_2, k)

        overlap_rate = []
        batch_docidx = []

        N = n * embeds_1.shape[0]

        for i, idx in enumerate(indices):

            ## filtered the overlapped and combine (harder a bit)
            I1_i, I2_i = I1[i][k0:k], I2[i][k0:k]
            S1_i, S2_i = S1[i][k0:k], S2[i][k0:k]
            overlap_1 = np.in1d(I1_i, I2_i)
            overlap_2 = np.in1d(I2_i, I1_i)

            ## exlude the overlap
            if exclude_overlap:
                I_i = np.append(I1_i[~overlap_1], I2_i[~overlap_2])
                S_i = np.append(S1_i[~overlap_1], S2_i[~overlap_2])
                ### reordering the lists via scores
                I_i = I_i[np.argsort(S_i)[::-1]]

            ## use the overlap
            else:
                I_i = I1_i[overlap_1]

            overall = len(I1_i) + len(I2_i) - sum(overlap_1)
            overlap_rate.append( sum(overlap_1) / overall )

            ## exclude repetitive and exclude the document
            I_i = [neg_idx for neg_idx in I_i if neg_idx not in indices + batch_docidx]

            batch_docidx += I_i[:n]

        # reorganize
        self.additional_log.update({'overlap_rate': np.mean(overlap_rate)})
        batch_docidx = batch_docidx[:N]
        # return batch_docidx

        return self.prepare_input(batch_docidx, 'span_tokens')

    def prepare_input(self, batch_docidx, field='span_tokens'):
        batch_token_ids = []
        for docidx in batch_docidx:
            batch_token_ids.append(self.dataset[int(docidx)][field])
        batch_inputs = build_mask(batch_token_ids)
        return batch_inputs

