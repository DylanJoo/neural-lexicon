import torch
import random
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from pyserini.encode import FaissRepresentationWriter
from pyserini.search import FaissSearcher
from .utils import batch_iterator, argdiff
from .data_utils import add_bos_eos, build_mask

@dataclass
class PRFDenseSearchResult:
    docid: str
    score: float
    vectors: [float]

class NegativeSpanMiner(FaissSearcher):
    """ Index spans for every documents, """

    def __init__(self, dataset, tokenizer, index_dir=None):

        self.dataset = dataset
        self.additional_log = {}

        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id

        if index_dir is None:
            self.index, self.docids = None, None
            index_dir = '/home/dju/indexes/temp/temp.faiss'
        else:
            self.index, self.docids = self.load_index(index_dir)

    @staticmethod
    def save_index(embed_vectors, index_dir=None):

        n, d = embed_vectors.shape
        index_dir = (index_dir or self.index_dir)
        embedding_writer = FaissRepresentationWriter(index_dir, d)
        embed_ids = list(range(n))

        with embedding_writer:
            for s, e in tqdm(batch_iterator(embed_ids, 128, True)):
                embedding_writer.write({
                    "id": embed_ids[s:e], "vector": embed_vectors[s:e]
                })
        print('index saved')

    def crop_depedent_from_docs(
        self, 
        embeds_1, 
        embeds_2,
        indices,
        n=1, k0=0, k=100, 
        exclude_overlap=True,
        start_from_hit=False,
        to_return='span',
        debug=None,
    ):
        """
        param
        -----
        embeds: search negative documents via doc embedding 
        n: int, number negative samples for each embeds
        k0: int, the threshold of positive samples (not used)
        k: int, the threshold of negative samples
        exclude_overlap: bool, remove the docs from overlap, preventing false neg.
        debug: bool, use the control the new setting.

        return
        ------
        vectors of the negatives with size of (batch_size x n). 
        the negatives are from span embeddings
        """
        if embeds_1.dim() == 1: # singe vetor
            embeds_1 = embeds_1.unsqueeze(0)
            embeds_2 = embeds_2.unsqueeze(0)

        ## search topK for each crops
        S1, I1, V1 = self.index.search_and_reconstruct(embeds_1, k)
        S2, I2, V2 = self.index.search_and_reconstruct(embeds_2, k)

        overlap_rate = []
        batch_docidx = []
        batch = []
        N = embeds_1.shape[0] * n # n*batch_size of negative
        while len(batch_docidx) < N:
            for i, idx in enumerate(indices):

                I1_i, I2_i = I1[i][k0:k], I2[i][k0:k]
                V1_i, V2_i = V1[i][k0:k], V2[i][k0:k]
                S1_i, S2_i = S1[i][k0:k], S2[i][k0:k]

                ## filtered the overlapped and combine (harder a bit)
                overlap_1 = np.in1d(I1_i, I2_i)
                overlap_2 = np.in1d(I2_i, I1_i)

                if exclude_overlap:
                    I_i = np.append(I1_i[~overlap_1], I2_i[~overlap_2])
                    V_i = np.append(V1_i[~overlap_1], V2_i[~overlap_2], axis=0)
                    S_i = np.append(S1_i[~overlap_1], S2_i[~overlap_2])
                else:
                    I_i = np.append(I1_i[~overlap_1], I2_i)
                    V_i = np.append(V1_i[~overlap_1], V2_i, axis=0)
                    S_i = np.append(S1_i[~overlap_1], S2_i)

                overlap_rate.append( sum(overlap_1) / (k-k0) )

                ### reordering the lists via scores
                I_i = I_i[np.argsort(S_i)[::-1]]
                V_i = V_i[np.argsort(S_i)[::-1]]

                if debug:
                    y = self.dataset.clusters[idx]

                    #### same-cluser-first-negative
                    hit_cluster = [1 if self.dataset.clusters[ii] == y else 2 \
                            for ii in I_i]
                    I_i = I_i[np.argsort(hit_cluster)]
                    V_i = V_i[np.argsort(hit_cluster)]

                ## find the hit if any
                try:
                    hit = I_i.index(idx)
                except:
                    hit = -1

                if start_from_hit:
                    j = hit + 1
                else:
                    j = 0

                ## exclude repetitive and exclude the document
                while (j < len(I_i)):
                    if (I_i[j] not in batch_docidx) and (j not in indices):
                        batch_docidx.append( I_i[j] )
                        batch.append( torch.tensor(V_i[j, :]) )
                        j += len(I_i)
                    j += 1

        # reorganize
        self.additional_log.update({'overlap_rate': np.mean(overlap_rate)})
        batch_docidx = batch_docidx[:N]
        batch = torch.stack(batch)
        batch = batch[:N]

        batch_token_ids = []

        for docidx in batch_docidx:
            neg_sample = self.dataset[docidx]

            # [old implmentation]
            # candidates, scores = list(zip(*self.dataset.spans[docidx]))
            # span_tokens = random.choices(candidates, weights=scores, k=1)[0] # sample by the cosine
            # batch_token_ids.append(add_bos_eos(span_tokens, self.bos, self.eos))

            ## 1. return spans
            ## 2. return dataset (bascially two crops)
            if to_return == 'span':
                batch_token_ids.append(neg_sample["span_tokens"])
            elif to_return == 'crop':
                batch_token_ids.append(neg_sample["q_tokens"])
                # batch_token_ids.append(neg_sample["c_tokens"]) # OOM

        batch_inputs = build_mask(batch_token_ids)
        return batch_inputs

    def span_depedent_from_docs(
        self, 
        embeds, 
        indices,
        n=1, 
        k0=0, 
        k=20, 
        start_from_hit=False,
    ):
        """
        param
        -----
        embeds: search negative documents via doc embedding (by spans average)
        n: int, number negative samples for each embeds
        k0: int, the threshold of positive samples (not used)
        k: int, the threshold of negative samples

        return
        ------
        vectors of the negatives with size of (batch_size x n)
        """
        if embeds.dim() == 1: # singe vetor
            embeds = embeds.unsqueeze(0)

        ## search topK for each representative-span
        S, I, V = self.index.search_and_reconstruct(embeds, k)

        arg_max_gap = []
        batch_docidx = []
        batch = []
        N = embeds.shape[0] * n # n*batch_size of negative
        while len(batch_docidx) < N:
            for i, idx in enumerate(indices):

                I_i, V_i, S_i = I[i], V[i], S[i]

                ## find the hit if any
                try:
                    hit = I_i.index(idx)
                except:
                    hit = -1

                ## find the hardest. sort by difference(gap)
                J = argdiff(S_i) 

                if start_from_hit:
                    select = hit + 1
                else:
                    select = 0

                while (select < len(I_i)):
                    if select == hit:
                        select += 1

                    j = J[select]
                    if (I_i[j] not in batch_docidx) and (j not in indices):
                        batch_docidx.append( I_i[J[j]] ) 
                        batch.append( torch.tensor(V_i[j, :]) )
                        select = len(I_i)
                        arg_max_gap.append(j)
                    select += 1

        # reorganize
        self.additional_log.update({'arg_max_gap': np.mean(arg_max_gap)})
        batch_docidx = batch_docidx[:N]
        batch = torch.stack(batch)
        batch = batch[:N]

        batch_token_ids = []
        for docidx in batch_docidx:
            candidates, scores = list(zip(*self.dataset.spans[docidx]))
            span_tokens = random.choices(candidates, weights=scores, k=1)[0] # sample by the cosine
            batch_token_ids.append(add_bos_eos(span_tokens, self.bos, self.eos))

        batch_inputs = build_mask(batch_token_ids)
        return batch_inputs

