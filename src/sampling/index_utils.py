import torch
import random
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

        self.spans = dataset.spans
        self.clusters = dataset.clusters
        self.documents = dataset.documents
        self.additional_log = {}

        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id

        if index_dir is None:
            self.index, self.docids = None, None
            index_dir = '/home/dju/indexes/temp/temp.faiss'
        else:
            self.index, self.docids = self.load_index(index_dir)

        if 'span' in index_dir:
            self.use_doc_by_span = True
            self.use_doc_by_doc = False
        else:
            self.use_doc_by_doc = True
            self.use_doc_by_span = False

    @staticmethod
    def save_index(embed_vectors, index_dir=None):

        n, d = embed_vectors.shape
        index_dir = (index_dir or self.index_dir)
        embedding_writer = FaissRepresentationWriter(index_dir, d)
        embed_ids = list(range(n))

        with embedding_writer:
            for s, e in batch_iterator(embed_ids, 128, True):
                embedding_writer.write({
                    "id": embed_ids[s:e], "vector": embed_vectors[s:e]
                })
        print('index saved')

    def crop_depedent_from_docs(
        self, 
        embeds_1, 
        embeds_2,
        indices,
        n=1, 
        k0=0, 
        k=100, 
        exclude_overlap=True,
        return_token_ids=False,
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
        vectors of the negatives with size of (batch_size x n)
        [todo] false negative filtering
        """
        if embeds_1.dim() == 1: # singe vetor
            embeds_1 = embeds_1.unsqueeze(0)
            embeds_2 = embeds_2.unsqueeze(0)

        ## search topK for each crops
        S1, I1, V1 = self.index.search_and_reconstruct(embeds_1, k)
        S2, I2, V2 = self.index.search_and_reconstruct(embeds_2, k)

        overlap_rate = []
        batch_docids = []
        batch = []
        N = embeds_1.shape[0] * n # n*batch_size of negative
        while len(batch_docids) < N:
            for i, idx in enumerate(indices):

                I1_i, I2_i = I1[i][k0:k], I2[i][k0:k]
                V1_i, V2_i = V1[i][k0:k], V2[i][k0:k]
                S1_i, S2_i = S1[i][k0:k], S2[i][k0:k]

                ## filtered the overlapped and combine (harder a bit)
                overlap_1 = np.in1d(I1_i, I2_i)
                overlap_2 = np.in1d(I2_i, I1_i)

                overlap_rate.append( sum(overlap_1) / (k-k0) )

                if exclude_overlap:
                    I_i = np.append(I1_i[~overlap_1], I2_i[~overlap_2])
                    V_i = np.append(V1_i[~overlap_1], V2_i[~overlap_2], axis=0)
                    S_i = np.append(S1_i[~overlap_1], S2_i[~overlap_2])
                else:
                    I_i = np.append(I1_i, I2_i)
                    V_i = np.append(V1_i, V2_i, axis=0)
                    S_i = np.append(S1_i, S2_i)

                ### reordering the lists via scores
                I_i = I_i[np.argsort(S_i)[::-1]]
                V_i = V_i[np.argsort(S_i)[::-1]]

                ## exclude the hit as well
                try:
                    hit = I_i.index(idx)
                except:
                    hit = 0

                ## find the hit and start from it 
                j = hit + 1
                while (j < len(I_i)):
                    if (I_i[j] not in batch_docids) and (j not in indices):
                        batch_docids.append( I_i[j] )
                        batch.append( torch.tensor(V_i[j, :]) )
                        j += len(I_i)
                    j += 1

        # reorganize
        self.additional_log = {'overlap_rate': np.mean(overlap_rate)}
        batch_docids = batch_docids[:N]
        batch = torch.stack(batch)
        batch = batch[:N]

        if return_token_ids:
            batch_token_ids = []
            for docid in batch_docids:
                candidates, scores = list(zip(*self.spans[docid]))
                span_tokens = random.choices(candidates, weights=scores, k=1)[0] # sample by the cosine
                batch_token_ids.append(add_bos_eos(span_tokens, self.bos, self.eos))

            # use documents
            # batch_token_ids = list(
            #         add_bos_eos(self.documents[docid][:128], self.bos, self.eos) \
            #                 for docid in batch_docids
            # )
            batch_inputs = build_mask(batch_token_ids)
            return batch_inputs
        else:
            return batch.to(embeds_1.device)

    def span_depedent_from_docs(
        self, 
        embeds, 
        indices,
        n=1, 
        k0=0, 
        k=20, 
        return_token_ids=False
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

        batch_docids = []
        batch = []
        N = embeds.shape[0] * n # n*batch_size of negative
        while len(batch_docids) < N:
            for i, idx in enumerate(indices):

                I_i, V_i, S_i = I[i], V[i], S[i]

                ## exclude the hit 
                try:
                    hit = I_i.index(idx) + 1
                except:
                    hit = -1

                ## find the hardest 
                #### sort by difference(gap)
                J = argdiff(S_i) 

                j = 0
                while (J[j] == hit) or (I_i[j] in batch_docids):
                    j += 1

                batch_docids.append( I_i[j] ) 
                batch.append( torch.tensor(V_i[j, :]) )

        # reorganize
        batch_docids = batch_docids[:N]
        batch = torch.stack(batch)
        batch = batch[:N]

        if return_token_ids:
            batch_token_ids = list(
                    add_bos_eos(self.documents[docid][:128], self.bos, self.eos) \
                            for docid in batch_docids
            )
            batch_inputs = build_mask(batch_token_ids)
            return batch_inputs
        else:
            return batch.to(embeds.device)

