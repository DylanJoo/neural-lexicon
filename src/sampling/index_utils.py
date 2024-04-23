import torch
import random
import numpy as np
from dataclasses import dataclass
from pyserini.encode import FaissRepresentationWriter
from pyserini.search import FaissSearcher
from .utils import batch_iterator
from .data_utils import build_mask 

@dataclass
class PRFDenseSearchResult:
    docid: str
    score: float
    vectors: [float]


class NegativeSpanMiner(FaissSearcher):
    """ Index spans for every documents, """

    def __init__(self, spans, clusters, index_dir=None):

        self.spans = spans
        self.clusters = clusters

        if index_dir is None:
            self.index, self.docids = None, None
            index_dir = '/home/dju/indexes/temp/temp.faiss'
        else:
            self.index, self.docids = self.load_index(index_dir)

        self.index_dir = index_dir

    def update_index(self, embed_vectors):

        n, d = vectors.shape
        embedding_writer = FaissRepresentationWriter(self.index_dir, d)
        embed_ids = list(range(n))

        with embedding_writer:
            for s, e in batch_iterator(embed_ids, 128, True):
                embedding_writer.write({
                    "id": embed_ids[s:e], "vector": embed_vectors[s:e]
                })
        print('index saved')

    def crop_depedent_from_docs_v1(
        self, 
        embeds_1, 
        embeds_2,
        indices,
        n=1, 
        k0=0, 
        k=20, 
        return_ids=False
    ):
        """
        param
        -----
        embeds: search negative documents via doc embedding 
        n: int, number negative samples for each embeds
        k1: int, the threshold of positive samples
        k2: int, the threshold of negative samples

        return
        ------
        vectors of the negatives with size of (batch_size x n)
        [todo] false negative filtering
        """
        if embeds_1.dim() == 1: # singe vetor
            embeds_1 = embeds_1.unsqueeze(0)
            embeds_2 = embeds_2.unsqueeze(0)

        _, I1, V1 = self.index.search_and_reconstruct(embeds_1, k)
        _, I2, V2 = self.index.search_and_reconstruct(embeds_2, k)

        ## collect into a group I1, I2
        batch_ids = []
        batch = []
        N = embeds_1.shape[0] * n
        while len(batch_ids) < N:
            for i, idx in enumerate(indices):
                # first ranking list filtering (this is EN)
                try:
                    bound = I1[i].index(idx)
                except:
                    bound = 0
                I1_i = I1[i][max(bound, k0):]
                V1_i = V1[i][max(bound, k0):]

                try:
                    bound = I2[i].index(idx)
                except:
                    bound = 0
                I2_i = I2[i][max(bound, k0):]
                V2_i = V2[i][max(bound, k0):]
                overlap = np.in1d(I2_i, I1_i)

                ## filter the overlapped and combine

                I_i = np.append(I1_i, I2_i[~overlap])
                V_i = np.append(V1_i, V2_i[~overlap], axis=0)

                j = random.sample(range(len(I_i)), 1)[0]
                batch_ids.append( I_i[j] ) 
                batch.append( torch.tensor(V_i[j, :]) )

        if return_ids:
            raise ValueError('no implemented yet')
        else:
            return torch.cat(batch).view(N, -1).to(embeds_1.device)

