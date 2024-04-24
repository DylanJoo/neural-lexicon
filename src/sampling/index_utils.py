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
        k=20, 
        return_ids=False
    ):
        """
        param
        -----
        embeds: search negative documents via doc embedding 
        n: int, number negative samples for each embeds
        k0: int, the threshold of positive samples (not used)
        k: int, the threshold of negative samples

        return
        ------
        vectors of the negatives with size of (batch_size x n)
        [todo] false negative filtering
        """
        if embeds_1.dim() == 1: # singe vetor
            embeds_1 = embeds_1.unsqueeze(0)
            embeds_2 = embeds_2.unsqueeze(0)

        S1, I1, V1 = self.index.search_and_reconstruct(embeds_1, k)
        S2, I2, V2 = self.index.search_and_reconstruct(embeds_2, k)

        ## collect into a group I1, I2
        batch_docids = []
        batch = []
        N = embeds_1.shape[0] * n
        while len(batch_docids) < N:
            for i, idx in enumerate(indices):

                I1_i, I2_i = I1[i], I2[i]
                V1_i, V2_i = V1[i], V2[i]
                S1_i, S2_i = S1[i], S2[i]

                ## filter the overlapped and combine (harder a bit)
                overlap_1 = np.in1d(I1_i, I2_i)
                overlap_2 = np.in1d(I2_i, I1_i)

                I_i = np.append(I1_i[~overlap_1], I2_i[~overlap_2])
                V_i = np.append(V1_i[~overlap_1], V2_i[~overlap_2], axis=0)
                S_i = np.append(S1_i[~overlap_1], S2_i[~overlap_2])

                ## reordering the lists
                I_i = I_i[np.argsort(S_i)]
                V_i = V_i[np.argsort(S_i)]

                ## exclude the hit as well
                try:
                    hit = I_i.index(idx)
                except:
                    hit = -1

                ## find the hardest 
                j = 0
                while (j < len(I_i)):
                    if (I_i[j] not in batch_docids) and (j != hit):
                        batch_docids.append( I_i[j] ) 
                        batch.append( torch.tensor(V_i[j, :]) )
                        j += len(I_i)
                    j += 1

        if return_ids:
            # raise ValueError('no implemented yet')
            return batch_docids
        else:
            return torch.cat(batch).view(N, -1).to(embeds_1.device)

