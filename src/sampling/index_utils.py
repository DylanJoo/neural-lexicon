from dataclasses import dataclass
from pyserini.encode import FaissRepresentationWriter
from pyserini.search import FaissSearcher
from .utils import batch_iterator

@dataclass
class PRFDenseSearchResult:
    docid: str
    score: float
    vectors: [float]


class NegativeSpanMiner(FaissSearcher):
    """ Index spans for every documents, """

    def __init__(self, dataset=None, dir=None):

        dir = (dir or '/home/dju/indexes/temp/temp.faiss')
        index_dir = os.path.join(dir, folder)

        self.index, self.docids = self.load_index(self, index_dir)
        self.labels = dataset.clusters
        self.spans = dataset.spans

    def fit(self, embed_vectors):

        n, d = vectors.shape
        embedding_writer = FaissRepresentationWriter(self.index_dir, d)
        embed_ids = list(range(n))

        with embedding_writer:
            for s, e in batch_iterator(embed_ids, 128, True):
                embedding_writer.write(
                        {"id": embed_ids[s:e], "vector": embed_vectors[s:e]}
                )

    def get_by_topk(self, q_ids, q_embeds, n=1, k1=10, k2=100):

        if q_embeds.dim() == 1: # singe vetor
            q_embeds = q_embeds.unsqueeze(0)

        _, I, V = self.index.search_and_reconstruct(q_embeds, k)

        spans_of_q = []
        ## false negative filtering
        batch = []
        while len(batch) <= q_embeds.shape[0] * n:
            for idx_list in I:

        return _, I, V

