import faiss
from operator import itemgetter
from pyserini.search import FaissSearcher
from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Tuple
import numpy as np

@dataclass
class DenseSearchResult:
    docid: str
    score: float

class FaissMaxSearcher(FaissSearcher): 
    
    def batch_search(self, queries: Union[List[str], np.ndarray], q_ids: List[str], k: int = 10,
                     threads: int = 1, return_vector: bool = False) -> Dict[str, List[DenseSearchResult]]: 

        Q = [self.query_encoder.encode(qtext) for qtext in queries]
        # q_embs = np.array([q[0] for q in Q]) # CLS pooling
        q_embs = np.array([q[1:].mean(0) for q in Q]) # mean exclude cls pooling

        n, m = q_embs.shape
        assert m == self.dimension

        faiss.omp_set_num_threads(threads) 

        D_, I_, V_ = self.index.search_and_reconstruct(q_embs, k)
        D, I = [], [] 

        for i, (indexes, vectors) in enumerate(zip(I_, V_)): 
            # reranking with maxsim
            scores = (Q[i][1:] @ vectors.T).max(0) # max(q_len n_docs)
            results = {idx: score for idx, score in zip(indexes, scores)}
            sorted_result = {k: v for k,v in sorted(results.items(), key=itemgetter(1), reverse=True)} 

            D.append(list(sorted_result.values()))
            I.append(list(sorted_result.keys()))

        return {key: [DenseSearchResult(self.docids[idx], score)
		      for score, idx in zip(distances, indexes) if idx != -1]
		for key, distances, indexes in zip(q_ids, D, I)}
