import numpy as np
import torch
import random
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from torch.utils.data.sampler import SequentialSampler, BatchSampler

class BinSampler(SequentialSampler):
    """
    sampling from clusters into bins
    - bin_counts: the size of each bins.
        - the i-th element of list means the size of the cluster i
    - indices: the exact indices of the document
        - the i-th element of dict is a list of canddiate in cluster i
    """
    def __init__(self, data_source: Sized, batch_size: int) -> None:
        self.bin_counts = np.bincount(data_source.clusters)
        start = [0] + self.bin_counts.cumsum()[:-1].tolist()
        end = self.bin_counts.cumsum()
        indices = data_source.clusters.argsort()
        self.indices = {i: indices[start[i]:end[i]] for i in range(len(self.bin_counts))}
        self._num_samples = len(data_source)
        self.batch_size = batch_size

    def __len__(self):
        return self._num_samples

    def __iter__(self) -> Iterator[List[int]]:
        bin_counts = self.bin_counts.copy()
        indices = self.indices.copy()
        # Until all the clusters has elements but all are less than BS
        while np.any(bin_counts >= self.batch_size):

            # select one cluster to sample. The cluster which has enoguth elements first.
            idx_bin_candidates = np.argwhere(bin_counts >= self.batch_size).flatten()
            idx_bin = random.choices(idx_bin_candidates, k=1)[0]

            # random shuffle the indices in that cluster and draw
            np.random.shuffle(indices[idx_bin])
            batch = indices[idx_bin][:self.batch_size]
            indices[idx_bin] = indices[idx_bin][self.batch_size:]
            bin_counts[idx_bin] -= len(batch)
            yield from iter(batch)

        # handle the rest of them (so some of elements in batch are from diff cluster)
        remaining_indices = [i for bin_list in list(indices.values()) for i in bin_list]
        while len(remaining_indices) > 0:
            batch = remaining_indices[:self.batch_size]
            remaining_indices = remaining_indices[self.batch_size:]
            yield from iter(batch)

class HierachicalSampler(BatchSampler):

    # so the batch will first aggregate BS * M, then distribute them with BS
    def __init__(
        self, 
        documents=None,
        encoder=None,
        batch_size=32,
        minibatch_size=320,
        doc_embeddings=None,
    ) -> None:
        """
        minibatch_size: the batchsize B multiple by M, 
            aiming to increasr the size of the representation space.
        n_clusters: 
            the number of clusters, which should be similar to M.
        """
        ## encoding arguments
        self.documents = documents
        self.encoder = encoder

        ## data arguments
        self.batch_size = batch_size 
        self.minibatch_size = minibatch_size 

        ## clustering
        self.orders = torch.randperm(len(data_source))
        self.kmeans = FaissKMeans(
                n_clusters=n_clusters or (minibatch_size // batch_size),
                device=device,
                **cluster_args
        )

        ## batch for distribution
        self.minibatch = []

    def _reorder_by_cluster(self, indices):

        # encode selected documents if needed
        if self.doc_embeddings is None:
            start = min(indices)
            documents = self.documents[start: start+len(indices)]

            for batch_docs in tqdm(batch_iteratoro(documents, 64)):
                pass

        # encode selected documents if needed
        self.kmeans.fit(embeddings)
        labels = self.kmeans.assign(embeddings).flatten()
        return indices[labels.argsort()]

    def __iter__(self) -> Iterator[List[int]]:

        indices_minibatch = []
        for i in range(len(self.data_source) // self.batch_size + 1):

            # preparing the batch of inputs
            while len(indices_minibatch) < self.minibatch_size:
                start = i * self.batch_size
                indices_minibatch += self.orders[start:(start+self.batch_size)]
                
            yield indices_minibatch
            # reorder via cluster results. empty the list
            indices_minibatch = []

            # [TODO] add the reorder for hard span extraction
            # if reached. calculate clusters and return the new orders
            # start = j * self.minibatch_size
            # self.orders[start: start+self.minibatch] = \
            #         self._reorder_by_cluster(inices_minibatch)

