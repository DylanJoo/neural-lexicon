import os
import glob
import logging
import random
import datetime
import torch

from .data import DocwiseIndCropping
from .cluster_utils import FaissKMeans

logger = logging.getLogger(__name__)

def load_dataset(opt, tokenizer):
    if opt.loading_mode == "from_scratch": 
        files = glob.glob(os.path.join(opt.train_data_dir, "*corpus*pkl"))
        assert len(files) == 1, 'more than one files'
        list_of_token_ids = torch.load(files[0], map_location="cpu")
        return ClusteredIndCropping(list_of_token_ids, opt.chunk_length, tokenizer, opt)

    elif opt.loading_mode == "from_precomputed": 
        files = glob.glob(os.path.join(opt.train_data_dir, "*.pt"))
        assert len(files) == 1, 'more than one files'
        return torch.load(files[0], map_location="cpu")


class ClusteredIndCropping(DocwiseIndCropping):

    def __init__(self, documents, chunk_length, tokenizer, opt):
        self.documents = [d for d in documents if len(d) >= chunk_length][:10]
        self.chunk_length = chunk_length
        self.tokenizer = tokenizer
        self.opt = opt
        self.opt.mask_id = tokenizer.mask_token_id 

        # span attrs
        self.spans = None
        self.spans_msim = 0 # the larger the better

        # cluster attrs
        self.clusters = None
        self.clusters_sse = 999 # the small the better

    def get_update_spans(
        self,
        encoder, 
        batch_size=64, 
        max_doc_length=256, 
        ngram_range=(2,3), 
        top_k_spans=5,
        return_doc_embeddings=False
    ):
        outputs = add_extracted_spans(
                documents=self.documents,
                encoder=encoder,
                batch_size=batch_size,
                max_doc_length=max_doc_length,
                ngram_range=ngram_range,
                top_k_spans=top_k_spans,
                bos_id=self.tokenizer.bos_token_id,
                eos_id=self.tokenizer.eos_token_id,
                return_doc_embeddings=return_doc_embeddings
        )

        self.spans = outputs[0]
        return outputs[1] if return_doc_embeddings else 0

    # [TODO] some alternatives: minibatch
    def get_update_clusters(
        self,
        embeddings,
        n_clusters=0.05,
        device='cpu',
        **cluster_args
    ):
        """
        Should work with constructing span embeddings
        """
        # TAS-B exps on MARCO was set default to 5%
        if isinstance(n_clusters, float):
            n_clusters = int(self.__len__ * 0.05)

        # [TODO] see if need shrinking it
        if cluster_args.pop('max_docs', False):
            embeddings = random.sample()

        start = datetime.datetime.now()
        kmeans = FaissKMeans(
                n_clusters=n_clusters, 
                device=device,
                **cluster_args
        )

        kmeans.fit(embeddings)
        self.clusters = kmeans.assign(embeddings).flatten()
        self.clusters_sse = kmeans.inertia_

        end = datetime.datetime.now()
        time_taken = (end - start).total_seconds() * 1000
        logger.info("Latency of cluster ({} documents {} clusters): {:.2f}ms".format(embeddings.shape[0], n_clusters, time_taken))

        return 0
