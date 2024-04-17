"""
Baseline: 
    X. Independant cropping: positive sampling. length-wise, dataset-wise
    1. Document-wise independent cropping: positive sampling, document-wise
    2. Document-wise independent cropping: plus in-document negative samples
"""
import glob
import logging
import random
import torch
from data_utils import *

from span import add_extracted_spans

logger = logging.getLogger(__name__)

def load_dataset(opt, tokenizer):
    datasets = {}
    # [TODO] check the other loading mode, maybe put the pre-extraced here.
    if opt.loading_mode == "full": 
        files = glob.glob(os.path.join(opt.train_data_dir, "*.pkl"))
    elif opt.loading_mode == "full_with_spans": 
        files = glob.glob(os.path.join(opt.train_data_dir, "*.pt"))

    assert len(files) == 1, 'more than one files'
    list_of_token_ids = torch.load(files[0], map_location="cpu")
    return DocwiseIndCropping(list_of_token_ids, opt.chunk_length, tokenizer, opt)

class DocwiseIndCropping(torch.utils.data.Dataset):

    def __init__(self, documents, chunk_length, tokenizer, opt, ngram_range=None):
        self.documents = [d for d in documents if len(d) >= chunk_length][:10]
        self.chunk_length = chunk_length
        self.tokenizer = tokenizer
        self.opt = opt
        self.opt.mask_id = tokenizer.mask_token_id 

        # span arguments
        self.spans = None
        self.ngram_range = ngram_range

    def get_update_spans(self, encoder):
        self.spans = add_extracted_spans(
                documents=self.documents,
                encoder=encoder,
                batch_size=64,
                max_doc_length=256,
                ngram_range=(2,3),
                top_k_spans=5,
                bos_id=self.tokenizer.bos_token_id,
                eos_id=self.tokenizer.eos_token_id
        )

    def __len__(self):
        return len(documents)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as fout:
            torch.save(self, fout)
        print('saved') 

    def _select_random_spans(self, index):
        candidates, scores = list(zip(*self.spans[index]))
        span_tokens = random.choices(candidates, weights=scores, k=1)[0]
        span_tokens = add_bos_eos(span_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)
        return span_tokens

    def __getitem__(self, index):
        document = self.documents[index] # the crop is belong to one doc
        start_idx = random.randint(0, len(document) - self.chunk_length - 1)
        end_idx = start_idx + self.chunk_length 
        tokens = document[start_idx:end_idx] 

        # fine the closest anchor of this span
        q_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        c_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        q_tokens = apply_augmentation(q_tokens, self.opt)
        c_tokens = apply_augmentation(c_tokens, self.opt)
        q_tokens = add_bos_eos(q_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)
        c_tokens = add_bos_eos(c_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)

        if self.spans is not None:
            span_tokens = self._select_random_spans(index)
            return {"q_tokens": q_tokens, "c_tokens": c_tokens, "span_tokens": span_tokens}
        else:
            return {"q_tokens": q_tokens, "c_tokens": c_tokens}

