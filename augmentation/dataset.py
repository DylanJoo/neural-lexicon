"""
Baseline: 
    X. Independant cropping: positive sampling. length-wise, dataset-wise
    1. Document-wise independent cropping: positive sampling, document-wise
    2. Document-wise independent cropping: plus in-document negative samples
"""
import random
from data_utils import *

class DocwiseIndCropping(torch.utils.data.Dataset):

    def __init__(self, documents, chunk_length, tokenizer, opt, ngram_range=None):
        self.documents = [d for d in documents if len(d) >= chunk_length]
        self.chunk_length = chunk_length
        self.tokenizer = tokenizer
        self.opt = opt
        self.opt.mask_id = tokenizer.mask_token_id 
        self.ngram_range = ngram_range
        self.spans = None

    def __len__(self):
        return len(documents)

    def _select_random_spans(self, index):
        candidates, scores = list(zip(*self.spans[index]))
        span = random.choices(candidates, weights=scores, k=1)
        return [self.tokenizer.bos_token_id] + span + [self.tokenizer.eos_token_id]

    def __getitem__(self, index):
        docucment = self.documents[index] # the crop is belong to one doc
        start_idx = random.randint(0, len(document) - chunk_length - 1)
        end_idx = start_idx + self.chunk_length 
        tokens = self.data[start_idx:end_idx] 

        # fine the closest anchor of this span
        q_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        c_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        q_tokens = apply_augmentation(q_tokens, self.opt)
        c_tokens = apply_augmentation(c_tokens, self.opt)
        q_tokens = add_bos_eos(q_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)
        c_tokens = add_bos_eos(c_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)

        if self.spans is not None:
            span = self._select_random_spans(index)
            return {"q_tokens": q_tokens, "c_tokens": c_tokens, "span": span}
        else:
            return {"q_tokens": q_tokens, "c_tokens": c_tokens}

