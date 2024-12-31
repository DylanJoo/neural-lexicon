import torch
from .data_utils import build_mask 
from collections import defaultdict

class Collator(object):
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, batch_examples):

        batch = defaultdict(list)
        for example in batch_examples:
            for k, v in example.items():
                batch[k].append(v)

        q_tokens, q_mask = build_mask(batch["q_tokens"])
        batch["q_tokens"] = q_tokens
        batch["q_mask"] = q_mask

        c_tokens, c_mask = build_mask(batch["c_tokens"])
        batch["c_tokens"] = c_tokens
        batch["c_mask"] = c_mask

        batch['data_index'] = torch.Tensor(batch['data_index']) 

        # derived from document
        if "span_tokens" in batch:
            span_tokens, span_mask = build_mask(batch["span_tokens"])
            batch["span_tokens"] = span_tokens
            batch["span_mask"] = span_mask

        # the original document
        if "d_tokens" in batch:
            d_tokens, d_mask = build_mask(batch["d_tokens"])
            batch["d_tokens"] = d_tokens
            batch["d_mask"] = d_mask

        return batch

