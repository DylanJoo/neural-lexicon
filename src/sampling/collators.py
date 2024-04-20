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
        c_tokens, c_mask = build_mask(batch["c_tokens"])

        batch["q_tokens"] = q_tokens
        batch["q_mask"] = q_mask
        batch["c_tokens"] = c_tokens
        batch["c_mask"] = c_mask

        # derived from document
        if "span_tokens" in batch:
            span_tokens, span_mask = build_mask(batch["span_tokens"])
            batch["span_tokens"] = span_tokens
            batch["span_mask"] = span_mask

        return batch

