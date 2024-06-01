import os
import glob
import sys
import json
import logging
import random
import datetime
import torch
import faiss
from tqdm import tqdm
from datasets import Dataset

from .data_utils import *
from .span_utils import *
from .utils import batch_iterator, cosine_top_k

logger = logging.getLogger(__name__)

class DatasetIndependentCropping(torch.utils.data.Dataset):

    def __init__(self, opt, tokenizer):
        super().__init__()
        self.opt = opt

        self.corpus_jsonl = opt.corpus_jsonl
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.opt.mask_token_id = tokenizer.mask_token_id # this is made for augmentation

        ## attrs
        self.chunk_length = opt.chunk_length
        self.min_chunk_length = opt.min_chunk_length

        ## preprocessing the raw corpus
        self.corpus = self._load_corpus()

        ## span extraction
        self.select_span_mode = opt.select_span_mode
        ### if the span has been precompute
        if opt.select_span_mode in ['top1', 'weighted', 'random']:
            self.spans = self._load_spans()

        ## Independent croping
        self.span_online_update = opt.span_online_update

    def _load_corpus(self, corpus_jsonl=None):

        file = (corpus_jsonl or self.opt.corpus_jsonl)
        self.corpus_jsonl = file

        to_return = []
        with open(file, 'r') as f:
            for line in f:
                doc = json.loads(line.strip())['input_ids']
                to_return.append( doc )

        # collect the normal length and duplicate the shorter (minimum)
        to_return_1 = [d for d in to_return if len(d) > self.chunk_length]
        to_return_0 = [d for d in to_return if len(d) <= self.chunk_length]
        n_replicate = (self.chunk_length // self.min_chunk_length)
        to_return_0 = [(d*n_replicate)[:self.chunk_length] for d in to_return_0 if len(d) > self.min_chunk_length]
        return to_return_1 + to_return_0

    def _load_spans(self, spans_jsonl=None):
        file = (spans_jsonl or self.opt.corpus_spans_jsonl)
        self.spans_jsonl = file

        to_return = []
        with open(file, 'r') as f:
            for line in f:
                spans = json.loads(line.strip())['spans']
                to_return.append( spans )
        return to_return

    def update_spans(self, data_indices, batch_d_tokens, batch_d_masks, batch_token_embeds, batch_doc_embeds):
        for i, data_index in enumerate(data_indices):
            token_embeds = batch_token_embeds[i]
            doc_embed = batch_doc_embeds[i]

            # [NOTE] so far, online update is for 'contextalized' span
            candidates, candidate_embeds = compute_span_embeds(
                    inputs=batch_d_tokens[i],
                    mask=batch_d_masks[i],
                    ngram_range=(10, 10), 
                    stride=5,
                    token_embeds=token_embeds,
                    span_pooling='mean',
            )
            topk_spans = cosine_top_k(doc_embed, candidates, candidate_embeds)
            self.spans[data_index] = topk_spans

    def init_spans_and_index(
        self,
        encoder, 
        batch_size=64, 
        max_doc_length=384, 
        ngram_range=(2,3), 
        stride=1,
        top_k=5,
        decontextualized=True,
        spans_writer=None,
        index_writer=None
    ):
        """
        Params
        ------
        encoder
        batch_size 
        max_doc_length
        ngram_range
        top_k

        """
        with torch.no_grad():

            for batch_docs in tqdm(
                    batch_iterator(self.corpus, batch_size), \
                    total=len(self.corpus)//batch_size+1
            ):
                # prepare document input with bos/eos
                batch_tokens = [torch.Tensor(
                    [self.bos_token_id]+d[:(max_doc_length-2)]+[self.eos_token_id]
                ) for d in batch_docs]

                batch_tokens, mask = build_mask(batch_tokens)
                batch_tokens, mask = batch_tokens.to(encoder.device), mask.to(encoder.device)
                outputs = encoder.encode(batch_tokens, mask, return_token_embeddings=True)

                batch_tokens = batch_tokens.detach().cpu().numpy()
                batch_doc_embeds = outputs[0].detach().cpu().numpy()
                batch_tokens_embeds = outputs[1].detach().cpu().numpy()
                mask = mask.detach().cpu().numpy()

                ## [build ngram candidates]
                if decontextualized:
                ### Assume spans is context-independent
                    span_embeds, doc2spans, span2tokens = batch_compute_span_embeds(
                            encoder=encoder,
                            documents=batch_docs,
                            bos=self.bos_token_id, eos=self.eos_token_id,
                            batch_size=batch_size * 2, # as it's ngram, we can cram more in one batch
                            ngram_range=ngram_range,
                            stride=stride
                    )
                    for i, doc_embed in enumerate(batch_doc_embeds):
                        candidate_indices = doc2spans[i].nonzero()[1]
                        candidates = [span2tokens[j] for j in candidate_indices]
                        candidate_embeds = span_embeds[candidate_indices]

                        topk_spans = cosine_top_k(doc_embed, candidates, candidate_embeds, top_k)
                        spans_writer.write(json.dumps({"spans": topk_spans})+'\n')
                else:
                ### Assum spans is context-dependent
                    for i, tokens in enumerate(batch_tokens):
                        doc_embed = batch_doc_embeds[i] # mean 
                        token_embeds = batch_tokens_embeds[i]

                        candidates, candidate_embeds = compute_span_embeds(
                                tokens, mask[i], ngram_range, stride, token_embeds, 'mean'
                        )

                        topk_spans = cosine_top_k(doc_embed, candidates, candidate_embeds, top_k)
                        spans_writer.write(json.dumps({"spans": topk_spans})+'\n')

                ## [encode document embeddings]
                if index_writer is not None:
                    n, d = batch_doc_embeds.shape
                    index_writer.write({"id": [str(0)] * n, "vector": batch_doc_embeds})


    def __getitem__(self, index):
        document = self.corpus[index]
        span_tokens = self._select_spans(index)

        # check the token length of the document
        offset =  len(document) - self.chunk_length
        if offset > 0:
            start_idx = random.randint(0, offset)
            end_idx = start_idx + self.chunk_length 
            tokens = document[start_idx:end_idx] 
        else:
            tokens = document

        # select crops
        bos, eos = self.bos_token_id, self.eos_token_id
        q_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        q_tokens = apply_augmentation(q_tokens, self.opt, span_tokens)
        q_tokens = add_bos_eos(q_tokens, bos, eos)

        c_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        c_tokens = apply_augmentation(c_tokens, self.opt, span_tokens)
        c_tokens = add_bos_eos(c_tokens, bos, eos)

        if span_tokens is not None:
            span_tokens = add_bos_eos(span_tokens, bos, eos)
            return {"q_tokens": q_tokens, 
                    "c_tokens": c_tokens, 
                    "span_tokens": span_tokens,
                    "data_index": index}
        else:
            return {"q_tokens": q_tokens, 
                    "c_tokens": c_tokens, 
                    "data_index": index}

        # add entire doc
        if self.span_online_update:
            d_tokens = add_bos_eos(document[:384], bos, eos)
            outputs.update({"d_tokens": d_tokens})

        return outputs

    def __len__(self):
        return len(self.corpus)

    def _select_spans(self, index):
        if self.select_span_mode is None:
            return None
        if isinstance(index, int) is False:
            return None

        candidates, scores = list(zip(*self.spans[index]))
        if self.select_span_mode == 'weighted':
            span_tokens = random.choices(candidates, weights=scores, k=1)[0] # sample by the cosine
        elif self.select_span_mode == 'top1':
            span_tokens = candidates[0]
        elif self.select_span_mode == 'random':
            span_tokens = random.choices(candidates, k=1)[0] 
        return span_tokens

