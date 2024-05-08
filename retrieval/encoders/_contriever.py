# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Optional

import faiss
import torch
from pyserini.encode import DocumentEncoder
from pyserini.search.faiss import QueryEncoder
from transformers import BertModel, AutoTokenizer
from sklearn.preprocessing import normalize
import sys
import torch.nn as nn

class BertModelWithOutput(BertModel):
    def __init__(self, config, add_pooling_layer=False, **kwargs):
        super().__init__(config, add_pooling_layer=add_pooling_layer, **kwargs)

        if not hasattr(config, "span_pooling"):
            self.config.span_pooling = None

        if self.config.span_pooling == "span_select_average":
            self.outputs = nn.Sequential(nn.Linear(self.config.hidden_size, 1), nn.Sigmoid())
        elif self.config.span_pooling == "span_select_weird":
            self.outputs = nn.Sequential(nn.Linear(self.config.hidden_size, 1), nn.Softmax(1))
        else:
            self.outputs = nn.Identity()

    @staticmethod
    def _mean_pooling(token_embeddings, attention_mask):
        last_hidden = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, input_ids, attention_mask, pooling, **kwargs):
        model_output = super().forward(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True, 
                **kwargs
        )
        last_hidden_states = model_output["last_hidden_state"]

        if pooling == "mean":
            embeddings = self._mean_pooling(last_hidden_states, attention_mask).detach().cpu().numpy()
        elif pooling == 'cls':
            embeddings = last_hidden_states[:, 0, :].detach().cpu().numpy()
        elif pooling == "mean_exclude_cls":
            embeddings = self._mean_pooling(last_hidden_states[:, 1:, :], attention_mask[:, 1:]).detach().cpu().numpy()
        else:
            embeddings = last_hidden_states.detach().cpu().numpy()

        if self.config.span_pooling is not None:
            if 'average' in self.config.span_pooling:
                select_prob = self.outputs(last_hidden_states[:, 1:])
                span_embeddings = torch.mean(last_hidden_states[:, 1:, :] * select_prob, dim=1).detach().cpu().numpy()
            elif 'weird' in self.config.span_pooling:
                select_prob = self.outputs(last_hidden_states[:, 1:])
                span_embeddings = torch.mean(last_hidden_states[:, 1:, :] * select_prob, dim=1).detach().cpu().numpy()
            else:
                span_embeddings = None
            return embeddings, span_embeddings
        else:
            return embeddings, None

class ContrieverDocumentEncoder(DocumentEncoder):
    def __init__(self, model_name, tokenizer_name=None, device='cuda', pooling='mean', l2_norm=False, use_span_embedding=False):
        self.device = device
        self.model = BertModelWithOutput.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or 'facebook/contriever')
        self.has_model = True
        self.pooling = pooling
        self.l2_norm = l2_norm
        self.use_span_embedding = use_span_embedding

    def encode(self, texts=None, titles=None, max_length=256, **kwargs):
        if titles is not None: 
            texts = [f'{title} {text}'.strip() for title, text in zip(titles, texts)]

        inputs = self.tokenizer(
            texts,
            max_length=max_length,
            padding='longest',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        ).to(self.device) 
        outputs = self.model(**inputs, pooling=self.pooling)

        embeddings = outputs[1] if self.use_span_embedding else outputs[0]

        if self.l2_norm:
            faiss.normalize_L2(embeddings)

        return embeddings

class ContrieverQueryEncoder(QueryEncoder):
    def __init__(self, model_name, tokenizer_name=None, device='cpu', pooling='mean', l2_norm=False, use_span_embedding=False):
        self.device = device
        self.model = BertModelWithOutput.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or 'facebook/contriever')
        self.pooling = pooling
        self.l2_norm = l2_norm
        self.use_span_embedding = use_span_embedding

    def encode(self, query: str, **kwargs):

        inputs = self.tokenizer(
            [query],
            max_length=64,
            padding='longest',
            truncation='only_first',
            add_special_tokens=True,
            return_tensors='pt'
        ).to(self.device)
        outputs = self.model(**inputs, pooling=self.pooling)

        embeddings = outputs[1] if self.use_span_embedding else outputs[0]

        if self.l2_norm:
            faiss.normalize_L2(embeddings)

        return embeddings.flatten()

    def batch_encode(self, texts, **kwargs):
        inputs = self.tokenizer(
            texts,
            max_length=64,
            padding='longest',
            truncation='only_first',
            add_special_tokens=True,
            return_tensors='pt'
        ).to(self.device)
        outputs = self.model(**inputs, pooling=self.pooling)

        embeddings = outputs[1] if self.use_span_embedding else outputs[0]

        if self.l2_norm:
            faiss.normalize_L2(embeddings)

        return embeddings
