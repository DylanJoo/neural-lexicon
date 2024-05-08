import os
import torch
import transformers
import torch.nn as nn
from transformers import BertModel

"""
- pooling: the typical sentence embedding. 
    options: ['cls', 'mean']
- span_pooling: the sub-sentence embedding. 
    options: ['dev']
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from transformers.modeling_outputs import BaseModelOutput

@dataclass
class ContrieverOutput(BaseModelOutput):
    emb: Optional[torch.FloatTensor] = None
    span_emb: Optional[torch.FloatTensor] = None
    last_hidden: Optional[torch.FloatTensor] = None

class Contriever(BertModel):
    def __init__(self, config, add_pooling_layer=False, pooling='mean', **kwargs):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.config.pooling = pooling
        self.additional_log = {}
        self.outputs = nn.Sequential(nn.Linear(self.config.hidden_size, 2)) # use gumbel function

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_multi_vectors=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True
        )

        last_hidden_states = model_output["last_hidden_state"]
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        # sentence representation
        if self.config.pooling == 'cls':
            if 'pooler_output' in model_output:
                emb = model_output['pooler_output']
            else:
                emb = last_hidden[:, 0]
        elif self.config.pooling == 'mean':
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        # if return_multi_vectors: only return single embeddings for sentence
        #     emb = last_hidden

        # sub-sentence representation (multiple vectors)
        bsz, max_len, hsz = last_hidden.size()
        kwargs = {'hidden': last_hidden, 'mask': attention_mask, 
                  'bsz': bsz, 'max_len': max_len, 'hsz': hsz,
                  'return_multi_vectors': return_multi_vectors}
        span_emb, span_mask = self._span_extract(**kwargs)

        return emb, span_emb

    def _span_extract(self, hidden, mask, bsz, max_len, return_multi_vectors, **kwargs):
        logits = self.outputs(hidden) # include CLS
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        start_probs = nn.functional.gumbel_softmax(start_logits, dim=1, hard=True)  
        end_probs = nn.functional.gumbel_softmax(end_logits, dim=1, hard=True)  

        start_probs_vec = start_probs.cumsum(-1)
        end_probs_vec = torch.flip(torch.flip(end_probs, [1]).cumsum(-1), [1])
        span_mask = start_probs_vec * end_probs_vec # B L

        avail = (span_mask * mask).sum(dim=1)
        self.additional_log['extract_ratio'] = (span_mask.sum(dim=1) / avail).mean()

        span_emb = hidden * span_mask[..., None] # B L H

        if return_multi_vectors:
            return span_emb, span_mask
        else:
            # take average with span
            span_emb = span_emb.sum(dim=1) / span_mask.sum(dim=1)[..., None]
            return span_emb, span_mask

