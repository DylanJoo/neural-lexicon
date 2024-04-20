import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from transformers.modeling_outputs import BaseModelOutput

@dataclass
class InBatchOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    acc: Optional[Tuple[torch.FloatTensor, ...]] = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None

class InBatchInteraction(nn.Module):
    def __init__(self, opt, retriever, tokenizer, label_smoothing=False,):
        super().__init__()

        self.opt = opt
        self.encoder = retriever

        # representation
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.norm_spans = opt.norm_spans
        self.label_smoothing = label_smoothing
        self.tokenizer = tokenizer

        # learning hyperparameter
        self.bidirectional = False
        self.tau = opt.temperature
        self.tau_span = opt.temperature_span

    def forward(self, q_tokens, q_mask, c_tokens, c_mask, span_tokens=None, span_mask=None, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        loss = 0.0
        ## query/context contrastive from random cropping 
        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask)[0]
        cemb = self.encoder(input_ids=c_tokens, attention_mask=c_mask)[0]
        if self.norm_query:
            qemb = F.normalize(qemb, p=2, dim=-1)
        if self.norm_doc:
            cemb = F.normalize(cemb, p=2, dim=-1)

        logs = {}
        scores = torch.einsum("id, jd->ij", qemb / self.tau, cemb)
        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        logs.update({'loss_sent': loss, 'acc_sent': accuracy})

        loss_sp = 0.0
        ## query-span/context-span contrastive from doc-derived spans
        if span_tokens is not None and span_mask is not None:
            spemb = self.encoder(input_ids=span_tokens, attention_mask=span_mask)[0]
            if self.norm_spans:
                spemb = F.normalize(spemb, p=2, dim=-1)

            scores_qsp = torch.einsum("id, jd->ij", qemb / self.tau_span, spemb)
            scores_csp = torch.einsum("id, jd->ij", cemb / self.tau_span, spemb)
            loss_sp = F.cross_entropy(scores_qsp, labels, label_smoothing=self.label_smoothing) + \
                    F.cross_entropy(scores_csp, labels, label_smoothing=self.label_smoothing)

            predicted_idx = torch.argmax(scores_qsp, dim=-1)
            accuracy_sp = 100 * (predicted_idx == labels).float().mean()
            logs.update({'loss_span': loss_sp, 'acc_span': accuracy_sp})

        logs.update(self.encoder.additional_log)
        loss = loss * self.opt.alpha + loss_sp * self.opt.beta

        return InBatchOutput(loss=loss, acc=accuracy, logs=logs)

    def get_encoder(self):
        return self.encoder

    # def forward_bidirectional(self, tokens, mask, **kwargs):
    #     bsz = len(tokens)
    #     labels = torch.arange(bsz, dtype=torch.long, device=tokens.device).view(-1, 2).flip([1]).flatten().contiguous()
    #
    #     emb = self.encoder(input_ids=tokens, attention_mask=mask)
    #
    #     scores = torch.matmul(emb/self.tau, emb.transpose(0, 1))
    #     scores.fill_diagonal_(float('-inf'))
    #
    #     loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing) 
    #
    #     predicted_idx = torch.argmax(scores, dim=-1)
    #     accuracy = 100 * (predicted_idx == labels).float().mean()
    #
    #     return {'loss': loss, 'acc': accuracy}


class InBatchInteractionWithSpan(InBatchInteraction):

    def forward(self, q_tokens, q_mask, c_tokens, c_mask, span_tokens, span_mask, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        # [objectives] 
        CELoss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        KLLoss = nn.KLDivLoss(reduction='batchmean')
        MSELoss = nn.MSELoss()

        # add the dataclass for outputs
        qemb, qsemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask)
        cemb, csemb = self.encoder(input_ids=c_tokens, attention_mask=c_mask)
        semb, ssemb = self.encoder(input_ids=span_tokens, attention_mask=span_mask)

        if self.norm_query:
            qemb = F.normalize(qemb, p=2, dim=-1)
            qsemb = F.normalize(qsemb, p=2, dim=-1)
        if self.norm_doc:
            cemb = F.normalize(cemb, p=2, dim=-1)
            csemb = F.normalize(csemb, p=2, dim=-1)

        logs = {}
        # [sentence]
        scores = torch.einsum("id, jd->ij", qemb / self.tau, cemb)
        loss_0 = CELoss(scores, labels)
        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        logs.update({'loss_sent': loss_0, 'acc_sent': accuracy})

        # [span-sent interaction]
        ## add loss of (q-span, doc) ## add loss of (query, d-span)
        scores_spans = torch.einsum("id, jd->ij", qsemb / self.tau_span, cemb)

        if self.opt.span_span_interaction:
            scores_spans2 = torch.einsum("id, jd->ij", csemb / self.tau_span, qsemb)
            loss_span = CELoss(scores_spans, labels) + CELoss(scores_spans2, labels) 
            logs.update({'acc_span2': accuracy_span2})
        else:
            loss_span = CELoss(scores_spans, labels) 

        predicted_idx = torch.argmax(scores_spans, dim=-1) # check only one as it's prbbly similar
        accuracy_span = 100 * (predicted_idx == labels).float().mean()

        logs.update({'loss_span': loss_span, 'acc_span': accuracy_span})
        logs.update(self.encoder.additional_log)

        loss = loss_0 * self.opt.alpha + loss_span * self.opt.beta
        return InBatchOutput(
	        loss=loss, 
	        acc=accuracy,
                logs=logs,
	)