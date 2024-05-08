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
    def __init__(
        self, 
        opt, 
        retriever, 
        tokenizer, 
        miner=None, 
        label_smoothing=False
    ):
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

        ## additional projection layer
        # hidden_size = retriever.config.hidden_size
        # self.non_linear = nn.Sequential(nn.Linear(2*hidden_size, hidden_size), nn.Tanh())

        ## negative miner
        self.miner = miner

    def forward(
        self, 
        q_tokens, q_mask, 
        c_tokens, c_mask, 
        span_tokens=None, span_mask=None, 
        data_index=None,
        **kwargs
    ):

        # [todo] see if therea anything different between dynamic and static negative vectors 
        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        loss = 0.0

        ## [encoding] query/context contrastive from random cropping 
        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask)[0]
        cemb = self.encoder(input_ids=c_tokens, attention_mask=c_mask)[0]
        if span_tokens is not None and span_mask is not None:
            spemb = self.encoder(
                    input_ids=span_tokens,
                    attention_mask=span_mask, 
                    pooling=self.opt.span_pooling
            )[0]

        ### [mining] negative samples
        #### from precomputed embeddings for sents
        #### (1) the doc embeddings are from real docs
        if (self.miner is not None) and (self.use_doc_by_doc):
            neg_vectors = self.crop_depedent_from_docs
                    embeds_1=qemb.detach().clone().cpu(), 
                    embeds_2=cemb.detach().clone().cpu(),
                    indices=data_index,
                    n=1, k0=10, k=100,
            ).to(self.encoder.device)

        #### (2) the doc embeddings are from proxy docs
        if (self.miner is not None) and (self.use_doc_by_spans):
            neg_vectors = self.miner(
                    embeds_1=spemb.detach().clone().cpu(), 
                    indices=data_index,
                    n=1, k0=10, k=100,
            ).to(self.encoder.device)

        ### [normalize]
        if self.norm_query:
            qemb = F.normalize(qemb, p=2, dim=-1)
        if self.norm_doc:
            cemb = F.normalize(cemb, p=2, dim=-1)
        if self.norm_spans and span_tokens is not None:
            spemb = F.normalize(spemb, p=2, dim=-1)

        ### [contrastive learning]
        CELoss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        KLLoss = nn.KLDivLoss(reduction='batchmean')
        MSELoss = nn.MSELoss()

        #### (a) sp compute
        loss = 0
        if (self.miner is not None) and self.miner.use_doc_by_doc:
            scores_q = torch.einsum("id, jd->ij", 
                    qemb / self.tau, torch.cat([cemb, neg_vectors], dim=0)
            )
            scores_c = torch.einsum("id, jd->ij", 
                    cemb / self.tau, torch.cat([qemb, neg_vectors], dim=0)
            )
        else:
            scores_q = torch.einsum("id, jd->ij", qemb / self.tau, cemb)
            scores_c = torch.einsum("id, jd->ij", cemb / self.tau, qemb)

        logs = {}
        loss = (CELoss(scores_q, labels) + CELoss(scores_c, labels)) /2
        predicted_idx = torch.argmax(scores_q, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        logs.update({'loss_sent': loss, 'acc_sent': accuracy})

        #### (b) spst compute
        loss_sp = 0.0
        if (self.miner is not None) and (self.miner.use_doc_by_spans):
            scores_qsp = torch.einsum("id, jd->ij", 
                    spemb, torch.cat([qemb, neg_vectors], dim=0) / self.tau_span
            )
            scores_csp = torch.einsum("id, jd->ij", 
                    spemb, torch.cat([cemb, neg_vectors], dim=0) / self.tau_span
            )
        else:
            ## [NOTE] not sure if there are difference between the multiplication order
            # scores_qsp = torch.einsum("id, jd->ij", qemb / self.tau_span, spemb)
            # scores_csp = torch.einsum("id, jd->ij", cemb / self.tau_span, spemb)
            scores_qsp = torch.einsum("id, jd->ij", spemb, qemb / self.tau_span)
            scores_csp = torch.einsum("id, jd->ij", spemb, cemb / self.tau_span)

        loss_sp = (CELoss(scores_qsp, labels) + CELoss(scores_csp, labels)) / 2
        predicted_idx = torch.argmax(scores_qsp, dim=-1)
        accuracy_sp = 100 * (predicted_idx == labels).float().mean()
        logs.update({'loss_span': loss_sp, 'acc_span': accuracy_sp})

            #### (c) stst regularization
            ## query-span-score vs. context-span-score distribution
            ### more like regularization
            target = F.softmax(scores_qsp, dim=1)
            logits_spans = F.log_softmax(scores_csp, dim=1)
            loss_distil = KLLoss(logits_spans, target)
            logs.update({'loss_span_distil': loss_distil})

        loss_sp = 0.0
        ## query-span/context-span contrastive from doc-derived spans
        if span_tokens is not None and span_mask is not None:
            spemb = self.encoder(
                    input_ids=span_tokens,
                    attention_mask=span_mask, 
                    pooling=self.opt.span_pooling
            )[0]

            if self.norm_spans:
                spemb = F.normalize(spemb, p=2, dim=-1)

            # span-conditioanl embeddings
            # span embeddings
            scores_qsp = torch.einsum("id, jd->ij", qemb / self.tau_span, spemb)
            scores_csp = torch.einsum("id, jd->ij", cemb / self.tau_span, spemb)
            loss_sp = (CELoss(scores_qsp, labels) + CELoss(scores_csp, labels)) / 2

            predicted_idx = torch.argmax(scores_qsp, dim=-1)
            accuracy_sp = 100 * (predicted_idx == labels).float().mean()
            logs.update({'loss_span': loss_sp, 'acc_span': accuracy_sp})

            ## query-span-score vs. context-span-score distribution
            ### more like regularization
            target = F.softmax(scores_qsp, dim=1)
            logits_spans = F.log_softmax(scores_csp, dim=1)
            loss_distil = KLLoss(logits_spans, target)
            logs.update({'loss_span_distil': loss_distil})

        logs.update(self.encoder.additional_log)
        loss = loss * self.opt.alpha + loss_sp * self.opt.beta + loss_distil * self.opt.gamma
        return InBatchOutput(loss=loss, acc=accuracy, logs=logs)

    def get_encoder(self):
        return self.encoder

    # def forward_bidirectional(self, tokens, mask, **kwargs):
    # becoming the default setup as it seems to be better

class InBatchInteractionWithSpan(InBatchInteraction):

    def forward(self, q_tokens, q_mask, c_tokens, c_mask, span_tokens, span_mask, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        # [objectives] 
        CELoss = nn.CrossEntropyLoss()
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
