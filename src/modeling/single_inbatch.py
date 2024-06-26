import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers.modeling_outputs import BaseModelOutput
from src.modeling import dist_utils 

@dataclass
class InBatchOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    acc: Optional[Tuple[torch.FloatTensor, ...]] = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None
    qemb: Optional[torch.FloatTensor] = None
    cemb: Optional[torch.FloatTensor] = None
    spemb: Optional[torch.FloatTensor] = None

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

        # distributed 
        self.is_ddp = dist.is_initialized()

        # learning hyperparameter
        self.bidirectional = False
        self.tau = opt.temperature
        self.tau_span = opt.temperature_span

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
        """
        [todo] see if therea anything different between dynamic and static negative vectors 
        """
        loss = 0.0
        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask)[0]
        cemb = self.encoder(input_ids=c_tokens, attention_mask=c_mask)[0]

        if self.is_ddp:
            gather_fn = dist_utils.gather
            qemb = gather_fn(qemb)
            cemb = gather_fn(cemb)
            data_index = gather_fn(data_index).detach().cpu().numpy()

        bsz = qemb.size(0) # query-centric batch 
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        ## [st-st]
        ### negative mining using precomputed embeddings
        if self.miner is not None:
            neg_inputs = self.miner.crop_depedent_from_docs(
                    embeds_1=qemb.clone().detach().cpu(), 
                    embeds_2=cemb.clone().detach().cpu(),
                    indices=data_index,
                    n=1, k0=0, k=100, 
                    exclude_overlap=True,
                    to_return='span',
                    debug=self.opt.my_debug
            )
            neg_vectors = self.encoder(
                    input_ids=neg_inputs[0].to(self.encoder.device),
                    attention_mask=neg_inputs[1].to(self.encoder.device)
            )[0]
        else:
            neg_vectors = None

        if self.norm_query:
            qemb = F.normalize(qemb, p=2, dim=-1)
        if self.norm_doc:
            cemb = F.normalize(cemb, p=2, dim=-1)

        if neg_vectors is not None:
            scores_q = torch.einsum("id, jd->ij", 
                    qemb / self.tau, torch.cat([cemb, neg_vectors], dim=0))
            scores_c = torch.einsum("id, jd->ij", 
                    cemb / self.tau, torch.cat([qemb, neg_vectors], dim=0))
        else:
            scores_q = torch.einsum("id, jd->ij", qemb / self.tau, cemb)
            scores_c = torch.einsum("id, jd->ij", cemb / self.tau, qemb)

        ## computing losses
        CELoss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        KLLoss = nn.KLDivLoss(reduction='batchmean')
        MSELoss = nn.MSELoss()

        logs = {}
        loss = (CELoss(scores_q, labels) + CELoss(scores_c, labels)) /2

        predicted_idx = torch.argmax(scores_q, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        logs.update({'loss_sent': loss, 'acc_sent': accuracy})

        loss_sp = 0.0
        loss_distil = 0.0

        ## [st-sp]
        ### query-span & context-span contrastive 
        if span_tokens is not None and span_mask is not None:
            spemb = self.encoder(
                    input_ids=span_tokens,
                    attention_mask=span_mask, 
                    pooling=self.opt.span_pooling
            )[0]
            if self.is_ddp:
                gather_fn = dist_utils.gather
                spemb = gather_fn(spemb)

            if self.miner is not None: 
                # [todo] for stsp objective, see what's the better doc mining strategy
                if self.opt.mine_neg_using == 'span':
                    neg_inputs = self.miner.span_depedent_from_docs(
                            embeds=spemb.detach().clone().cpu(), 
                            indices=data_index,
                            n=1, k0=0, k=100,
                    )
                    neg_vectors = self.encoder(
                            input_ids=neg_inputs[0].to(self.encoder.device),
                            attention_mask=neg_inputs[1].to(self.encoder.device)
                    )[0]
            else:
                neg_vectors = None

            if self.norm_spans:
                spemb = F.normalize(spemb, p=2, dim=-1)

            if neg_vectors is not None:
                scores_qsp = torch.einsum("id, jd->ij", 
                        qemb / self.tau, torch.cat([spemb, neg_vectors], dim=0))
                scores_csp = torch.einsum("id, jd->ij", 
                        cemb / self.tau, torch.cat([spemb, neg_vectors], dim=0))
            else:
                scores_qsp = torch.einsum("id, jd->ij", qemb / self.tau_span, spemb)
                scores_csp = torch.einsum("id, jd->ij", cemb / self.tau_span, spemb)

            loss_sp = (CELoss(scores_qsp, labels) + CELoss(scores_csp, labels)) / 2

            predicted_idx = torch.argmax(scores_qsp, dim=-1)
            accuracy_sp = 100 * (predicted_idx == labels).float().mean()
            logs.update({'loss_span': loss_sp, 'acc_span': accuracy_sp})

            ## [sp-sp]
            target = F.softmax(scores_qsp, dim=1)
            logits_spans = F.log_softmax(scores_csp, dim=1)
            loss_distil = KLLoss(logits_spans, target)
            logs.update({'loss_span_distil': loss_distil})

        logs.update(self.encoder.additional_log)
        if self.miner is not None:
            logs.update(self.miner.additional_log)

        loss = loss * self.opt.alpha + loss_sp * self.opt.beta + loss_distil * self.opt.gamma
        return InBatchOutput(loss=loss, acc=accuracy, logs=logs, qemb=qemb, cemb=cemb, spemb=spemb)

    def get_encoder(self):
        return self.encoder

    # def forward_bidirectional(self, tokens, mask, **kwargs):
    # becoming the default setup as it seems to be better

# class InBatchInteractionWithSpan(InBatchInteraction):
#
#     def forward(self, q_tokens, q_mask, c_tokens, c_mask, span_tokens, span_mask, **kwargs):
#
#         bsz = len(q_tokens)
#         labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)
#
#         # [objectives] 
#         CELoss = nn.CrossEntropyLoss()
#         KLLoss = nn.KLDivLoss(reduction='batchmean')
#         MSELoss = nn.MSELoss()
#
#         # add the dataclass for outputs
#         qemb, qsemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask)
#         cemb, csemb = self.encoder(input_ids=c_tokens, attention_mask=c_mask)
#         semb, ssemb = self.encoder(input_ids=span_tokens, attention_mask=span_mask)
#
#         if self.norm_query:
#             qemb = F.normalize(qemb, p=2, dim=-1)
#             qsemb = F.normalize(qsemb, p=2, dim=-1)
#         if self.norm_doc:
#             cemb = F.normalize(cemb, p=2, dim=-1)
#             csemb = F.normalize(csemb, p=2, dim=-1)
#
#         logs = {}
#         # [sentence]
#         scores = torch.einsum("id, jd->ij", qemb / self.tau, cemb)
#         loss_0 = CELoss(scores, labels)
#         predicted_idx = torch.argmax(scores, dim=-1)
#         accuracy = 100 * (predicted_idx == labels).float().mean()
#         logs.update({'loss_sent': loss_0, 'acc_sent': accuracy})
#
#         # [span-sent interaction]
#         ## add loss of (q-span, doc) ## add loss of (query, d-span)
#         scores_spans = torch.einsum("id, jd->ij", qsemb / self.tau_span, cemb)
#
#         if self.opt.span_span_interaction:
#             scores_spans2 = torch.einsum("id, jd->ij", csemb / self.tau_span, qsemb)
#             loss_span = CELoss(scores_spans, labels) + CELoss(scores_spans2, labels) 
#             logs.update({'acc_span2': accuracy_span2})
#         else:
#             loss_span = CELoss(scores_spans, labels) 
#
#         predicted_idx = torch.argmax(scores_spans, dim=-1) # check only one as it's prbbly similar
#         accuracy_span = 100 * (predicted_idx == labels).float().mean()
#
#         logs.update({'loss_span': loss_span, 'acc_span': accuracy_span})
#         logs.update(self.encoder.additional_log)
#
#         loss = loss_0 * self.opt.alpha + loss_span * self.opt.beta
#         return InBatchOutput(
# 	        loss=loss, 
# 	        acc=accuracy,
#                 logs=logs,
# 	)
