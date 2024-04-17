import faiss
import torch
import torch.nn as nn
from transformers import BertModel

class BERTEncoder:
    def __init__(self, model_name, device='cpu', pooling='mean'):
        self.device = device
        self.model = BertEmbeddings.from_pretrained(model_name)
        self.model.to(self.device)
        self.pooling = pooling

    def set_model(self, model_obj):
        self.model = model_obj.eval()

    def encode(self, input_ids, attention_mask=None, **kwargs):
        embeddings = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pooling=self.pooling,
                **kwargs
        )
        return embeddings

class BertEmbeddings(BertModel):
    def __init__(self, config, add_pooling_layer=False, **kwargs):
        super().__init__(config, add_pooling_layer=add_pooling_layer, **kwargs)

    @staticmethod
    def _mean_pooling(token_embeddings, attention_mask):
        if attention_mask is not None:
            last_hidden = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        else:
            return last_hidden.mean(dim=1)

    def forward(self, input_ids, attention_mask, pooling, **kwargs):
        model_output = super().forward(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True, 
                **kwargs
        )
        last_hidden_states = model_output["last_hidden_state"]
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if pooling == 'cls':
            emb = last_hidden[:, 0]
        elif pooling == 'mean':
            emb = self._mean_pooling(last_hidden, attention_mask)

        return emb


