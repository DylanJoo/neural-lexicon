from encoders import ContrieverDocumentEncoder, ContrieverQueryEncoder
from encoders._contriever import BertModelWithOutput
from transformers import AutoTokenizer

class GTEDocumentEncoder(ContrieverDocumentEncoder):
    def __init__(self, model_name, tokenizer_name=None, device='cuda', pooling='mean', l2_norm=False, use_span_embedding=False):
        self.device = device
        self.model = BertModelWithOutput.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or 'thenlper/gte-base')
        self.has_model = True
        self.pooling = pooling
        self.l2_norm = l2_norm
        self.use_span_embedding = use_span_embedding

class GTEQueryEncoder(ContrieverQueryEncoder):
    def __init__(self, model_name, tokenizer_name=None, device='cpu', pooling='mean', l2_norm=False, use_span_embedding=False):
        self.device = device
        self.model = BertModelWithOutput.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or 'thenlper/gte-base')
        self.pooling = pooling
        self.l2_norm = l2_norm
        self.use_span_embedding = use_span_embedding

