import torch
from span import *
from encoders import BERTEncoder
from transformers import AutoTokenizer


encoder = BERTEncoder('facebook/contriever')
documents = torch.load('../corpus.jsonl.pkl')
tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')

spans = add_extracted_spans(documents[:2], encoder, ngram_range=(3,10))
for i, (span, score) in enumerate(spans):
    print('content', tokenizer.decode(documents[i][:100]))
    print(tokenizer.decode(span))

