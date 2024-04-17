import torch
from span import *
from encoders import BERTEncoder
from transformers import AutoTokenizer

model_name='sentence-transformers/all-MiniLM-L6-v2'
# model_name='thenlper/gte-base'
model_name='facebook/contriever-msmarco'
encoder = BERTEncoder(model_name)
documents = torch.load('../corpus.jsonl.pkl')
tokenizer = AutoTokenizer.from_pretrained(model_name)

spans = add_extracted_spans(documents[:2], encoder, ngram_range=(2,3), top_k_spans=5)
for i, span_scores in enumerate(spans):
    print(tokenizer.decode(documents[i]))
    print()
    for span, score in span_scores:
        print(tokenizer.decode(span), score)
    print()

