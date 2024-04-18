import torch
from cluster_utils import add_extracted_spans
from encoders import BERTEncoder
from transformers import AutoTokenizer


device = 'cuda'
device = 'cpu'

dataset_name ='scidocs'
model_name ='facebook/contriever'
encoder = BERTEncoder(model_name, device=device)
documents = torch.load(f'/home/dju/datasets/temp/{dataset_name}/corpus.jsonl.pkl')
tokenizer = AutoTokenizer.from_pretrained(model_name)

spans, all_doc_embeddings = add_extracted_spans(
    documents=documents[:3],
    encoder=encoder, 
    batch_size=2,
    max_doc_length=5,
    ngram_range=(2,3), 
    top_k_spans=2,
    bos_id=101,
    eos_id=102,
    return_doc_embeddings=True
)

# for i, span_scores in enumerate(spans):
#     print(tokenizer.decode(documents[i]))
#     print()
#     for span, score in span_scores:
#         print(tokenizer.decode(span), score)
#     print()
print(all_doc_embeddings.shape)

from cluster_utils import FaissKMeans

kmeans = FaissKMeans(n_clusters=3, device=device)
kmeans.fit(all_doc_embeddings)
