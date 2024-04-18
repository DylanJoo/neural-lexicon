import torch
from src.sampling.encoders import BERTEncoder
from transformers import AutoTokenizer
from src.options import DataOptions

from src.sampling.data import load_dataset

device='cuda'
model_name='facebook/contriever'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.bos_token = '[CLS]'
tokenizer.eos_token = '[SEP]'
encoder = BERTEncoder(model_name, device=device)

for dataset_name in ['scifact', 'scidocs', 'trec-covid']:
    # setup for ind-cropping
    data_opt = DataOptions(
            train_data_dir=f'/home/dju/datasets/temp/{dataset_name}', 
            chunk_length=256,
            loading_mode='from_scratch'
    )
    dataset = load_dataset(data_opt, tokenizer)

    ## [span extraction]
    # this can be updated anytime
    doc_embeddings = dataset.get_update_spans(
            encoder,
            batch_size=128,
            max_doc_length=384,
            ngram_range=(2,3),
            top_k_spans=10,
            return_doc_embeddings=True
    )

    ## [clustering]
    dataset.get_update_clusters(
            doc_embeddings, 
            n_clusters=3, 
            device=device
    )

    ## [save and load (testing)]
    dataset.save(f'/home/dju/datasets/temp/{dataset_name}/doc.span.10.clusuter.3.pt')
    data_opt.loading_mode='from_precomputed'
    dataset = load_dataset(data_opt, tokenizer)

    print('span\n', dataset.spans[0:3])
    print('cluster\n', dataset.clusters[0:3])
    print('done')
