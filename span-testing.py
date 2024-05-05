import torch
from src.sampling.data import load_dataset
from src.sampling.encoders import BERTEncoder
from transformers import AutoTokenizer
from src.options import DataOptions

# baseline
model_name='facebook/contriever'

# the finetuned 
model_name='models/ckpt/contriever-baseline/scidocs/checkpoint-2000'

# strong baseline
# model_name='thenlper/gte-base'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.bos_token = '[CLS]'
tokenizer.eos_token = '[SEP]'
encoder = BERTEncoder(model_name, device='cpu')

# for dataset_name in ['scifact', 'scidocs', 'trec-covid']:
for dataset_name in ['scifact']:
    # setup for ind-cropping
    data_opt = DataOptions(
            train_data_dir=f'/home/dju/datasets/beir/{dataset_name}/dw-ind-cropping', 
            chunk_length=256,
            loading_mode='from_scratch',
            precompute_with_spans=True
    )
    dataset = load_dataset(data_opt, tokenizer)

    dataset.documents = dataset.documents[:5]

    doc_embeddings = dataset.init_spans(
            encoder,
            batch_size=2,
            max_doc_length=384,
            ngram_range=(2,3),
            top_k_spans=10,
            return_doc_embeddings=True,
            doc_embeddings_by_spans=False
    )

    print('\ndoc')
    print(tokenizer.decode(dataset.documents[0], add_special_tokens=False)[:100])
    print('\nspans')
    for i, spans in enumerate(dataset.spans[0:3]):
        print(i, 'document and the corresponding span')
        for span in spans[:5]:
            tokens, score = span
            print(tokenizer.decode(tokens, add_special_tokens=False), score)
        print('\n')
