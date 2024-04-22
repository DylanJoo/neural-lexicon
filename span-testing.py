import torch
from src.sampling.data import load_dataset
from src.sampling.encoders import BERTEncoder
from transformers import AutoTokenizer
from src.options import DataOptions

# baseline
# model_name='facebook/contriever'

# the finetuned 
# model_name='models/ckpt/contriever-dev.baseline/scidocs'

# strong baseline
# model_name='thenlper/gte-base'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.bos_token = '[CLS]'
tokenizer.eos_token = '[SEP]'
encoder = BERTEncoder(model_name, device='cpu')

# for dataset_name in ['scifact', 'scidocs', 'trec-covid']:
for dataset_name in ['scidocs']:
    # setup for ind-cropping
    data_opt = DataOptions(
            train_data_dir=f'/home/dju/datasets/temp/{dataset_name}', 
            chunk_length=256,
            loading_mode='from_scratch'
    )
    dataset = load_dataset(data_opt, tokenizer)

    dataset.documents = dataset.documents[:5]
    dataset.get_update_spans(
            encoder,
            batch_size=128,
            max_doc_length=384,
            ngram_range=(2,3),
            top_k_spans=10
    )

    print('\ndoc')
    print(tokenizer.decode(dataset.documents[0], add_special_tokens=False)[:100])
    print('\nspans')
    for span in dataset.spans[0]:
        print(tokenizer.decode(span[0], add_special_tokens=False), span[1])
