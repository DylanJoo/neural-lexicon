import torch
from src.sampling.data import load_dataset
from src.sampling.encoders import BERTEncoder
from transformers import AutoTokenizer
from src.options import DataOptions

model_name='facebook/contriever'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.bos_token = '[CLS]'
tokenizer.eos_token = '[SEP]'
encoder = BERTEncoder(model_name, device='cuda')

# for dataset_name in ['scifact', 'scidocs', 'trec-covid']:
for dataset_name in ['scidocs']:
    # setup for ind-cropping
    data_opt = DataOptions(
            train_data_dir=f'/home/dju/datasets/temp/{dataset_name}', 
            chunk_length=256,
            loading_mode='from_scratch'
    )
    dataset = load_dataset(data_opt, tokenizer)

    # this can be updated anytime
    dataset.get_update_spans(
            encoder,
            batch_size=128,
            max_doc_length=384,
            ngram_range=(2,3),
            top_k_spans=10
    )

    dataset.save(f'/home/dju/datasets/temp/{dataset_name}/doc.with.spans.pt')
    dataset_copy = torch.load(open(f'/home/dju/datasets/temp/{dataset_name}/doc.10.spans.pt', 'rb'))
    print(dataset_copy[0])
    print('done')
