from dataset import load_dataset
from ind_cropping.options import DataOptions

model_name='thenlper/gte-base'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.bos_token = '[CLS]'
tokenizer.eos_token = '[SEP]'

from encoders import BERTEncoder
encoder = BERTEncoder(model_name, device='cuda')

data_opt = DataOptions(train_data_dir='/home/dju/neural-lexicon/parsed/scifact')
dataset = load_dataset(data_opt, tokenizer)
dataset.get_spans(encoder)

for i, d in enumerate(dataset):
    print(tokenizer.decode(d['q_tokens'].long()))
    print(tokenizer.decode(d['c_tokens'].long()))
    print(tokenizer.decode(d['span_tokens'].long()))
    if i > 2:
        break

dataset.save(f'/home/dju/datasets/test_collection/scifact/train.pt')
print('done')
