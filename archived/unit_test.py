import torch
from transformers import AutoTokenizer
from src.options import DataOptions, ModelOptions

data_opt = DataOptions(
        corpus_jsonl=f'/home/dju/datasets/beir/scifact/collection_tokenized/corpus_tokenized.jsonl', 
        corpus_spans_jsonl=f'/home/dju/datasets/beir/scifact/collection_tokenized/spans_tokenized.jsonl', 
        chunk_length=256,
        min_chunk_length=32,
        select_span_mode='weighted'
)
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
tokenizer.bos_token = '[CLS]'
tokenizer.eos_token = '[SEP]'

from src.sampling.encoders import BERTEncoder
encoder = BERTEncoder("facebook/contriever")

from src.sampling.data import DatasetIndependentCropping
dataset = DatasetIndependentCropping(data_opt, tokenizer)

from src.sampling.neg_miner import NegativeSpanMiner 
miner = NegativeSpanMiner(
        dataset=dataset, 
        tokenizer=tokenizer, 
        index_dir='/home/dju/indexes/beir-neg/scifact'
)

with torch.no_grad():
    # miner.prior_negative_mining(encoder, top_k=100)
    inputs = tokenizer(['algorithm'], return_tensors='pt')
    embed = encoder.encode(inputs['input_ids']).mean(axis=0).unsqueeze(0)
    S, I = miner.index.search(embed, 10)
    print(I)
    for i in I[0]:
        print(tokenizer.decode(miner.dataset[int(i)]['q_tokens'].long()))

    # negs = miner.crop_depedent_from_docs(
    #         encoder.encode(dataset[0]['q_tokens'].long()[None, :]),
    #         encoder.encode(dataset[0]['c_tokens'].long()[None, :]),
    #         indices=[0],
    #         n=1, k0=0, k=100,
    #         exclude_overlap=True,
    #         to_return='crop'
    # )

    # for idx in range(10):
    #     print(f'\n{idx}\n')
    #     print(tokenizer.decode(dataset[idx]['span_tokens'].long()))
    #     negatives = miner.negatives[idx]
    #     for nid in negatives:
    #         negative_inputs = dataset[int(nid)]['q_tokens'].long()
    #         print(tokenizer.decode(negative_inputs))
