import os
import torch
import argparse
import faiss
from transformers import AutoTokenizer

from src.sampling.encoders import BERTEncoder
from src.options import DataOptions
from src.sampling.data_utils import build_mask
from src.sampling.data import load_dataset
from src.sampling.index_utils import NegativeSpanMiner

def main(args, name):
    # searching
    data_opt = DataOptions(
            train_data_dir=f'/home/dju/datasets/beir/{dataset_name}/dw-ind-cropping', 
            chunk_length=256,
            loading_mode='doc2spans'
    )
    dataset = load_dataset(data_opt, tokenizer)
    dataset.documents = dataset.documents[:3]

    for tokens, score in dataset.spans[0]:
        print(score, tokenizer.decode(tokens))

    # [negative mining]
    # index_dir = f'/home/dju/indexes/temp/{args.prefix}_{name}'
    # miner = NegativeSpanMiner(
    #         spans=dataset.spans[:10], 
    #         clusters=dataset.clusters[:10], 
    #         index_dir=index_dir
    # )
    # outputs[0]: distance
    # outputs[1]: indices
    # outputs[2]: vectors
    # input_ids, mask = build_mask(torch.tensor([dataset.documents[0]]))
    # embeddings_1 = encoder.encode(input_ids=input_ids, attention_mask=mask)
    # testing = torch.cat([embeddings_1, embeddings_1], dim=0)
    # testing = testing.detach().cpu()
    # miner.crop_depedent_from_docs_v1(testing, testing, [0])

    ## [span extraction]
    from src.sampling.span_utils import add_extracted_spans
    # from src.sampling.span_utils_dev import add_extracted_spans

    outputs = add_extracted_spans(
            encoder=encoder,
            documents=dataset.documents,
            batch_size=10,
            max_doc_length=512,
            ngram_range=(2,3),
            top_k_spans=10,
            bos_id=tokenizer.bos_token_id,
            eos_id=tokenizer.eos_token_id,
            return_doc_embeddings=False
    )
    # print(spans[0])
    for tokens, score in outputs[0][0]:
        print(score, tokenizer.decode(tokens))

    ## [clustering]
    ## [save and load (testing)]
    ## [quick testing]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name_or_path", default='thenlper/gte-base', type=str)
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--prefix", default='doc_emb', type=str)
    args = parser.parse_args()

    #
    encoder = BERTEncoder(
            args.encoder_name_or_path, device=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path or args.encoder_name_or_path
    )
    tokenizer.bos_token = '[CLS]'
    tokenizer.eos_token = '[SEP]'

    for dataset_name in ['scifact']:
        main(args, dataset_name)
