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
            train_data_dir=f'/home/dju/datasets/temp/{dataset_name}', 
            chunk_length=256,
            loading_mode='from_precomputed'
    )
    dataset = load_dataset(data_opt, tokenizer)

    index_dir = f'/home/dju/indexes/temp/{args.prefix}_{name}'
    miner = NegativeSpanMiner(
            spans=dataset.spans[:10], 
            clusters=dataset.clusters[:10], 
            index_dir=index_dir
    )
    # outputs[0]: distance
    # outputs[1]: indices
    # outputs[2]: vectors
    input_ids, mask = build_mask(torch.tensor([dataset.documents[0]]))
    embeddings_1 = encoder.encode(input_ids=input_ids, attention_mask=mask)
    testing = torch.cat([embeddings_1, embeddings_1], dim=0)
    testing = testing.detach().cpu()
    miner.crop_depedent_from_docs_v1(testing, testing, [0])

    ## [span extraction]
    ## [clustering]
    ## [save and load (testing)]
    ## [quick testing]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name_or_path", default='facebook/contriever', type=str)
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

    for dataset_name in ['scifact', 'scidocs']:
    # for dataset_name in ['scifact']:
        main(args, dataset_name)
