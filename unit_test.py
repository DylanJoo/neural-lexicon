from src.sampling.data import load_dataset
import os
import torch
from src.sampling.encoders import BERTEncoder
from transformers import AutoTokenizer

from src.options import DataOptions
from src.sampling.data import load_dataset

import argparse

import faiss

def main(args, name):
    # searching

    data_opt = DataOptions(
            train_data_dir=f'/home/dju/datasets/temp/{dataset_name}', 
            chunk_length=256,
            loading_mode='from_precomputed'
    )
    dataset = load_dataset(data_opt, tokenizer)
    dataset.documents = dataset.documents[:10]

    doc_embeddings = dataset.get_update_spans(
            encoder, 
            batch_size=64, 
            max_doc_length=256, 
            ngram_range=(2,3), 
            top_k_spans=5,
            return_doc_embeddings=True,
            doc_embeddings_by_spans=True
    )

    # outputs[0]: distance
    # outputs[1]: indices
    # outputs[2]: vectors

    # print(tokenizer.decode(dataset.spans[0][0][0]), dataset.clusters[0])
    # for j in outputs[1][0]:
    #     print(tokenizer.decode(dataset.spans[j][0][0]), dataset.clusters[j])

    ## [span extraction]
    ## [clustering]
    ## [save and load (testing)]
    ## [quick testing]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name_or_path", default='facebook/contriever', type=str)
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str)
    parser.add_argument("--device", default='cpu', type=str)
    # parser.add_argument("--num_spans", default=10, type=int)
    # parser.add_argument("--num_clusters", default=0.05, type=float)
    # parser.add_argument("--batch_size", default=128, type=int)
    # parser.add_argument("--saved_file_format", default='doc.span.{}.cluster.{}.pt', type=str, required=True)
    # parser.add_argument("--loading_mode", default=None, type=str, required=True)
    # faiss index
    # parser.add_argument("--faiss_output", default=None, type=str)
    args = parser.parse_args()

    encoder = BERTEncoder(args.encoder_name_or_path, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path or args.encoder_name_or_path
    )
    tokenizer.bos_token = '[CLS]'
    tokenizer.eos_token = '[SEP]'

    for dataset_name in ['scifact', 'scidocs']:
    # for dataset_name in ['scifact']:
        main(args, dataset_name)
