import os
import torch
from src.sampling.encoders import BERTEncoder
from transformers import AutoTokenizer

from src.options import DataOptions
from src.sampling.data import load_dataset
from src.sampling.index_utils import NegativeSpanMiner
from src.sampling.utils import batch_iterator

import argparse

def calculate_spans_and_clusters(args, dataset_name):

    # initialized encoders/tokenizers
    encoder = BERTEncoder(args.encoder_name_or_path, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path or args.encoder_name_or_path)
    tokenizer.bos_token = '[CLS]'
    tokenizer.eos_token = '[SEP]'

    data_opt = DataOptions(
            train_data_dir=f'/home/dju/datasets/beir/{dataset_name}/dw-ind-cropping', 
            chunk_length=256,
            loading_mode='from_scratch',
            preprocessing='replicate'
    )
    dataset = load_dataset(data_opt, tokenizer)
    # dataset.documents = dataset.documents[:10] # shrink for debugging

    ## [span extraction]
    K=args.num_spans
    print('span extraction start')
    doc_embeddings = dataset.init_spans(
            encoder,
            batch_size=args.batch_size,
            max_doc_length=384,
            ngram_range=(args.min_ngrams, args.max_ngrams),
            top_k_spans=10,
    )
    print('span extraction done')

    ## [save and load (testing)]
    path = os.path.join(f'/home/dju/datasets/beir/{dataset_name}/dw-ind-cropping', 
                        args.saved_file_format.format(K))
    dataset.save(path)

    ## [quick testing]
    data_opt.loading_mode = args.loading_mode
    dataset = load_dataset(data_opt, tokenizer)

    print('checking span\n', dataset.spans[0])

    return doc_embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name_or_path", default='contriever', type=str)
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str)
    parser.add_argument("--min_ngrams", default=2, type=int)
    parser.add_argument("--max_ngrams", default=3, type=int)
    parser.add_argument("--num_spans", default=10, type=int)
    parser.add_argument("--num_clusters", default=0.05, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--saved_file_format", default='doc.span.{}.cluster.{}.pt', type=str, required=True)
    parser.add_argument("--loading_mode", default=None, type=str, required=True)
    parser.add_argument("--device", default='cpu', type=str)
    # faiss index
    parser.add_argument("--faiss_output", default=None, type=str)
    parser.add_argument("--doc_embeddings_by_spans", default=False, action='store_true')
    # dataset
    args = parser.parse_args()

    # for dataset_name in ['trec-covid']:
    for dataset_name in ['scifact', 'scidocs', 'trec-covid']:
        print(dataset_name, 'spans and cluster precomputing')

        doc_embeddings = calculate_spans_and_clusters(args, dataset_name)
        if args.faiss_output is not None:
            NegativeSpanMiner.save_index(
                    embed_vectors=doc_embeddings,
                    index_dir=os.path.join('/home/dju/indexes/temp', args.faiss_output + dataset_name)
            )
            print('indexing done\n')
        else:
            print('NO indexing\n')
