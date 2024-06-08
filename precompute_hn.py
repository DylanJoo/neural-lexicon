import os
import torch
import argparse
import logging

from transformers import AutoTokenizer

from src.options import DataOptions
from src.sampling.data import DatasetIndependentCropping
from src.sampling.utils import batch_iterator
from src.sampling.encoders import BERTEncoder

from src.sampling.miner import NegativeSpanMiner

logger = logging.getLogger(__name__)

def main(args):

    ## [setup] initialized encoders/tokenizers
    encoder = BERTEncoder(args.encoder_name_or_path, device=args.device, pooling='mean')
    tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path or args.encoder_name_or_path
    )
    tokenizer.bos_token = '[CLS]'
    tokenizer.eos_token = '[SEP]'

    ### [prepare] arguments
    data_opt = DataOptions(
            corpus_jsonl=args.corpus_jsonl,
            corpus_spans_jsonl=args.corpus_spans_jsonl,
            prebuilt_negative_jsonl=args.negative_jsonl,
            prebuilt_faiss_dir=args.faiss_dir,
            chunk_length=256,
            min_chunk_length=32,
            select_span_mode=None,
            preprocessing='replicate'
    )
    dataset = DatasetIndependentCropping(data_opt, tokenizer)

    ## [negative mining]
    logger.info('mining start')
    negative_writer = open(args.negative_jsonl, 'w') 
    negative_miner = NegativeSpanMiner(data_opt, dataset, tokenizer)
    negative_miner.precompute_prior_negatives(
            encoder=encoder,
            batch_size=64,
            n_samples=10,
            top_k=100,
            negative_writer=negative_writer
    )
    negative_writer.close()
    logger.info('mining done')

    ## [quick testing]
    data_opt.prebuilt_negative_jsonl = args.prebuilt_negative_jsonl
    negative_miner = NegativeSpanMiner(data_opt, dataset, tokenizer)

    print('total number\n', len(dataset))
    print('checking\n', dataset[0])
    print(tokenizer.decode(dataset[0]['q_tokens'].long()))
    print(tokenizer.decode(dataset[0]['span_tokens'].long()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name_or_path", default='contriever', type=str)
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str)
    parser.add_argument("--corpus_jsonl", default=None, type=str, required=True)
    parser.add_argument("--corpus_spans_jsonl", default=None, type=str, required=False)
    parser.add_argument("--faiss_dir", default=None, type=str)
    parser.add_argument("--negative_jsonl", default=None, type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--device", default='cpu', type=str)
    args = parser.parse_args()

    print(f'precomuting preliminary data for {args.corpus_jsonl}.')

    main(args)

    print('done')
