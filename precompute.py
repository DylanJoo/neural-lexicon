import os
import torch
import argparse
import logging

from transformers import AutoTokenizer

from src.options import DataOptions
from src.sampling.data import DatasetIndependentCropping
from src.sampling.utils import batch_iterator
from src.sampling.miner import NegativeSpanMiner
from src.sampling.encoders import BERTEncoder

from pyserini.encode import FaissRepresentationWriter

logger = logging.getLogger(__name__)

def main(args):

    ## [setup] initialized encoders/tokenizers
    encoder = BERTEncoder(args.encoder_name_or_path, device=args.device, pooling='mean')
    tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path or args.encoder_name_or_path
    )
    tokenizer.bos_token = '[CLS]'
    tokenizer.eos_token = '[SEP]'

    ### prepare arguments
    data_opt = DataOptions(
            corpus_jsonl=args.corpus_jsonl,
            corpus_spans_jsonl=args.corpus_spans_jsonl,
            chunk_length=256,
            min_chunk_length=32,
            select_span_mode=None,
            preprocessing='replicate'
    )
    dataset = DatasetIndependentCropping(data_opt, tokenizer)

    ## [span extraction]
    logger.info('span extraction start')
    spans_writer = open(args.corpus_spans_jsonl, 'w') 
    if args.faiss_index_dir:
        index_writer = FaissRepresentationWriter(args.faiss_index_dir, encoder.model.config.hidden_size)
        with index_writer:
            dataset.init_spans_and_index(
                    encoder,
                    batch_size=args.batch_size,
                    max_doc_length=384,
                    ngram_range=(args.min_ngrams, args.max_ngrams),
                    stride=args.stride,
                    top_k=args.num_spans,
                    decontextualized=args.decontextualized,
                    spans_writer=spans_writer,
                    index_writer=index_writer
            )
    else:
        dataset.init_spans_and_index(
                encoder,
                batch_size=args.batch_size,
                max_doc_length=384,
                ngram_range=(args.min_ngrams, args.max_ngrams),
                stride=args.stride,
                top_k=args.num_spans,
                decontextualized=args.decontextualized,
                spans_writer=spans_writer,
                index_writer=None
        )
    spans_writer.close()
    logger.info('span extraction done')

    ## [quick testing]
    data_opt.select_span_mode = 'weighted'
    dataset = DatasetIndependentCropping(data_opt, tokenizer)
    print('total number\n', len(dataset))
    print('checking\n', dataset[0])
    print(tokenizer.decode(dataset[0]['q_tokens'].long()))
    print(tokenizer.decode(dataset[0]['span_tokens'].long()))
    print(tokenizer.decode(dataset[0]['span_tokens'].long()))
    print(tokenizer.decode(dataset[0]['span_tokens'].long()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name_or_path", default='contriever', type=str)
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str)
    parser.add_argument("--corpus_jsonl", default=None, type=str, required=True)
    parser.add_argument("--corpus_spans_jsonl", default=None, type=str, required=True)
    parser.add_argument("--faiss_index_dir", default=None, type=str)
    parser.add_argument("--decontextualized", default=False, action='store_true')
    parser.add_argument("--min_ngrams", default=2, type=int)
    parser.add_argument("--max_ngrams", default=3, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--num_spans", default=10, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--device", default='cpu', type=str)
    args = parser.parse_args()

    print(f'precomuting preliminary data for {args.corpus_jsonl}.')

    main(args)

    print('done')
