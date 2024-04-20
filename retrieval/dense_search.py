import os
import sys
import json
import argparse
from tqdm import tqdm 

from pyserini.search import FaissSearcher

from searcher import FaissMaxSearcher
from utils import load_topic, batch_iterator

from encoders import ContrieverQueryEncoder, GTEQueryEncoder

encoder_class_map = {
    "contriever": ContrieverQueryEncoder,
    "gte": GTEQueryEncoder
}

def search(args):

    query_encoder = encoder_class_map[args.encoder_class](
        args.encoder_path, 
        device=args.device,
        pooling=args.pooling if args.use_multiple_vectors is False else 'none',
        l2_norm=args.l2_norm,
        use_span_embedding=True if args.use_span_embedding else False
    )

    if args.use_multiple_vectors:
        searcher = FaissMaxSearcher(args.index, query_encoder)
    else:
        searcher = FaissSearcher(args.index, query_encoder)

    topics = load_topic(args.topic)
    qids = list(topics.keys())
    qtexts = list(topics.values())
    output = open(args.output, 'w')

    for (start, end) in tqdm(
            batch_iterator(range(0, len(qids)), args.batch_size, True),
            total=(len(qids)//args.batch_size)+1
    ):
        qids_batch = qids[start: end]
        qtexts_batch = qtexts[start: end]
        hits = searcher.batch_search(
                queries=qtexts_batch, 
                q_ids=qids_batch, 
                threads=10,
                k=args.k,
        )

        for key, value in hits.items():
            for i in range(len(hits[key])):
                output.write(
                        f'{key} Q0 {hits[key][i].docid:4} {i+1} {hits[key][i].score:.5f} faiss\n'
                )

    output.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=1000, type=int)
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--encoder_path", default=None, type=str)
    parser.add_argument("--encoder_class", default='contriever', type=str)
    parser.add_argument("--topic", default=None, type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--device", default='cpu', type=str)
    # additiona model setup
    parser.add_argument("--pooling", default='cls', type=None)
    parser.add_argument("--l2_norm", default=False, action='store_true')
    parser.add_argument("--use_span_embedding", action='store_true', default=False)
    parser.add_argument("--use_multiple_vectors", action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)
    search(args)
