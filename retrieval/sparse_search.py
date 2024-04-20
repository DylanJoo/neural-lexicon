import os
import collections
from tqdm import tqdm 
import json
import argparse
from pyserini.search.lucene import LuceneImpactSearcher
from utils import load_topic, batch_iterator

from encoders import SpladeQueryEncoder

def search(args):

    query_encoder = SpladeQueryEncoder(args.encoder, device=args.device)
    query_encoder.model.eval()
    searcher = LuceneImpactSearcher(args.index, query_encoder, args.min_idf)
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
		qids=qids_batch, 
		threads=4,
		k=args.k
	)
	for key, value in hits.items():
	    for i in range(len(hits[key])):
		output.write(
			f'{key} Q0 {hits[key][i].docid:4} {i+1} {hits[key][i].score:.5f} SPLADE\n'
		)

    output.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=1000, type=int)
    parser.add_argument("--min_idf",type=float, default=0)
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--topic", default=None, type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--device", default='cpu', type=str)
    args = parser.parse_args()

    search(args)
    print("Done")
