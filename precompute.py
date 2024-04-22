import os
import torch
from src.sampling.encoders import BERTEncoder
from transformers import AutoTokenizer

from src.options import DataOptions
from src.sampling.data import load_dataset

import argparse

from pyserini.encode import FaissRepresentationWriter
import faiss

TEMP_DIR = '/home/dju/indexes/temp'

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def build_faiss_index(args, name, embed_ids, embed_vectors):
    # build 
    index_dir = os.path.join(TEMP_DIR, args.faiss_output + name)
    dimension = embed_vectors.shape[-1]
    embedding_writer = FaissRepresentationWriter(index_dir, dimension)

    with embedding_writer:
        for s, e in batch_iterator(embed_ids, 64, True):
            embed_id = embed_ids[s:e]
            embed_vector = embed_vectors[s:e]
            embedding_writer.write({"id": embed_ids, "vector": embed_vector})

    # testing 
    index_path = os.path.join(index_dir, 'index')
    docid_path = os.path.join(index_dir, 'docid')
    index = faiss.read_index(index_path)

    # searching
    emb_q = embed_vectors[0:1]
    distances, indexes, vectors = index.search_and_reconstruct(emb_q, 10)
    vectors = vectors[0]
    distances = distances.flat
    indexes = indexes.flat
    print(
            [(embed_ids[idx], score) for score, idx, vector in zip(distances, indexes, vectors) if idx != -1]
    )
    return 0

def calculate_spans_and_clusters(args, dataset_name):

    data_opt = DataOptions(
            train_data_dir=f'/home/dju/datasets/temp/{dataset_name}', 
            chunk_length=256,
            loading_mode='from_scratch'
    )
    dataset = load_dataset(data_opt, tokenizer)
    dataset.documents = dataset.documents

    ## [span extraction]
    K=args.num_spans
    doc_embeddings = dataset.get_update_spans(
            encoder,
            batch_size=args.batch_size,
            max_doc_length=384,
            ngram_range=(2,3),
            top_k_spans=10,
            return_doc_embeddings=True,
            doc_embeddings_by_spans=True
    )
    print(doc_embeddings.shape)

    ## [clustering]
    N=args.num_clusters
    N_used = dataset.get_update_clusters(
            doc_embeddings, 
            n_clusters=N,
            min_points_per_centroid=32,
            device=args.device
    )

    ## [save and load (testing)]
    path = os.path.join(f'/home/dju/datasets/temp/{dataset_name}', 
                        args.saved_file_format.format(K, N_used))
    dataset.save(path)

    ## [quick testing]
    data_opt.loading_mode=args.loading_mode
    dataset = load_dataset(data_opt, tokenizer)

    print('span\n', dataset.spans[0])
    print('cluster\n', dataset.clusters[0])
    print('done')

    return doc_embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name_or_path", default='contriever', type=str)
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str)
    parser.add_argument("--num_spans", default=10, type=int)
    parser.add_argument("--num_clusters", default=0.05, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--saved_file_format", default='doc.span.{}.cluster.{}.pt', type=str, required=True)
    parser.add_argument("--loading_mode", default=None, type=str, required=True)
    parser.add_argument("--device", default='cpu', type=str)
    # faiss index
    parser.add_argument("--faiss_output", default=None, type=str)
    args = parser.parse_args()

    encoder = BERTEncoder(args.encoder_name_or_path, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path or args.encoder_name_or_path)
    tokenizer.bos_token = '[CLS]'
    tokenizer.eos_token = '[SEP]'

    # for dataset_name in ['scifact', 'scidocs', 'trec-covid']:
    for dataset_name in ['scifact']:
        doc_embeddings = calculate_spans_and_clusters(args, dataset_name)

        if args.faiss_output:
            build_faiss_index(args, dataset_name, list(range(len(doc_embeddings))), doc_embeddings)
