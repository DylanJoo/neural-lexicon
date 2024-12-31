#!/bin/sh
#SBATCH --job-name=SR
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# Start the experiment.
index_dir=${HOME}/indexes/beir
data_dir=${HOME}/datasets/beir

for dataset in trec-covid nfcorpus fiqa arguana webis-touche2020 quora scidocs scifact nq hotpotqa dbpedia-entity fever climate-fever;do
    # Indexing
    # python -m pyserini.index.lucene \
    #     --collection BeirMultifieldCollection \
    #     --input ${data_dir}/${dataset}/collection \
    #     --index ${index_dir}/${dataset}/bm25-multifield.lucene \
    #     --generator DefaultLuceneDocumentGenerator \
    #     --threads 32 \
    #     --fields title 

    # Search
    # python retrieval/bm25_search.py \
    #     --k 100 --k1 0.9 --b 0.4 \
    #     --index ${index_dir}/${dataset}/bm25-multifield.lucene \
    #     --topic ${data_dir}/${dataset}/queries.jsonl \
    #     --batch_size 32 \
    #     --fields contents=1.0 title=1.0 \
    #     --output runs/bm25/run.beir.${dataset}.bm25-multifield.txt

    # Eval
    echo -ne "beir-${dataset}  | bm25 | ${dataset} | reproduced | "
    ~/trec_eval-9.0.7/trec_eval -c -m ndcg_cut.10 -m recall.100 \
        ${data_dir}/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt \
        runs/bm25/run.beir.${dataset}.bm25-multifield.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done
