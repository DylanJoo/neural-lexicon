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
index_dir=${HOME}/indexes/lotte
data_dir=${HOME}/datasets/lotte

# Setting of encoders
exp=bm25

# Eval on testing set and search type
split=test 
query_type=search

# Go
for dataset in lifestyle recreation science technology writing;do

    # Indexing
    # python -m pyserini.index.lucene \
    #     --collection JsonCollection \
    #     --input ${data_dir}/${dataset}/$split/collection \
    #     --index ${index_dir}/${dataset}/bm25.lucene \
    #     --generator DefaultLuceneDocumentGenerator \
    #     --threads 32

    # Search
    # python retrieval/bm25_search.py \
    #     --k 100 --k1 0.9 --b 0.4 \
    #     --index ${index_dir}/${dataset}/bm25.lucene \
    #     --topic ${data_dir}/${dataset}/$split/questions.$query_type.tsv \
    #     --batch_size 32 \
    #     --output runs/$exp/run.lotte.${dataset}.bm25.txt

    # Eval (trec)
    echo -ne "lotte-search | ${exp} | ${dataset} | reproduced | "
    ~/trec_eval-9.0.7/trec_eval -c -m ndcg_cut.10 -m recall.100 \
        ${data_dir}/${dataset}/$split/qrels.lotte-${dataset}-$split.$query_type.txt \
        runs/${exp}/run.lotte.${dataset}.bm25.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done

# Eval (S@5)
echo "[Lotte-${split}]"
python tools/evaluate_lotte_rankings.py \
        --k 5 --split $split \
        --data_dir ${data_dir} \
        --rankings_dir runs/bm25/
