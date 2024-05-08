#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=ret
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=%x.%j.debug

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# Start the experiment.
index_dir=${HOME}/indexes/beir
data_dir=${HOME}/datasets/beir

cd ${HOME}/neural-lexicon

# Go
backbone=contriever
exp=baseline
encoder=facebook/contriever-msmarco

for dataset in scifact scidocs trec-covid;do
    echo indexing...
    python3 retrieval/dense_index.py input \
        --corpus ${data_dir}/${dataset}/collection \
        --fields text title \
        --shard-id 0 \
        --shard-num 1 output \
        --embeddings ${index_dir}/${dataset}-${exp}.faiss \
        --to-faiss encoder \
        --encoder-class ${backbone} \
        --encoder ${encoder} \
        --pooling mean \
        --fields text title \
        --batch 32 \
        --max-length 256 \
        --device cuda

    echo searching...
    python retrieval/dense_search.py \
        --k 100  \
        --index ${index_dir}/${dataset}-${exp}.faiss \
        --encoder_path ${encoder} \
        --encoder_class ${backbone} \
        --topic ${data_dir}/${dataset}/queries.jsonl \
        --batch_size 64 \
        --pooling mean \
        --device cuda \
        --output runs/baseline-${exp}/run.beir.${dataset}.${backbone}.${encoder}.txt

    echo -ne "beir-${dataset}.${exp}.${dummy} | " 
    ~/trec_eval-9.0.7/trec_eval \
        -c -m ndcg_cut.10 -m recall.100 \
        ${data_dir}/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt \
        runs/baseline-${exp}/run.beir.${dataset}.${backbone}.${encoder}.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done
