#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=RT
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.output

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# Start the experiment.
index_dir=${HOME}/indexes/beir
data_dir=${HOME}/datasets/beir

# Setting of encoders
backbone=contriever
exp=ft-on-ms
encoder=facebook/contriever-msmarco

# Go
## smaller corpus
for dataset in trec-covid nfcorpus fiqa arguana webis-touche2020 quora scidocs scifact;do

## the larger copurs
# for dataset in nq hotpotqa dbpedia-entity fever climate-fever;do

    echo indexing...
    python3 retrieval/dense_index.py input \
        --corpus ${data_dir}/${dataset}/collection \
        --fields text title \
        --shard-id 0 \
        --shard-num 1 output \
        --embeddings ${index_dir}/${dataset}/${backbone}-${exp}.faiss \
        --to-faiss encoder \
        --encoder-class ${backbone} \
        --encoder ${encoder} \
        --pooling mean \
        --fields text title \
        --batch-size 32 \
        --max-length 256 \
        --device cuda

    echo searching...
    python retrieval/dense_search.py \
        --k 100  \
        --index ${index_dir}/${dataset}/${backbone}-${exp}.faiss \
        --encoder_path ${encoder} \
        --encoder_class ${backbone} \
        --topic ${data_dir}/${dataset}/queries.jsonl \
        --qrels ${data_dir}/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt \
        --batch_size 64 \
        --pooling mean \
        --device cuda \
        --output runs/${backbone}-${exp}/run.beir.${dataset}.${backbone}.${exp}.txt

    echo -ne "beir | ${exp} | ${dataset} | reproduced | "
    ~/trec_eval-9.0.7/trec_eval \
        -c -m ndcg_cut.10 -m recall.100 \
        ${data_dir}/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt \
        runs/${backbone}-${exp}/run.beir.${dataset}.${backbone}.${exp}.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done
