#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=RT
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=%x.%j.output

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# Start the experiment.
index_dir=${HOME}/indexes/lotte
data_dir=${HOME}/datasets/lotte

# Setting of encoders
backbone=contriever
exp=ft-on-ms
encoder=facebook/contriever-msmarco

# select testing set with search (we will ignore the identified for brevity)
split=test 
query_type=search

# Go
for dataset in lifestyle recreation science technology writing;do

    # echo indexing...
    # python3 retrieval/dense_index.py input \
    #     --corpus ${data_dir}/${dataset}/$split/collection \
    #     --fields text \
    #     --shard-id 0 \
    #     --shard-num 1 output \
    #     --embeddings ${index_dir}/${dataset}/${backbone}-${exp}.faiss \
    #     --to-faiss encoder \
    #     --encoder-class ${backbone} \
    #     --encoder ${encoder} \
    #     --pooling mean \
    #     --fields text \
    #     --batch-size 32 \
    #     --max-length 256 \
    #     --device cuda

    # echo searching... # we only use the query-type "search"
    # python retrieval/dense_search.py \
    #     --k 100  \
    #     --index ${index_dir}/${dataset}/${backbone}-${exp}.faiss \
    #     --encoder_path ${encoder} \
    #     --encoder_class ${backbone} \
    #     --topic ${data_dir}/${dataset}/$split/questions.$query_type.tsv \
    #     --batch_size 64 \
    #     --pooling mean \
    #     --device cuda \
    #     --output runs/${backbone}-${exp}/run.lotte.${dataset}.${backbone}.${exp}.txt

    echo -ne "lotte-search | ${exp} | ${dataset} | reproduced | "
    ~/trec_eval-9.0.7/trec_eval \
        -c -m ndcg_cut.10 -m recall.100 \
        ${data_dir}/${dataset}/$split/qrels.lotte-${dataset}-$split.$query_type.txt \
        runs/${backbone}-${exp}/run.lotte.${dataset}.${backbone}.${exp}.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done

# Eval (S@5)
echo "[Lotte-${split}]"
python tools/evaluate_lotte_rankings.py \
        --k 5 --split $split \
        --data_dir ${data_dir} \
        --retrieval contriever.ft-on-ms \
        --rankings_dir runs/contriever-ft-on-ms
