#!/bin/sh
#SBATCH --job-name=infer-exp1
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
index_dir=${HOME}/indexes/lotte
data_dir=${HOME}/datasets/lotte
# Setting of encoders
backbone=contriever
exp=baseline

for dataset in writing recreation technology lifestyle;do

    # Go
    # for ckpt in 1000 2000;do
    #
    #     encoder=models/ckpt/${backbone}-${exp}/${dataset}/checkpoint-${ckpt}
    #     echo indexing...
    #     python3 retrieval/dense_index.py input \
    #         --corpus ${data_dir}/${dataset}/test/collection \
    #         --fields text \
    #         --shard-id 0 \
    #         --shard-num 1 output \
    #         --embeddings ${index_dir}/${dataset}/${backbone}-${exp}.faiss \
    #         --to-faiss encoder \
    #         --encoder-class ${backbone} \
    #         --encoder ${encoder} \
    #         --pooling mean \
    #         --fields text \
    #         --batch-size 32 \
    #         --max-length 256 \
    #         --device cuda
    #
    #     echo searching...
    #     python retrieval/dense_search.py \
    #         --k 100  \
    #         --index ${index_dir}/${dataset}/${backbone}-${exp}.faiss \
    #         --encoder_path ${encoder} \
    #         --encoder_class ${backbone} \
    #         --topic ${data_dir}/${dataset}/test/questions.search.tsv \
    #         --batch_size 64 \
    #         --pooling mean \
    #         --device cuda \
    #         --output runs/${backbone}-${exp}/run.lotte.${dataset}.${backbone}.${exp}.txt

        echo -ne "lotte-test | ${exp} | ${dataset} | ${ckpt} | "
        ~/trec_eval-9.0.7/trec_eval \
            -c -m ndcg_cut.10 \
            ${data_dir}/${dataset}/test/qrels.lotte-${dataset}-test.search.txt \
            runs/${backbone}-${exp}/run.lotte.${dataset}.${backbone}.${exp}.txt \
            | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
    done
done
