#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=tokenize
#SBATCH --partition cpu
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# Start the experiment.
data_dir=${HOME}/datasets

## BEIR
for dataset in trec-covid nfcorpus fiqa arguana webis-touche2020 quora scidocs scifact nq hotpotqa dbpedia-entity fever climate-fever;do
    mkdir -p ${data_dir}/beir/${dataset}/collection_tokenized

    python tools/tokenization.py \
        --tokenizer bert-base-uncased \
        --datapath ${data_dir}/beir/${dataset}/collection/corpus.jsonl \
        --overwrite \
        --outdir ${data_dir}/beir/${dataset}/collection_tokenized
    echo done
done

## Lotte
# for dataset in lifestyle recreation science technology writing;do
#     mkdir -p ${data_dir}/lotte/${dataset}/test/collection_tokenized
#
#     python tools/tokenization.py \
#         --tokenizer bert-base-uncased \
#         --datapath ${data_dir}/lotte/${dataset}/test/collection/docs00.json \
#         --overwrite \
#         --outdir ${data_dir}/lotte/${dataset}/test/collection_tokenized \
#         --field contents
#     echo done
# done
