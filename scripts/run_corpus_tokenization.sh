#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=tokenize
#SBATCH --partition cpu
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env
TEMP=${HOME}/datasets/temp

cd ${HOME}/neural-lexicon/

for dataset in trec-covid scidocs scifact;do
    mkdir -p ${TEMP}/${dataset}/
    python preprocess/tokenization.py \
        --tokenizer bert-base-uncased \
        --datapath ${HOME}/datasets/beir/${dataset}/collection/corpus.jsonl \
        --overwrite \
        --outdir ${TEMP}/${dataset}
    echo done
done
