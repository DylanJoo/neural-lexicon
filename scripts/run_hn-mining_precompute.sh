#!/bin/sh
#SBATCH --job-name=pre
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

cd ${HOME}/neural-lexicon

index_dir=${HOME}/indexes/beir
data_dir=${HOME}/datasets/beir
encoder=thenlper/gte-base

# for dataset in trec-covid nfcorpus fiqa arguana webis-touche2020 quora scidocs scifact nq hotpotqa dbpedia-entity fever climate-fever;do
# only small done

for dataset in scidocs scifact trec-covid nfcorpus fiqa arguana webis-touche2020 quora;do
    python precompute_hn.py \
    --encoder_name_or_path ${encoder} \
    --tokenizer_name_or_path ${encoder} \
    --corpus_jsonl ${data_dir}/${dataset}/collection_tokenized/corpus_tokenized.jsonl \
    --negative_jsonl ${data_dir}/${dataset}/collection_tokenized/negative_docidx.jsonl \
    --faiss_dir ${index_dir}-neg/${dataset} \
    --batch_size 128 \
    --device cuda
done

