#!/bin/sh
#SBATCH --job-name=compute
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

cd ${HOME}/neural-lexicon

# [Doc embeddings By doc]
## the weak one from self
python precompute.py \
    --encoder_name_or_path facebook/contriever \
    --tokenizer_name_or_path facebook/contriever \
    --num_spans 10 \
    --num_clusters 0.05 \
    --batch_size 128 \
    --saved_file_format 'doc.by.doc.{}.cluster.{}.pt' \
    --loading_mode from_precomputed \
    --faiss_output doc_emb_ \
    --device cuda

## the strong one from GTE
python precompute.py \
    --encoder_name_or_path thenlper/gte-base \
    --tokenizer_name_or_path thenlper/gte-base \
    --num_spans 10 \
    --num_clusters 0.05 \
    --batch_size 128 \
    --saved_file_format 'doc.by.doc.{}.cluster.{}.pt.strong' \
    --loading_mode from_strong_precomputed \
    --device cuda


# [Doc embeddings By spans]
## do this before everything previously has been checked
# python precompute.py \
#     --encoder_name_or_path facebook/contriever \
#     --tokenizer_name_or_path facebook/contriever \
#     --num_spans 10 \
#     --num_clusters 0.05 \
#     --batch_size 128 \
#     --saved_file_format 'doc.by.spans.{}.cluster.{}.pt' \
#     --loading_mode from_precomputed \
#     --faiss_output spans_emb_ \
#     --device cuda
