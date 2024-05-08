#!/bin/sh
#SBATCH --job-name=SPANS
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:1
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

cd ${HOME}/neural-lexicon

# self encoder
# [Doc embeddings By doc]
echo 'self span + clusters from doc embeddings + doc indexing'
python precompute.py \
    --encoder_name_or_path facebook/contriever \
    --tokenizer_name_or_path facebook/contriever \
    --min_ngrams 2 --max_ngrams 3 \
    --num_spans 10 \
    --batch_size 128 \
    --saved_file_format 'doc2spans.{}.pt.ctrv' \
    --loading_mode doc2spans \
    --device cuda

# [Doc embeddings By spans]
## actaully, the spans would be identical. 
## Only the clusters and indexing are difffrom the previopus one.
# echo 'self span + clusters from spans embeddings + spans indexing'
# python precompute.py \
#     --encoder_name_or_path facebook/contriever \
#     --tokenizer_name_or_path facebook/contriever \
#     --num_spans 10 \
#     --num_clusters 0.05 \
#     --batch_size 128 \
#     --saved_file_format 'doc.by.spans.{}.cluster.{}.pt.ctrv' \
#     --doc_embeddings_by_spans  \
#     --loading_mode from_precomputed \
#     --device cuda  \
#     --faiss_output spans_emb_ctrv_

# GTE, the stronger encode
# echo 'teacher span + clusters from doc embeddings + doc indexing'
# python precompute.py \
#     --encoder_name_or_path thenlper/gte-base \
#     --tokenizer_name_or_path thenlper/gte-base \
#     --num_spans 10 \
#     --num_clusters 0.05 \
#     --batch_size 128 \
#     --saved_file_format 'doc.by.doc.{}.cluster.{}.pt.gte' \
#     --loading_mode from_strong_precomputed \
#     --device cuda \
#     --faiss_output doc_emb_gte_

# echo 'teacher span + clusters from spans embeddings + spans indexing'
# python precompute.py \
#     --encoder_name_or_path thenlper/gte-base \
#     --tokenizer_name_or_path thenlper/gte-base \
#     --num_spans 10 \
#     --num_clusters 0.05 \
#     --batch_size 128 \
#     --saved_file_format 'doc.by.spans.{}.cluster.{}.pt.gte' \
#     --doc_embeddings_by_spans  \
#     --loading_mode from_strong_precomputed \
#     --device cuda  \
#     --faiss_output spans_emb_gte_
