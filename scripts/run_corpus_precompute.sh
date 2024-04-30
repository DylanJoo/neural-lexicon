#!/bin/sh
#SBATCH --job-name=compute
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:1
#SBATCH --mem=64G
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
echo 'self span + clusters from doc embeddings + doc indexing'
python precompute.py \
    --encoder_name_or_path facebook/contriever \
    --tokenizer_name_or_path facebook/contriever \
    --num_spans 10 \
    --num_clusters 0.05 \
    --batch_size 128 \
    --saved_file_format 'doc.by.doc.{}.cluster.{}.pt.ctrv' \
    --loading_mode from_precomputed \
    --device cuda \
    --faiss_output doc_emb_ctrv_

# [Doc embeddings By spans]
## actaully, the spans would be identical. Only the clusters and indexing are diff
## from the previopus one.
echo 'self span + clusters from spans embeddings + spans indexing'
python precompute.py \
    --encoder_name_or_path facebook/contriever \
    --tokenizer_name_or_path facebook/contriever \
    --num_spans 10 \
    --num_clusters 0.05 \
    --batch_size 128 \
    --saved_file_format 'doc.by.spans.{}.cluster.{}.pt.ctrv' \
    --doc_embeddings_by_spans  \
    --loading_mode from_precomputed \
    --device cuda \
    --faiss_output spans_emb_ctrv_

# the stronger encoder, GTE.
# However, during training, it's incorret to use this indexing for other types.
echo 'teacher span + clusters from doc embeddings + doc indexing'
python precompute.py \
    --encoder_name_or_path thenlper/gte-base \
    --tokenizer_name_or_path thenlper/gte-base \
    --num_spans 10 \
    --num_clusters 0.05 \
    --batch_size 128 \
    --saved_file_format 'doc.by.doc.{}.cluster.{}.pt.gte' \
    --loading_mode from_strong_precomputed \
    --device cuda \
    --faiss_output doc_emb_gte_

echo 'teacher span + clusters from spans embeddings + spans indexing'
python precompute.py \
    --encoder_name_or_path thenlper/gte-base \
    --tokenizer_name_or_path thenlper/gte-base \
    --num_spans 10 \
    --num_clusters 0.05 \
    --batch_size 128 \
    --saved_file_format 'doc.by.spans.{}.cluster.{}.pt.gte' \
    --doc_embeddings_by_spans  \
    --loading_mode from_strong_precomputed \
    --device cuda \
    --faiss_output spans_emb_gte_
