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

# for dataset in trec-covid nfcorpus fiqa arguana webis-touche2020 quora scidocs scifact nq hotpotqa dbpedia-entity fever climate-fever;do
# only small done

# decontextualized
# for dataset in scidocs scifact trec-covid nfcorpus fiqa arguana webis-touche2020 quora;do
for dataset in scidocs scifact;do
    python precompute.py \
    --encoder_name_or_path facebook/contriever \
    --tokenizer_name_or_path facebook/contriever \
    --corpus_jsonl ${data_dir}/${dataset}/collection_tokenized/corpus_tokenized.jsonl \
    --corpus_spans_jsonl ${data_dir}/${dataset}/collection_tokenized/spans_tokenized.jsonl \
    --min_ngrams 2 --max_ngrams 3 --stride 1 \
    --decontextualized \
    --num_spans 10 \
    --batch_size 128 \
    --device cuda
done

# contextualized
# for dataset in scidocs scifact trec-covid nfcorpus fiqa arguana webis-touche2020 quora;do
#     python precompute.py \
#     --encoder_name_or_path facebook/contriever \
#     --tokenizer_name_or_path facebook/contriever \
#     --corpus_jsonl ${data_dir}/${dataset}/collection_tokenized/corpus_tokenized.jsonl \
#     --corpus_spans_jsonl ${data_dir}/${dataset}/collection_tokenized/ctx_spans_tokenized.jsonl \
#     --min_ngrams 10 --max_ngrams 10 --stride 5 \
#     --num_spans 10 \
#     --batch_size 128 \
#     --device cuda
# done

# index_dir=${HOME}/indexes/lotte
# data_dir=${HOME}/datasets/lotte
# # science 
# for dataset in lifestyle recreation technology writing;do
#     python precompute.py \
#         --encoder_name_or_path facebook/contriever \
#         --tokenizer_name_or_path facebook/contriever \
#         --corpus_jsonl ${data_dir}/${dataset}/test/collection_tokenized/docs00.json \
#         --corpus_spans_jsonl ${data_dir}/${dataset}/test/collection_tokenized/spans_tokenized.jsonl \
#         --faiss_index_dir ${index_dir}-neg/${dataset} \
#         --min_ngrams 5 --max_ngrams 6 \
#         --num_spans 10 \
#         --batch_size 128 \
#         --device cuda
# done
