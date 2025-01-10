#!/bin/sh
#SBATCH --job-name=train
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=2
#SBATCH --mem=96G
#SBATCH --array=3-4%1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate ledr

# Start the experiment.
index_dir=${HOME}/indices/beir
data_dir=${HOME}/datasets/beir
backbone=contriever
ckpt=facebook/contriever
exp=span-hn
span_jsonl=spans_tokenized.gte.jsonl


MULTIJOBS=/home/jju/multi-jobs.txt
dataset=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
echo $dataset

# Go
torchrun --nproc_per_node 4 train.py \
    --model_name ${ckpt} \
    --corpus_jsonl ${data_dir}/${dataset}/collection_tokenized/corpus_tokenized.jsonl \
    --corpus_spans_jsonl ${data_dir}/${dataset}/collection_tokenized/${span_jsonl} \
    --select_span_mode weighted \
    --output_dir models/ckpt/${backbone}-${exp}-alpha:1/${dataset} \
    --per_device_train_batch_size 256 \
    --temperature 0.25 --temperature_span 0.25 \
    --pooling mean --span_pooling mean \
    --alpha 1.0 --beta 1.0 --gamma 0.0 \
    --learning_rate 1.25e-7 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --weight_decay 0. \
    --chunk_length 256 \
    --min_chunk_length 32 \
    --ratio_min 0.1 --ratio_max 0.5 \
    --n_negative_samples 1 \
    --mine_neg_using crops \
    --prebuilt_faiss_dir ${index_dir}-neg/${dataset} \
    --prebuilt_negative_jsonl ${data_dir}/${dataset}/collection_tokenized/negative_docidx.jsonl \
    --save_strategy steps \
    --max_steps 10000 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --num_train_epochs 50 \
    --save_total_limit 10 \
    --fp16 --wandb_project exp1-single-dr \
    --report_to wandb --run_name ${dataset}-${exp}-alpha:1
