#!/bin/sh
#SBATCH --job-name=train.exp1
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# Start the experiment.
index_dir=${HOME}/indexes/beir
data_dir=${HOME}/datasets/beir
backbone=contriever
ckpt=facebook/contriever
exp=span

for dataset in scidocs scifact;do

    # Go
    torchrun --nproc_per_node 1 \
        train.py \
        --model_name ${ckpt} \
        --corpus_jsonl ${data_dir}/${dataset}/collection_tokenized/corpus_tokenized.jsonl \
        --corpus_spans_jsonl ${data_dir}/${dataset}/collection_tokenized/spans_tokenized.jsonl \
        --output_dir models/ckpt/${backbone}-${exp}/${dataset} \
        --per_device_train_batch_size 16 \
        --temperature 0.1 --temperature_span 0.1 \
        --alpha 1.0 --beta 1.0 --gamma 0.1 \
        --augmentation mask_from_span \
        --prob_augmentation 1.0 \
        --learning_rate 5e-5 \
        --chunk_length 256 \
        --min_chunk_length 32 \
        --select_span_mode weighted \
        --save_strategy steps \
        --max_steps 1500 \
        --save_steps 500 \
        --save_total_limit 4 \
        --warmup_steps 500 \
        --fp16 --wandb_project debug \
        --report_to wandb --run_name ${dataset}-${exp} 
done
