#!/bin/sh
#SBATCH --job-name=train.exp1
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate ledr

# Start the experiment.
data_dir=${HOME}/datasets/beir
backbone=contriever
ckpt=facebook/contriever

exp=span
span_jsonl=spans_tokenized.jsonl
augmentation=mask_from_span

# exp=span-ctx
# span_jsonl=ctx_spans_tokenized.jsonl
# augmentation=mask_from_span

for dataset in scidocs scifact trec-covid nfcorpus fiqa arguana webis-touche2020 quora;do

    # Go
    torchrun --nproc_per_node 2 \
        train.py \
        --model_name ${ckpt} \
        --corpus_jsonl ${data_dir}/${dataset}/collection_tokenized/corpus_tokenized.jsonl \
        --corpus_spans_jsonl ${data_dir}/${dataset}/collection_tokenized/${span_jsonl} \
        --select_span_mode weighted \
        --output_dir models/ckpt/${backbone}-${exp}/${dataset} \
        --per_device_train_batch_size 32 \
        --temperature 0.1 --temperature_span 0.1 \
        --pooling mean --span_pooling mean \
        --alpha 1.0 --beta 1.0 --gamma 0.0 \
        --learning_rate 5e-5 \
        --chunk_length 256 \
        --min_chunk_length 32 \
        --ratio_min 0.1 --ratio_max 0.5 \
        --augmentation ${augmentation} \
        --prob_augmentation 1.0 \
        --save_strategy steps \
        --max_steps 1500 \
        --save_steps 500 \
        --save_total_limit 4 \
        --fp16 --wandb_project exp1-single-dr  \
        --report_to wandb --run_name ${dataset}-${exp}
done
