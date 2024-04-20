#!/bin/sh
#SBATCH --job-name=baseline
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

backbone=contriever
ckpt=facebook/contriever

# Start the experiment.
for dataset in scifact scidocs;do
for method in dev-baseline;do
    exp=${method}

    # Go
    torchrun --nproc_per_node 2 \
        train.py \
        --model_name facebook/contriever \
        --train_data_dir /home/dju/datasets/temp/${dataset} \
        --loading_mode from_precomputed \
        --output_dir models/ckpt/${backbone}-${exp}/${dataset} \
        --per_device_train_batch_size 32 \
        --temperature 0.1 --temperature_span 0.5 \
        --select_span_mode max \
        --pooling mean --span_pooling ${exp} \
        --alpha 1.0 --beta 0.0 \
        --chunk_length 256 \
        --save_strategy steps \
        --max_steps 2000 \
        --save_steps 1000 \
        --save_total_limit 2 \
        --warmup_ratio 0.1 \
        --fp16 --wandb_project tas_span \
        --report_to wandb --run_name ${dataset}-${exp}.baseline
done
done
