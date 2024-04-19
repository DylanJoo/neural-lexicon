#!/bin/sh
#SBATCH --job-name=test
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

backbone=contriever
ckpt=facebook/contriever

# Start the experiment.
for dataset in scifact;do
for method in dev;do
    exp=${method}

    # Go
    torchrun --nproc_per_node 1 \
        train.py \
        --model_name facebook/contriever \
        --train_data_dir /home/dju/datasets/temp/${dataset} \
        --loading_mode from_precomputed \
        --output_dir models/ckpt/${backbone}-${exp}/${dataset} \
        --per_device_train_batch_size 32 \
        --temperature 0.1 \
        --temperature_span 0.5 \
        --pooling cls \
        --span_pooling ${exp} \
        --select_span_mode uniform \
        --chunk_length 256 \
        --num_train_epochs 2 \
        --save_strategy epoch \
        --save_total_limit 2 \
        --warmup_ratio 0.1 \
        --fp16 \
        --report_to wandb --run_name ${dataset}-${exp} --wandb_project debug
done
done
