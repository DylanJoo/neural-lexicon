#!/bin/sh
#SBATCH --job-name=BASE
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:2
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

backbone=contriever
ckpt=facebook/contriever
# backbone=gte
# ckpt=thenlper/gte-base

# for scifact: training longer is better
# for scidocs: training longer is worsen
# Start the experiment.
# for dataset in scidocs scifact;do
for dataset in scidocs scifact;do
    method=baseline
    exp=${method}

    # Go
    torchrun --nproc_per_node 2 \
        train.py \
        --model_name ${ckpt} \
        --train_data_dir /home/dju/datasets/beir/${dataset}/dw-ind-cropping \
        --loading_mode doc2spans \
        --select_span_mode weighted \
        --output_dir models/ckpt/${backbone}-${exp}/${dataset} \
        --per_device_train_batch_size 32 \
        --temperature 0.1 --temperature_span 0.1 \
        --pooling mean --span_pooling mean \
        --alpha 1.0 --beta 0.0 --gamma 0.0 \
        --chunk_length 256 \
        --save_strategy epoch \
        --num_train_epochs 4 \
        --save_total_limit 4 \
        --warmup_ratio 0.1 \
        --fp16 --wandb_project ssldr-exp1 \
        --report_to wandb --run_name ${dataset}-${exp}
done
