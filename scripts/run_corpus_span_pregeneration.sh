#!/bin/sh
#SBATCH --job-name=gen-sp
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:1
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

cd ${HOME}/neural-lexicon

python span_pregeneration.py
