#!/bin/sh
#SBATCH --job-name=debug
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

cd ${HOME}/neural-lexicon/src/sampling

python unit_test.py
