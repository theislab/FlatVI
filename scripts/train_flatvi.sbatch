#!/bin/bash

#SBATCH -o ./logs/train_flatvi.output

#SBATCH -e ./logs/train_flatvi.error

#SBATCH -J train_flatvi

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

conda activate flatvi
export HYDRA_FULL_ERROR=1

# Embryoid body dataset

# Pancreas dataset

# MEF dataset
