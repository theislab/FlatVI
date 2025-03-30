#!/bin/bash

#SBATCH -o ./logs/train_neuralfim_pancreas.txt

#SBATCH -e ./logs/train_neuralfim_pancreas.txt

#SBATCH -J train_neuralfim_pancreas

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate flatvi_baselines_env

cd /home/icb/alessandro.palma/environment/scCFM/notebooks/rebuttals_icml/train_neuralfim

python train_neuralfim_pancreas.py
 