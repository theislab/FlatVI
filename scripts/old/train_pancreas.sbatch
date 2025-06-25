#!/bin/bash

#SBATCH -o ./logs/1_sweep_geom_eb.output

#SBATCH -e ./logs/1_sweep_geom_eb.error

#SBATCH -J 1_sweep_geom_eb

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate scCFM
export HYDRA_FULL_ERROR=1

python ../scCFM/train_hydra/train_vae.py \
checkpoint=geom_vae \
datamodule=4_sweep_geom_pancreas \
early_stopping=geom_vae \
geometric_vae=geom_vae \
hydra=geom_vae \
launcher=slurm_cluster \
logger=4_pancreas \
model=geom_vae \
train=4_sweep_geom_pancreas \
trainer=geom_vae \

