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

cd ../flatvi/train_hydra/

# Embryoid body dataset
python ../scCFM/train_hydra/train_vae.py \
checkpoint=geom_vae \
datamodule=VAE_geom_eb \
logger=VAE_geom_eb \
model=vae_default \
train=VAE_geom_eb \
trainer=VAE_default \

# Pancreas dataset
# python ../scCFM/train_hydra/train_vae.py \
# checkpoint=geom_vae \
# datamodule=CFM_geom_pancreas \
# logger=CFM_geom_pancreas \
# model=vae_default \
# train=VAE_geom_pancreas \
# trainer=VAE_default \

# MEF dataset
# python ../scCFM/train_hydra/train_vae.py \
# checkpoint=geom_vae \
# datamodule=VAE_geom_schiebinger \
# logger=VAE_geom_schiebinger \
# model=vae_default \
# train=VAE_geom_schiebinger \
# trainer=VAE_default \
