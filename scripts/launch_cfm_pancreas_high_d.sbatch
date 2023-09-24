#!/bin/bash

#SBATCH -o ./logs/try.output

#SBATCH -e ./logs/try.error

#SBATCH -J try

#SBATCH -p gpu_p

#SBATCH --gres=gpu:1

#SBATCH --qos=gpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate scCFM
export HYDRA_FULL_ERROR=1

cd ../scCFM/train_hydra
# python train_cfm.py datamodule=CFM_pancreas_geodesic_high logger=CFM_pancreas_geodesic_high train=CFM_pancreas_geodesic_high
# python train_cfm.py datamodule=CFM_pancreas_flat_high logger=CFM_pancreas_flat_high train=CFM_pancreas_flat_high
python train_cfm.py datamodule=CFM_pancreas_high logger=CFM_pancreas_vae_high train=CFM_pancreas_vae_high