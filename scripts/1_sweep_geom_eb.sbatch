#!/bin/bash

#SBATCH -o ./logs/1_sweep_geom_eb.output

#SBATCH -e ./logs/1_sweep_geom_eb.error

#SBATCH -J 1_sweep_geom_eb

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

# python ../scCFM/train_hydra/train_vae.py --multirun hydra=geom_vae checkpoint=geom_vae datamodule=1_sweep_geom_eb early_stopping=geom_vae geometric_vae=geom_vae launcher=slurm_cluster model=geom_vae train=1_sweep_geom_eb trainer=geom_vae logger=1_sweep_geom_eb
# python ../scCFM/train_hydra/train_vae.py checkpoint=geom_vae datamodule=2_sweep_geom_hein early_stopping=geom_vae geometric_vae=geom_vae launcher=slurm_cluster model=geom_vae train=2_sweep_geom_hein trainer=geom_vae logger=2_sweep_geom_hein
# python ../scCFM/train_hydra/train_vae.py --multirun hydra=geom_vae checkpoint=geom_vae datamodule=3_sweep_geom_schiebinger early_stopping=geom_vae geometric_vae=geom_vae launcher=slurm_cluster model=geom_vae train=3_sweep_geom_schiebinger trainer=geom_vae logger=3_sweep_geom_schiebinger
