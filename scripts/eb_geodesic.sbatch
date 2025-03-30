#!/bin/bash

#SBATCH -o ./logs/cfm_eb_geodesic.output

#SBATCH -e ./logs/cfm_eb_geodesic.error

#SBATCH -J cfm_eb_geodesic

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

# python ../scCFM/train_hydra/train_cfm.py checkpoint=eb_geodesic datamodule=eb_geodesic early_stopping=geom_vae launcher=slurm_cluster logger=eb_geodesic model=eb_geodesic train=eb_geodesic trainer=eb_geodesic net=default 

# python ../scCFM/train_hydra/train_cfm.pyx --multirun checkpoint=cfm datamodule=hein_cfm early_stopping=cfm hydra=cfm_hein launcher=slurm_cluster logger=cfm_hein model=eb_geodesic train=cfm_hein trainer=cfm_hein net=default 

python ../scCFM/train_hydra/train_cfm.py checkpoint=eb_geodesic datamodule=eb_geodesic early_stopping=geom_vae launcher=slurm_cluster logger=eb_geodesic model=eb_geodesic train=eb_geodesic trainer=eb_geodesic net=default 
