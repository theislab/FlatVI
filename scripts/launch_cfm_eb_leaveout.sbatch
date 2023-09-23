#!/bin/bash

#SBATCH -o ./logs/launch_cfm_eb_leaveout.output

#SBATCH -e ./logs/launch_cfm_eb_leaveout.error

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
python train_cfm.py --multirun hydra=CFM_eb_geodesic_leaveout datamodule=CFM_eb_geodesic_leaveout logger=CFM_eb_geodesic_leaveout train=CFM_eb_geodesic_leaveout 
