#!/bin/bash

# Parameters
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=6
#SBATCH --error=/home/icb/alessandro.palma/environment/scCFM/scCFM/train_hydra/multirun/2023-09-23/15-25-30/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_cfm
#SBATCH --mem=90GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home/icb/alessandro.palma/environment/scCFM/scCFM/train_hydra/multirun/2023-09-23/15-25-30/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --signal=USR2@120
#SBATCH --time=1440
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/icb/alessandro.palma/environment/scCFM/scCFM/train_hydra/multirun/2023-09-23/15-25-30/.submitit/%A_%a/%A_%a_%t_log.out --error /home/icb/alessandro.palma/environment/scCFM/scCFM/train_hydra/multirun/2023-09-23/15-25-30/.submitit/%A_%a/%A_%a_%t_log.err /home/icb/alessandro.palma/miniconda3/envs/scCFM/bin/python -u -m submitit.core._submit /home/icb/alessandro.palma/environment/scCFM/scCFM/train_hydra/multirun/2023-09-23/15-25-30/.submitit/%j
