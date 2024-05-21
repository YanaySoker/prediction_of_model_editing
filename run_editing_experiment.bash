#!/bin/bash

#SBATCH -N 1 # number of minimum nodes
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH -o slurm.%N.%j.out # stdout goes here
#SBATCH -e slurm.%N.%j.out # stderr goes here

nvidia-smi
CONDA_HOME=$home/miniconda3
CONDA_ENV=prediction_editing_env
CUDA_LAUNCH_BLOCKING=1 
python /home/prediction_of_model_editing/editing_experiment.py  "$1" "$2" "$3"
