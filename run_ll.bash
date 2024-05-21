#!/bin/bash
#SBATCH -N 1 # number of minimum nodes
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH -o slurm.%N.%j.out # stdout goes here
#SBATCH -e slurm.%N.%j.out # stderr goes here

# sbatch -p nlp -A nlp -w nlp-a40-1 --gres=gpu:1 run_ll.bash 


nvidia-smi
CONDA_HOME=$home/miniconda3
CONDA_ENV=spcf_of_rome
CUDA_LAUNCH_BLOCKING=1 
python /home/yanay.soker/Specificity_of_ROME/rome/logit_lens_experiment.py "$1" "$2"


