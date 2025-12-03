#!/bin/bash
#SBATCH --job-name=pe
#SBATCH --partition=agpu
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

source ../../.bashrc
conda activate act_max

time srun python path_effconn_benchmark.py
time python concat_result.py