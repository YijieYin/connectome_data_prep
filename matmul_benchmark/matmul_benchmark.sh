#!/bin/bash

# SLURM submission script for multiple runs with different parameters

#SBATCH --job-name=mm
#SBATCH --partition=ml
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --output=result_%A.out  # %A is replaced by job ID, %a by array index
#SBATCH --error=error_%A.err
#SBATCH --nodelist=fmg104

# for interactive, run the following line: 
# srun --gres=gpu:1 --mem=256G --partition=ml --nodelist=fmg103 --pty bash
# for multiple GPUs: 
# srun --gres=gpu:4 --partition=gpu --mem=128G --ntasks-per-node=4 --pty bash

# Activate your Python environment
source ../../.bashrc

conda activate act_max

python matmul_benchmark.py