#!/bin/bash
#SBATCH --job-name=pe
#SBATCH --partition=agpu
#SBATCH --ntasks=2              # 1 task per GPU
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --mem=40G # memory per node

source ../../.bashrc
conda activate act_max

srun python eonly_pathfinding.py
