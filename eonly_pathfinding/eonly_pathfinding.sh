#!/bin/bash
#SBATCH --job-name=pe
#SBATCH --ntasks=20              
#SBATCH --cpus-per-task=112
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

source ../../.bashrc
conda activate act_max

time srun python eonly_pathfinding.py
time python concat_result.py