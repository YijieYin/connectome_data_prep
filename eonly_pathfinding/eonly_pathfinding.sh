#!/bin/bash
#SBATCH --job-name=pe
#SBATCH --ntasks=20              
#SBATCH --cpus-per-task=100 # 500 GB 
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

source ../../.bashrc
conda activate act_max

srun python eonly_pathfinding.py
python concat_result.py