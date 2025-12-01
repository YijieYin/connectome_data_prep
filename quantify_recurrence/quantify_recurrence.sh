#!/bin/bash
#SBATCH --job-name=re
#SBATCH --ntasks=20              
#SBATCH --cpus-per-task=112
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

source ../../.bashrc
conda activate act_max

srun python quantify_recurrence.py
python concat_result.py