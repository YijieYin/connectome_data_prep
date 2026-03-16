#!/bin/bash

#SBATCH --job-name=ad_split
#SBATCH --partition=cpu
#SBATCH --ntasks=20              
#SBATCH --cpus-per-task=112
#SBATCH --output=result_%A.out  # %A is replaced by job ID, %a by array index
#SBATCH --error=error_%A.err

# Activate your Python environment - you should be in folder dev 
source ../.bashrc

conda activate flyconnectome

# first general axon-dendrite split, to calculate e.g. segregation index
# writing to axon_in, axon_out, dendrite_in, dendrite_out folders
time srun python Python/fafb_ad_split/axon_dendrite_split.py
# manual fixes, has the same content as axon_dendrite_split_FAFB.ipynb
# writing to axon_in, axon_out, dendrite_in, dendrite_out folders
python Python/fafb_ad_split/manual_split.py
# put together into edge list for each neuron in folder syn_count
time srun python Python/fafb_ad_split/make_el.py
# then go to axon_dendrite_split_FAFB.ipynb, section make_adj, to get the overall results 

# to re-run, first remove all contents of : 
# 1. folders in syn_count, and 
# 2. folders axon_in, axon_out, dendrite_in, dendrite_out
# 3. content of seg_indices 