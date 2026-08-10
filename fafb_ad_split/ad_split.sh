#!/bin/bash

#SBATCH --job-name=ad_split
#SBATCH --partition=cpu
#SBATCH --ntasks=20              
#SBATCH --cpus-per-task=112
#SBATCH --output=fafb_ad_split/result_%A.out  # %A is replaced by job ID, %a by array index
#SBATCH --error=fafb_ad_split/error_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yy432@cam.ac.uk

# Activate your Python environment - you should be in folder interpret_connectome 
source ../.bashrc

conda activate flyconnectome

# remove intermediate results from the previous run
# -mindepth 1 so from_nosplit_syn_count.csv and to_nosplit_syn_count.csv go too
find /cephfs2/yyin/ad_split/syn_count/ -mindepth 1 -type f -delete
find /cephfs2/yyin/ad_split/{axon_in,axon_out,dendrite_in,dendrite_out} -mindepth 1 -type f -delete
# split_failures may not exist yet on a first run, hence the 2>/dev/null
find /cephfs2/yyin/ad_split/{seg_indices,split_failures} -maxdepth 1 -type f -delete 2>/dev/null

# stop at the first failing stage: a stage that dies part-way leaves the later
# ones running on incomplete/stale input, which silently produces wrong results
set -e

# first general axon-dendrite split, to calculate e.g. segregation index
# writing to axon_in, axon_out, dendrite_in, dendrite_out folders
time srun python fafb_ad_split/axon_dendrite_split.py
# manual fixes, has the same content as axon_dendrite_split_FAFB.ipynb
# writing to axon_in, axon_out, dendrite_in, dendrite_out folders
python fafb_ad_split/manual_split.py
# put together into edge list for each neuron in folder syn_count
time srun python fafb_ad_split/make_el.py
# get the overall results 
time python fafb_ad_split/make_adj.py

# to re-run, first remove all contents of : 
# 1. folders in syn_count (using `find /cephfs2/yyin/ad_split/syn_count/ -mindepth 1 -type f -delete`), and
# 2. folders axon_in, axon_out, dendrite_in, dendrite_out (`find /cephfs2/yyin/ad_split/{axon_in,axon_out,dendrite_in,dendrite_out} -mindepth 1 -type f -delete`)
# 3. content of seg_indices and split_failures (`find /cephfs2/yyin/ad_split/{seg_indices,split_failures} -maxdepth 1 -type f -delete`)