#!/bin/bash

#SBATCH --job-name=ad_split
#SBATCH --partition=cpu
#SBATCH --ntasks=20              
#SBATCH --cpus-per-task=112
#SBATCH --output=mcns_ad_split/result_%A.out  # %A is replaced by job ID, %a by array index
#SBATCH --error=mcns_ad_split/error_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yy432@cam.ac.uk

# Activate your Python environment - you should be in folder interpret_connectome 
source ../.bashrc

conda activate flyconnectome

# remove intermediate results from the previous run 
find /cephfs2/yyin/mcns_ad_split/syn_count/ -mindepth 2 -type f -delete
find /cephfs2/yyin/mcns_ad_split/{axon_in,axon_out,dendrite_in,dendrite_out} -mindepth 1 -type f -delete
find /cephfs2/yyin/mcns_ad_split/seg_indices -maxdepth 1 -type f -delete

# first general axon-dendrite split, to calculate e.g. segregation index
# writing to axon_in, axon_out, dendrite_in, dendrite_out folders
time srun python mcns_ad_split/axon_dendrite_split.py
# manual fixes, has the same content as axon_dendrite_split_mcns.ipynb
# writing to axon_in, axon_out, dendrite_in, dendrite_out folders
python mcns_ad_split/manual_split.py
# put together into edge list for each neuron in folder syn_count
time srun python mcns_ad_split/make_el.py
# get the overall results 
time python mcns_ad_split/make_adj.py

# to re-run, first remove all contents of : 
# 1. folders in syn_count (using `find /cephfs2/yyin/mcns_ad_split/syn_count/ -mindepth 2 -type f -delete`), and 
# 2. folders axon_in, axon_out, dendrite_in, dendrite_out (`find /cephfs2/yyin/mcns_ad_split/{axon_in,axon_out,dendrite_in,dendrite_out} -mindepth 1 -type f -delete`)
# 3. content of seg_indices (`find /cephfs2/yyin/mcns_ad_split/seg_indices -maxdepth 1 -type f -delete`)