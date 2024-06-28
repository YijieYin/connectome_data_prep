#!/bin/bash

# SLURM submission script for multiple runs with different parameters

#SBATCH --job-name=matmul_adult_cb_neuron
#SBATCH --partition=ml
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=result_%A.out  # %A is replaced by job ID, %a by array index
#SBATCH --error=error_%A.err
#SBATCH --time=01:00:00

# Activate your Python environment
source ../.bashrc

conda activate act_max

inprop_path='data/neuprint_inprop_optic.npz'
meta_path='data/neuprint_meta_optic.csv'
n_steps=6
output_path='precomputed/maleCNS_neuprint_optic_neuron/'
prefix='maleCNS_neuprint_optic_neuron_'

# uncomment the following lines to record memory usage 
# nvidia-smi -l 1 > gpu_usage.log &  # Log GPU usage every second
# GPU_SMIPID=$!

# optional flags: --nroot  # for n-rooting the results 
python matmul_hpc.py --inprop_path $inprop_path --meta_path $meta_path --n_steps $n_steps --output_path $output_path --prefix $prefix 

# uncomment the following lines to record memory usage 
# kill $GPU_SMIPID
# echo "Logged GPU usage to gpu_usage.log"