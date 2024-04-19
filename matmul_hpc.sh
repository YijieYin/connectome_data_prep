#!/bin/bash

# SLURM submission script for multiple runs with different parameters

#SBATCH --job-name=matmul_adult_cb_neuron
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/cephfs2/yyin/random_inputs/output_messages/result_%A.out  # %A is replaced by job ID, %a by array index
#SBATCH --error=/cephfs2/yyin/random_inputs/error_messages/error_%A.err
#SBATCH --time=01:00:00

# Activate your Python environment
source ../.bashrc

conda activate act_max

inprop_path='data/adult_inprop_cb_neuron.npz'
meta_path='data/adult_cb_neuron_meta.csv'
n_steps = 6
output_path='precomputed/'
prefix='adult_cb_neuron_'


# Call your script with parameters
# Assuming the modified script accepts initializations and output directory as arguments
python activation_maximisation.py --inprop_path $inprop_path --meta_path $meta_path --n_steps $n_steps --output_path $output_path --prefix $prefix
