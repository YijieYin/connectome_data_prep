#!/bin/bash

# SLURM submission script for multiple runs with different parameters

#SBATCH --job-name=test_multiple_inputs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=result_%A_%a.out  # %A is replaced by job ID, %a by array index
#SBATCH --error=error_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --array=1-10  # Example for 100 tasks

# Activate your Python environment
source .bashrc

conda activate act_max

inprop_path='data/adult_type_inprop.npz'
meta_path='data/adult_meta_meta.npz'
target_index=13702
num_layers=9
optimised_input_path="/cephfs2/yyin/random_inputs/optimised_input/"
output_dir="/cephfs2/yyin/random_inputs/output/"
job_id=${SLURM_ARRAY_TASK_ID}


# Call your script with parameters
# Assuming the modified script accepts initializations and output directory as arguments
python activation_maximisation.py --inprop_path $inprop_path --meta_path $meta_path --target_index $target_index --num_layers $num_layers --optimised_input_path $optimised_input_path --output_path $output_dir --job_id $job_id
