#!/bin/bash

# SLURM submission script for multiple runs with different parameters

#SBATCH --job-name=act_max
#SBATCH --partition=ml
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=result_%A.out  # %A is replaced by job ID, %a by array index
#SBATCH --error=error_%A.err
#SBATCH --time=01:00:00
#SBATCH --array=1
#SBATCH --nodelist=fmg102

# Activate your Python environment
source ../.bashrc

conda activate act_max

inprop_path='data/adult_inprop_cb_neuron_no_CX_axonic_postsynapses.npz'
meta_path='data/adult_cb_neuron_meta_no_CX_axonic_postsynapses.csv'
# inprop_path='data/adult_inprop_cb_neuron_NoPENpostsynapsesInEB_noPostsynapsesInFBforTangential.npz'
# meta_path='data/adult_cb_neuron_meta_NoPENpostsynapsesInEB_noPostsynapsesInFBforTangential.csv'
num_iterations=60
optimised_input_path="/cephfs2/yyin/tangential/optimised_input/"
output_dir="/cephfs2/yyin/tangential/output/"
weights_path='/cephfs2/yyin/moving_bump/updated_weights/'
# array_id=${SLURM_ARRAY_TASK_ID}
array_id=${SLURM_JOB_ID}

# nvidia-smi -l 1 > gpu_usage.log &  # Log GPU usage every second
# GPU_SMIPID=$!

# Call your script with parameters
# Assuming the modified script accepts initializations and output directory as arguments
python activation_maximisation.py --inprop_path $inprop_path --meta_path $meta_path --num_iterations $num_iterations --optimised_input_path $optimised_input_path --output_path $output_dir --weights_path $weights_path --array_id $array_id  

# kill $GPU_SMIPID
# echo "Logged GPU usage to gpu_usage.log"
