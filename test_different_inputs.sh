#!/bin/bash

# SLURM submission script for multiple runs with different parameters

#SBATCH --job-name=test_multiple_inputs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=/cephfs2/yyin/random_inputs/output_messages/result_%A_%a.out  # %A is replaced by job ID, %a by array index
#SBATCH --error=/cephfs2/yyin/random_inputs/error_messages/error_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --array=1-30  # Example for 100 tasks

# Activate your Python environment
source ../.bashrc

conda activate act_max

inprop_path='data/adult_type_inprop.npz'
meta_path='data/adult_type_meta.csv'
target_index='12303,12304,12305,12306,12307,12308,12309,12310,12311,12312,12313,12314,12315,12316,12317,12318,12319,12320,12321'
num_layers=3
optimised_input_path="/cephfs2/yyin/random_inputs/optimised_input/"
output_dir="/cephfs2/yyin/random_inputs/output/"
array_id=${SLURM_ARRAY_TASK_ID}


# Call your script with parameters
# Assuming the modified script accepts initializations and output directory as arguments
python activation_maximisation.py --inprop_path $inprop_path --meta_path $meta_path --target_indices $target_index --num_layers $num_layers --optimised_input_path $optimised_input_path --output_path $output_dir --array_id $array_id

$inprop_path='data/adult_type_inprop.npz'                                                                                                                      
$meta_path='data/adult_type_meta.csv'                                                                                                                          
$target_index='12303,12304,12305,12306,12307,12308,12309,12310,12311,12312,12313,12314,12315,12316,12317,12318,12319,12320,12321'                              
$num_layers=3                                                                                                                                                  
$optimised_input_path='optimised_input/'                                                                                                                       
$output_dir='output/'                                                                                                                                          
$array_id = 1

python activation_maximisation.py --inprop_path $inprop_path --meta_path $meta_path --target_indices $target_index --num_layers $num_layers --optimised_input_path $optimised_input_path --output_path $output_dir --array_id $array_id --wandb $true
