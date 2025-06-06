#!/usr/bin/env bash

 # job name
#SBATCH -J period
 # output file
#SBATCH -o out/%A_%a.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 1
#SBATCH --requeue
 # total limit (hh:mm:ss)
#SBATCH -t 48:00:00
#SBATCH --mem=50G
# #SBATCH --gres=gpu:1
# #SBATCH --partition=ellis

source /home/sca63/mambaforge/etc/profile.d/conda.sh

# Enable errexit (exit on error)
set -e

# conda activate bnn_period
# conda activate bnn_chaos_model
conda activate bnn_new_pysr
python "$@"
