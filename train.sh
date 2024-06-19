#!/usr/bin/env bash

 # job name
#SBATCH -J bnn_chaos
 # output file (%j expands to jobID)
#SBATCH -o out/%A.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 1
#SBATCH --requeue
 # total limit (hh:mm:ss)
#SBATCH -t 24:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=ellis

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate bnn_chaos_model

# Enable errexit (exit on error)
set -e

version=$((1 + RANDOM % 999999))
version2=$((1 + RANDOM % 999999))

# python -u find_minima.py --total_steps 300000 --version $version --f1_variant linear --f2_variant mlp 
# first run of pysr on f2 (direct)
python -u sr.py --version 29170 --target f2_direct --seed 0
# second run of pysr on f2 (input is summary stats + equations)
# python -u sr.py --version 29170 --target f2_direct --residual --previous_sr_path sr_results/92985.pkl --seed 0

# for direct pysr validation loss evaluation
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5456.pkl --pysr_f2_model_selection best --total_steps 100 --load_f1 29170

# for residual pysr validation loss evaluation
# python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5456.pkl --pysr_f2_residual sr_results/92071.pkl --pysr_f2_model_selection best --pysr_f2_residual_model_selection best --total_steps 100 --load_f1 29170

# for residual (equations) pysr validation loss
#python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5456.pkl --pysr_f2_residual sr_results/92985.pkl --pysr_f2_model_selection best --pysr_f2_residual_model_selection best --total_steps 100 --load_f1 29170
python -u find_minima.py --version $version2 --eval --pysr_f2 sr_results/5456.pkl --pysr_f2_residual sr_results/52420.pkl --pysr_f2_model_selection best --pysr_f2_residual_model_selection best --total_steps 100 --load_f1 29170


# .latex_table()
# model.equations_
# model.equations_.plot(“Complexity”, "Loss")
