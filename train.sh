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

# python -u find_minima.py --total_steps 300000 --version $version --slurm_id $SLURM_JOB_ID "$@" --f1_variant linear --f2_variant mlp 
# python -u sr.py --version 29766 --target f2 --seed 0

#for direct pysr validation loss evaluation
#python -u find_minima.py --total_steps 300000 --version 30590 --slurm_id $SLURM_JOB_ID "$@" --pysr_f2 sr_results/51084.pkl --load 30590 --f1_variant pysr_frozen

#for residual pysr validation loss evaluation
python -u find_minima.py --total_steps 300000 --version 29766 --slurm_id $SLURM_JOB_ID "$@" --f2_variant pysr_residual --pysr_f2 sr_results/33796.pkl --load 29766

# python -u run_swag.py --total_steps 300000 --swa_steps 50000 --version $version --slurm_id $SLURM_JOB_ID "$@" --f1_variant linear --f2_variant pysr_residual
# python -u find_minima.py --total_steps 300000 --swa_steps 50000  --angles --no_mmr --no_nan --no_eplusminus --version $version2 --slurm_id $SLURM_JOB_ID "$@" --f1_variant bimt --pysr_model sr_results/hall_of_fame_27379_0.pkl 
# python -u run_swag.py --total_steps 300000 --swa_steps 50000 --angles --no_mmr --no_nan --no_eplusminus --version $version2 --slurm_id $SLURM_JOB_ID "$@" --f1_variant bimt --pysr_model sr_results/hall_of_fame_27379_0.pkl 
# python -u sr.py --version $version --target f2 --seed 0 #--time_in_hours 0.1 --max_size 60

# .latex_table()
# model.equations_
# model.equations_.plot(“Complexity”, "Loss")