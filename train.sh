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

version=$((1 + RANDOM % 9999))

python -u find_minima.py --total_steps 300000 --swa_steps 50000  --angles --no_mmr --no_nan --no_eplusminus --version $version --slurm_id $SLURM_JOB_ID "$@"
python -u run_swag.py --total_steps 300000 --swa_steps 50000 --angles --no_mmr --no_nan --no_eplusminus --version $version --slurm_id $SLURM_JOB_ID "$@"
python -u main_figures.py --total_steps 300000 --swa_steps 50000  --angles --no_mmr --no_nan --no_eplusminus --version $version --slurm_id $SLURM_JOB_ID "$@"

# python -u main_figures.py --total_steps 300000 --swa_steps 50000  --angles --no_mmr --no_nan --no_eplusminus --version 3045 --seed 1
# python -u main_figures.py --total_steps 300000 --swa_steps 50000  --angles --no_mmr --no_nan --no_eplusminus --version 1278 --seed 1
# python -u main_figures.py --total_steps 300000 --swa_steps 50000  --angles --no_mmr --no_nan --no_eplusminus --version 1083 --seed 1
