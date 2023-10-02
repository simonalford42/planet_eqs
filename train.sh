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
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --partition=ellis

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate bnn_chaos_model

# Enable errexit (exit on error)
set -e

version=$((1 + RANDOM % 9999))

# python find_minima.py --total_steps 300000 --swa_steps 50000  --angles --no_mmr --no_nan --no_eplusminus --version $version --pysr_model 'results/hall_of_fame_7955_5.pkl' --slurm_id $SLURM_JOB_ID "$@"
python run_swag.py --total_steps 300000 --swa_steps 50000 --angles --no_mmr --no_nan --no_eplusminus --version $version --slurm_id $SLURM_JOB_ID "$@"
# python main_figures.py --version $version --total_steps 300000 --swa_steps 50000 --angles --no_mmr --no_nan --no_eplusminus --plot "$@"
