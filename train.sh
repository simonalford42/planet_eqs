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
# for seed in `seq 0 2`; do
# Generate a random number between 1 and 9999
random_number=$(shuf -i 1-9999 -n 1)

python find_minima.py --total_steps 300000 --swa_steps 50000 --version $random_number --angles --no_mmr --no_nan --no_eplusminus --slurm_id $SLURM_JOB_ID "$@"
python run_swag.py --total_steps 300000 --swa_steps 50000 --version $random_number --angles --no_mmr --no_nan --no_eplusminus --slurm_id $SLURM_JOB_ID "$@"
# done
