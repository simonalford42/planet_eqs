#!/usr/bin/env bash

 # job name
#SBATCH -J planet_sr
 # output file (%j expands to jobID)
#SBATCH -o sr_out/%A.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 32
#SBATCH --requeue
 # total limit (hh:mm:ss)
#SBATCH -t 02:00:00
#SBATCH --mem=100G
#SBATCH --partition=default_partition

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate bnn_chaos_model

python -u sr.py --slurm_id $SLURM_JOB_ID --slurm_name $SLURM_JOB_NAME "$@"
