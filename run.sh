#!/usr/bin/env bash

 # job name
#SBATCH -J run
 # output file (%j expands to jobID)
#SBATCH -o %A.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 32
#SBATCH --requeue
 # total limit (hh:mm:ss)
#SBATCH -t 02:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate bnn_chaos_model

python -u "$@"

