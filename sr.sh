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
#SBATCH -t 09:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=ellis

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate new_bnn

python -u "$@"
