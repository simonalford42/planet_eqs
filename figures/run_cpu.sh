#!/usr/bin/env bash

#SBATCH -J resonant
#SBATCH -o out/%A_%a.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH -t 72:00:00
#SBATCH --mem=50G

source /home/sca63/mambaforge/etc/profile.d/conda.sh

set -e

conda activate planet_eqs
python -u "$@"
