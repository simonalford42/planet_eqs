#!/usr/bin/env bash

 # job name
#SBATCH -J job
 # output file
#SBATCH -o out/%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
 # total limit (hh:mm:ss)
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --partition=default_partition

# exammple call:
# sbatch --array=0-16 job_array.sh --Ngrid 4 --max_t 100000 --compute

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate bnn_period
srun --time=2-00:00:00 python -u period_ratio_figure.py --ground_truth --job_array "$@"
