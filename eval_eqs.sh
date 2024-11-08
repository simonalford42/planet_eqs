#!/usr/bin/env bash

 # job name
#SBATCH -J eval_eqs
 # output file (%j expands to jobID)
#SBATCH -o out/%A.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 1
#SBATCH --requeue
 # total limit (hh:mm:ss)
#SBATCH -t 48:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=ellis

source /home/sca63/mambaforge/etc/profile.d/conda.sh
conda activate bnn_chaos_model

# Enable errexit (exit on error)
set -e

# example usage:
# bash eval_eqs.sh --version 24880 --pysr_version 11003

# 1. do the main figures comparisons
python -u main_figures.py "$@"
python -u main_figures.py --plot_random "$@"

# 2. do the equal spaced planet comparisons
cd figures
python -u multiswag_5_planet.py --paper-ready "$@"

# 3. do the period ratio figure comparisons
conda activate bnn_period
python -u period_ratio_figure.py --Ngrid 1600 --compute "$@"
python -u period_ratio_figure.py --Ngrid 1600 --plot "$@"
