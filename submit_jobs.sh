#!/usr/bin/env bash
sbatch train.sh --seed 2 --total_steps 300000
sbatch train.sh --seed 3 --total_steps 300000
sbatch --partition gpu train.sh --seed 4 --total_steps 300000
sbatch --partition gpu train.sh --seed 5 --total_steps 300000
