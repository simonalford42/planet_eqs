#!/usr/bin/env bash
# sbatch train.sh --seed 0 --total_steps 200000
# sbatch train.sh --seed 1 --total_steps 200000
# sbatch --partition gpu train.sh --seed 0 --total_steps 100000
# sbatch --partition gpu train.sh --seed 1 --total_steps 100000
# sbatch --partition gpu train.sh --seed 0 --total_steps 50000
# sbatch --partition gpu train.sh --seed 1 --total_steps 50000
sbatch train.sh --seed 0 --total_steps 10000
# sbatch --partition gpu train.sh --seed 1 --total_steps 10000
