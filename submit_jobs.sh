#!/usr/bin/env bash

# sbatch --partition ellis --time=02:00:00 sr_slurm.sh --time_in_hours 1
# sbatch --partition ellis --time=08:00:00 sr_slurm.sh --time_in_hours 7
# sbatch --time=01:00:00 sr_slurm.sh --time_in_hours 0.5

# sbatch --partition gpu train.sh --l1_reg
# sbatch --partition gpu train.sh --f1_variant pysr --pysr_model sr_results/hall_of_fame_1278_1_0.pkl
sbatch --partition gpu train.sh --f1_variant bimt --pysr_model sr_results/hall_of_fame_1278_1_0.pkl