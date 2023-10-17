#!/usr/bin/env bash

# sbatch --partition gpu train.sh --special_args zero
# sbatch --partition gpu train.sh --special_args identity
# sbatch train.sh --random_features
sbatch train.sh --pysr_model results/hall_of_fame_4995_0.pkl
sbatch train.sh --pysr_model results/hall_of_fame_4995_0.pkl --freeze_pysr
