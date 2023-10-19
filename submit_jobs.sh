#!/usr/bin/env bash

sbatch --partition gpu train.sh --seed 0 --latent 1
# sbatch --partition gpu train.sh --f1_variant random_features --seed 1
# sbatch --partition gpu train.sh --f1_variant identity --seed 1
# sbatch --partition gpu train.sh --f1_variant zero --seed 1
# sbatch --partition ellis train.sh --seed 1 --f1_variant pysr_frozen --pysr_model sr_results/hall_of_fame_9723_0.pkl
# sbatch --partition ellis train.sh --seed 1 --f1_variant pysr --pysr_model sr_results/hall_of_fame_9723_0.pkl
# sbatch sr_slurm.sh
