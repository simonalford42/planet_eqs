#!/usr/bin/env bash
sbatch train.sh
sbatch train.sh --f1_variant random_features
sbatch --partition gpu train.sh --f1_variant identity
sbatch --partition gpu train.sh --f1_variant zero
