#!/usr/bin/env bash

sbatch --partition gpu train.sh --total_steps 300000 --random-features --seed 0
