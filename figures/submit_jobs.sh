#!/usr/bin/env bash

# sbatch -J megno4 --partition gpu run.sh 0 4
sbatch -J megno20 --partition gpu run.sh 0 20
sbatch -J megno80 --partition gpu run.sh 0 80
# sbatch -J model4 --partition gpu run.sh 1 4
# sbatch -J model20 --partition gpu run.sh 1 20
# sbatch -J model80 --partition gpu run.sh 1 80


