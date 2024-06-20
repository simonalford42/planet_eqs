#!/usr/bin/env bash

# ---------------------- Thu June 20 ----------------------
sbatch -J p0_4 --partition gpu run.sh --Ngrid 4 --ix 0 --total 4
sbatch -J p1_4 --partition gpu run.sh --Ngrid 4 --ix 1 --total 4
sbatch -J p2_4 --partition gpu run.sh --Ngrid 4 --ix 2 --total 4
sbatch -J p3_4 --partition gpu run.sh --Ngrid 4 --ix 3 --total 4

# ---------------------- Wed June 19 ----------------------
# sbatch -J megno4 --partition gpu run.sh 0 4
# sbatch -J megno20 --partition gpu run.sh 0 20
# sbatch -J megno80 --partition gpu run.sh 0 80
# sbatch -J model4 --partition gpu run.sh 1 4
# sbatch -J model20 --partition gpu run.sh 1 20
# sbatch -J model80 --partition gpu run.sh 1 80


