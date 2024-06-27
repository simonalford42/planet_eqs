#!/usr/bin/env bash


# sbatch -J p1_800 --partition gpu run.sh --Ngrid 800 --ix 1 --total 40 --std --compute
# sbatch -J p21_800 --partition gpu run.sh --Ngrid 800 --ix 21 --total 40 --std --compute
# sbatch -J p8_800 --partition gpu run.sh --Ngrid 800 --ix 8 --total 40 --std --compute
# sbatch -J p16_800 --partition gpu run.sh --Ngrid 800 --ix 16 --total 40 --std --compute

# sbatch -J p1_800 --partition gpu run.sh --Ngrid 800 --ix 1 --total 40 --std --compute
# sbatch -J p21_800 --partition gpu run.sh --Ngrid 800 --ix 21 --total 40 --std --compute
# sbatch -J p8_800 --partition gpu run.sh --Ngrid 800 --ix 8 --total 40 --std --compute
# sbatch -J p16_800 --partition gpu run.sh --Ngrid 800 --ix 16 --total 40 --std --compute

sbatch -J p21_1600 --partition gpu run.sh --Ngrid 1600 --ix 21 --total 80 --std --compute
sbatch -J p22_1600 --partition gpu run.sh --Ngrid 1600 --ix 22 --total 80 --std --compute
sbatch -J p23_1600 --partition gpu run.sh --Ngrid 1600 --ix 23 --total 80 --std --compute
sbatch -J p24_1600 --partition gpu run.sh --Ngrid 1600 --ix 24 --total 80 --std --compute
sbatch -J p25_1600 --partition gpu run.sh --Ngrid 1600 --ix 25 --total 80 --std --compute
sbatch -J p58_1600 --partition gpu run.sh --Ngrid 1600 --ix 58 --total 80 --std --compute
sbatch -J p59_1600 --partition gpu run.sh --Ngrid 1600 --ix 59 --total 80 --std --compute
sbatch -J p60_1600 --partition gpu run.sh --Ngrid 1600 --ix 60 --total 80 --std --compute
sbatch -J p61_1600 --partition gpu run.sh --Ngrid 1600 --ix 61 --total 80 --std --compute
sbatch -J p74_1600 --partition gpu run.sh --Ngrid 1600 --ix 74 --total 80 --std --compute
