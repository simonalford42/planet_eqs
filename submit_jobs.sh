#!/usr/bin/env bash

# -------------------- Wed 1/11/24 jobs ------------------

sbatch -J f2_sr2 --partition ellis --time 08:00:00 sr.sh --version 21101 --target f2 --time_in_hours 7
sbatch -J f2_sr3 --partition ellis --time 02:00:00 sr.sh --version 21101 --target f2 --time_in_hours 1
sbatch -J f2_sr4 --partition ellis --time 24:00:00 sr.sh --version 21101 --target f2 --time_in_hours 23

# sbatch -J lf2_5_1 --partition gpu train.sh --f2_depth 1 --hidden 5 --f1_variant linear --seed 1
# sbatch -J lf2_10_1 --partition gpu train.sh --f2_depth 1 --hidden 10 --f1_variant linear --seed 1
# sbatch -J lf2_20_1 --partition gpu train.sh --f2_depth 1 --hidden 20 --f1_variant linear --seed 1
# sbatch -J lf2_40_1 --partition gpu train.sh --f2_depth 1 --hidden 40 --f1_variant linear --seed 1
# sbatch -J lf2_80_1 --partition gpu train.sh --f2_depth 1 --hidden 80 --f1_variant linear --seed 1

# sbatch -J lf2_5_2 --partition gpu train.sh --f2_depth 2 --hidden 5 --f1_variant linear
# sbatch -J lf2_10_2 --partition gpu train.sh --f2_depth 2 --hidden 10 --f1_variant linear
# sbatch -J lf2_20_2 --partition gpu train.sh --f2_depth 2 --hidden 20 --f1_variant linear
# sbatch -J lf2_40_2 --partition gpu train.sh --f2_depth 2 --hidden 40 --f1_variant linear
# sbatch -J lf2_80_2 --partition gpu train.sh --f2_depth 2 --hidden 80 --f1_variant linear

# sbatch -J lf2_5_3 --partition gpu train.sh --f2_depth 3 --hidden 5 --f1_variant linear
# sbatch -J lf2_10_3 --partition gpu train.sh --f2_depth 3 --hidden 10 --f1_variant linear
# sbatch -J lf2_20_3 --partition gpu train.sh --f2_depth 3 --hidden 20 --f1_variant linear
# sbatch -J lf2_40_3 --partition gpu train.sh --f2_depth 3 --hidden 40 --f1_variant linear
# sbatch -J lf2_80_3 --partition gpu train.sh --f2_depth 3 --hidden 80 --f1_variant linear

# -------------------- Tue 1/10/24 jobs ------------------

# sbatch -J lf2_5_-1 --partition gpu train.sh --f2_depth -1 --hidden 5 --f1_variant linear
# sbatch -J lf2_10_-1 --partition gpu train.sh --f2_depth -1 --hidden 10 --f1_variant linear
# sbatch -J lf2_20_-1 --partition gpu train.sh --f2_depth -1 --hidden 20 --f1_variant linear
# sbatch -J lf2_40_-1 --partition gpu train.sh --f2_depth -1 --hidden 40 --f1_variant linear
# sbatch -J lf2_80_-1 --partition gpu train.sh --f2_depth -1 --hidden 80 --f1_variant linear

# sbatch -J lf2_5_0 --partition gpu train.sh --f2_depth 0 --hidden 5 --f1_variant linear
# sbatch -J lf2_10_0 --partition gpu train.sh --f2_depth 0 --hidden 10 --f1_variant linear
# sbatch -J lf2_20_0 --partition gpu train.sh --f2_depth 0 --hidden 20 --f1_variant linear
# sbatch -J lf2_40_0 --partition gpu train.sh --f2_depth 0 --hidden 40 --f1_variant linear
# sbatch -J lf2_80_0 --partition gpu train.sh --f2_depth 0 --hidden 80 --f1_variant linear

# sbatch -J lf2_5_1 --partition gpu train.sh --f2_depth 1 --hidden 5 --f1_variant linear
# sbatch -J lf2_10_1 --partition gpu train.sh --f2_depth 1 --hidden 10 --f1_variant linear
# sbatch -J lf2_20_1 --partition gpu train.sh --f2_depth 1 --hidden 20 --f1_variant linear
# sbatch -J lf2_40_1 --partition gpu train.sh --f2_depth 1 --hidden 40 --f1_variant linear
# sbatch -J lf2_80_1 --partition gpu train.sh --f2_depth 1 --hidden 80 --f1_variant linear

# -------------------- Mon 1/9/24 jobs -------------------

# sbatch -J z5 -t 48:00:00 --partition ellis train.sh --load 10367 --total_steps 1000000 # f2_reg = 5
# sbatch -J z10 -t 48:00:00 --partition ellis train.sh --load 14695  --total_steps 1000000 # f2_reg = 10
# sbatch -J z20 -t 48:00:00 --partition ellis train.sh --load 4  --total_steps 1000000 # f2_reg = 20
# sbatch -J z100 -t 48:00:00 --partition ellis train.sh --load 5  --total_steps 1000000 # f2_reg = 100

# sbatch -J f2_5_-1 --partition gpu train.sh --f2_depth -1 --hidden 5
# sbatch -J f2_10_-1 --partition gpu train.sh --f2_depth -1 --hidden 10
# sbatch -J f2_20_-1 --partition gpu train.sh --f2_depth -1 --hidden 20
# sbatch -J f2_40_-1 --partition gpu train.sh --f2_depth -1 --hidden 40
# sbatch -J f2_80_-1 --partition gpu train.sh --f2_depth -1 --hidden 80

# -------------------- Thu 1/4/24 jobs -------------------

# sbatch -J z5 --partition ellis train.sh --load 10367  # f2_reg = 5
# sbatch -J z10 --partition ellis train.sh --load 14695  # f2_reg = 10
# sbatch -J z20 --partition ellis train.sh --load 4  # f2_reg = 20
# sbatch -J z100 --partition ellis train.sh --load 5  # f2_reg = 100

# sbatch -J f2_10_0 --partition gpu train.sh --f2_depth 0 --hidden 10
# sbatch -J f2_10_1 --partition gpu train.sh --f2_depth 1 --hidden 10
# sbatch -J f2_5_1 --partition gpu train.sh --f2_depth 1 --hidden 5
# sbatch -J f2_5_2 --partition gpu train.sh --f2_depth 2 --hidden 5

# sbatch -J f2_sr2 --partition gpu --time 08:00:00 sr.sh --version 21101 --target f2 --time_in_hours 8

# -------------------- Wed 1/3/24 jobs -------------------

# sbatch -J f2_linear --partition ellis train.sh --f2_linear

# sbatch -J f2_20_0 --partition gpu train.sh --f2_depth 0 --hidden 20
# sbatch -J f2_20_1 --partition gpu train.sh --f2_depth 1 --hidden 20
# sbatch -J f2_40_1 --partition gpu train.sh --f2_depth 1 --hidden 40
# sbatch -J f2_40_2 --partition gpu train.sh --f2_depth 2 --hidden 40
# sbatch -J f2_80_2 --partition gpu train.sh --f2_depth 2 --hidden 80

# sbatch -J l_v_nn1 --partition gpu train.sh --f1_variant linear --latent 1
# sbatch -J l_v_nn2 --partition gpu train.sh --f1_variant linear --latent 2
# sbatch -J l_v_nn4 --partition gpu train.sh --f1_variant linear --latent 4
# sbatch -J l_v_nn8 --partition gpu train.sh --f1_variant linear --latent 8
# sbatch -J l_v_nn16 --partition gpu train.sh --f1_variant linear --latent 16
# sbatch -J l_v_nn32 --partition gpu train.sh --f1_variant linear --latent 32
# sbatch -J l_v_nn64 --partition gpu train.sh --f1_variant linear --latent 64
# sbatch -J l_v_nn128 --partition gpu train.sh --f1_variant linear --latent 128

# sbatch -J nn_v_l1 --partition gpu train.sh --latent 1
# sbatch -J nn_v_l2 --partition gpu train.sh --latent 2
# sbatch -J nn_v_l4 --partition gpu train.sh --latent 4
# sbatch -J nn_v_l8 --partition gpu train.sh --latent 8
# sbatch -J nn_v_l16 --partition gpu train.sh --latent 16
# sbatch -J nn_v_l32 --partition gpu train.sh --latent 32
# sbatch -J nn_v_l64 --partition gpu train.sh --latent 64
# sbatch -J nn_v_l128 --partition gpu train.sh --latent 128

# sbatch -J lw015 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.15
# sbatch -J lw005 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.005
# sbatch -J lw02 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.2
# sbatch -J lw05 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.5
# sbatch -J lw07 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.7

# sbatch -J l3 --partition ellis train.sh --load 19698
