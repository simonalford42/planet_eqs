#!/usr/bin/env bash

# -------------------- Fri 1/26 jobs --------------------------

# sbatch -J pred2 --partition gpu train.sh --f2_variant ifthen --n_predicates 10
# sbatch -J pred2 --partition gpu train.sh --f2_variant ifthen --n_predicates 1
# sbatch -J pred2 --partition gpu train.sh --f2_variant ifthen --n_predicates 100

# sbatch -J pred --partition gpu train.sh --f2_variant ifthen --n_predicates 10
# sbatch -J pred --partition gpu train.sh --f2_variant ifthen --n_predicates 1
# sbatch -J pred --partition gpu train.sh --f2_variant ifthen --n_predicates 100

# sbatch -J nobias --partition gpu train.sh --f1_variant linear --no_bias
# sbatch -J nobias1 --partition gpu train.sh --f1_variant linear --no_bias --l1_reg weights --l1_coeff 2
# sbatch -J nobias2 --partition gpu train.sh --f1_variant linear --no_bias --l1_reg weights --l1_coeff 0.2

# sbatch -J k1 --partition gpu train.sh --load 95944 --prune_f1_topk 1
# sbatch -J k2 --partition gpu train.sh --load 95944 --prune_f1_topk 2
# sbatch -J k3 --partition gpu train.sh --load 95944 --prune_f1_topk 3
# sbatch -J k5 --partition gpu train.sh --load 95944 --prune_f1_topk 5
# sbatch -J k10 --partition gpu train.sh --load 95944 --prune_f1_topk 10
# sbatch -J t-4 --partition gpu train.sh --load 95944 --prune_f1_threshold 1e-4
# sbatch -J t-3 --partition gpu train.sh --load 95944 --prune_f1_threshold 1e-3
# sbatch -J t-2 --partition gpu train.sh --load 95944 --prune_f1_threshold 1e-2
# sbatch -J t-1 --partition gpu train.sh --load 95944 --prune_f1_threshold 1e-1

# -------------------- Thu 1/25 jobs --------------------------

# sbatch -J k3 --partition gpu train.sh --load 95944 --prune_f1_topk 3
# sbatch -J k5 --partition gpu train.sh --load 95944 --prune_f1_topk 5
# sbatch -J k10 --partition gpu train.sh --load 95944 --prune_f1_topk 10
# sbatch -J t-5 --partition gpu train.sh --load 95944 --prune_f1_threshold 1e-5
# sbatch -J t-6 --partition gpu train.sh --load 95944 --prune_f1_threshold 1e-6
# sbatch -J t-4 --partition gpu train.sh --load 95944 --prune_f1_threshold 1e-4
# sbatch -J t-3 --partition gpu train.sh --load 95944 --prune_f1_threshold 1e-3
# sbatch -J t-2 --partition gpu train.sh --load 95944 --prune_f1_threshold 1e-2
# sbatch -J t-1 --partition gpu train.sh --load 95944 --prune_f1_threshold 1e-1


# etc.
# sbatch -J 1eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 1 --freeze_pysr_f2 --eval
# sbatch -J 20eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 20 --freeze_pysr_f2 --eval
# sbatch -J 14eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --freeze_pysr_f2 --eval

# sbatch -J id_sr2 --partition gpu --mem 100G --time 02:00:00 sr.sh --version 4434 --seed 1 --target f2 --time_in_hours 1

# -------------------- Tue 1/23 jobs --------------------

# sbatch -J abs1 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 1
# sbatch -J abs2 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 2
# sbatch -J abs5 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 5
# sbatch -J abs10 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 10
# sbatch -J abs50 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 50
# sbatch -J abs100 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 100

# sbatch -J id_sr2 --partition gpu --time 02:00:00 sr.sh --version 4434 --seed 1 --target f2 --time_in_hours 1

# sbatch -J 2eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 2 --freeze_pysr_f2
# sbatch -J 14eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --freeze_pysr_f2
# sbatch -J 25eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 25 --freeze_pysr_f2
# sbatch -J 60eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 60 --freeze_pysr_f2

# sbatch -J 2res --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 2 --pysr_f2_residual
# sbatch -J 14res --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --pysr_f2_residual
# sbatch -J 25res --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 25 --pysr_f2_residual
# sbatch -J 60res --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 60 --pysr_f2_residual

# sbatch -J 14_f1_fine --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --freeze_pysr_f2

# -------------------- Mon 1/22 jobs --------------------

# sbatch -J abslw015 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.15
# sbatch -J abslw005 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.005
# sbatch -J abslw02 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.2
# sbatch -J abslw05 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.5
# sbatch -J abslw07 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.7

# crashed last time...
# sbatch -J lw02_std40 --partition gpu --time 02:00:00 sr.sh --version 63524 --target f2 --time_in_hours 1 --std_percent_threshold 0.40

# crashed last time...
# sbatch -J lw02_08 --partition ellis --time 08:00:00 sr.sh --version 63524 --target f2 --time_in_hours 6
# sbatch -J lw02_24 --partition ellis --time 24:00:00 sr.sh --version 63524 --target f2 --time_in_hours 22

# wrong seed last time...
# sbatch -J id_sr --partition gpu --time 02:00:00 sr.sh --version 4434 --seed 1 --target f2 --time_in_hours 1

# -------------------- Fri 1/19/24 jobs ------------------


# sbatch -J 1eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 1

# -------------------- Thu 1/18/24 jobs ------------------

# training the residual
# sbatch -J f2_res --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_f2_residual

# sbatch -J 1eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 1
# sbatch -J 14eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14
# sbatch -J 25eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 25
# sbatch -J 60eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 60

# sbatch -J more_f2 --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl'

# sbatch -J id_sr --partition gpu --time 02:00:00 sr.sh --version 4434 --target f2 --time_in_hours 1

# sbatch -J lw02_std5 --partition gpu --time 02:00:00 sr.sh --version 63524 --target f2 --time_in_hours 1 --std_percent_threshold 0.05
# sbatch -J lw02_std10 --partition gpu --time 02:00:00 sr.sh --version 63524 --target f2 --time_in_hours 1 --std_percent_threshold 0.10
# sbatch -J lw02_std20 --partition gpu --time 02:00:00 sr.sh --version 63524 --target f2 --time_in_hours 1 --std_percent_threshold 0.20
# sbatch -J lw02_std40 --partition gpu --time 02:00:00 sr.sh --version 63524 --target f2 --time_in_hours 1 --std_percent_threshold 0.40
# sbatch -J lw02_std70 --partition gpu --time 02:00:00 sr.sh --version 63524 --target f2 --time_in_hours 1 --std_percent_threshold 0.70

# sbatch -J lw02_08 --partition ellis --time 08:00:00 sr.sh --version 63524 --target f2 --time_in_hours 7
# sbatch -J lw02_24 --partition ellis --time 24:00:00 sr.sh --version 63524 --target f2 --time_in_hours 23


# -------------------- Wed 1/11/24 jobs ------------------

# sbatch -J f2_sr2 --partition ellis --time 08:00:00 sr.sh --version 21101 --target f2 --time_in_hours 7
# sbatch -J f2_sr3 --partition ellis --time 02:00:00 sr.sh --version 21101 --target f2 --time_in_hours 1
# sbatch -J f2_sr4 --partition ellis --time 24:00:00 sr.sh --version 21101 --target f2 --time_in_hours 23

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
