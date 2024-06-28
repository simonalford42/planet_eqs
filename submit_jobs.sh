#!/usr/bin/env bash

# ------------------------------- Thu June 27 -----------------------------------

# sbatch -J sr --partition ellis --time 24:00:00 sr.sh --time_in_hours 22 --version 43139
# bash train.sh --load_f1 43139 --pysr_f2 'sr_results/20592.pkl' --pysr_f2_model_selection best --eval
# bash train.sh --load_f1 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_f2_model_selection 14 --eval
# bash train.sh --load_f1 43139 --pysr_f2 'sr_results/87545.pkl' --pysr_f2_model_selection best --eval
# bash train.sh --load_f1 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_f2_model_selection best --eval

# sbatch -J sr --partition ellis sr.sh --time_in_hours 0.05 --version 21101 --target f22

# bash train.sh --load_f1 21101 --pysr_f2 'sr_results/76025.pkl' --pysr_f2_model_selection best --eval
# bash train.sh --load_f1 21101 --pysr_f2 'sr_results/65404.pkl' --pysr_f2_model_selection best --eval
# bash train.sh --load_f1 21101 --pysr_f2 'sr_results/61400.pkl' --pysr_f2_model_selection best --eval

# sbatch -J sr --partition ellis sr.sh --time_in_hours 0.05 --version 21101 --target f2

# bash train.sh --load_f1 21101 --pysr_f2 'sr_results/61400.pkl' --pysr_f2_model_selection best --eval

sbatch -J sr --partition gpu sr.sh --time_in_hours 0.05 --version 43139 --target f2
# sbatch -J sr --partition gpu --time 2:00:00 sr.sh --time_in_hours 1 --version 43139 --target f2
# sbatch -J sr --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 43139 --target f2
# sbatch -J sr --partition gpu --time 25:00:00 sr.sh --time_in_hours 24 --version 43139 --target f2

# ------------------------------- Tue June 25 -----------------------------------

# bash train.sh --total_steps 50 --load_f1 43139 --pysr_f2 'sr_results/66312.pkl' --pysr_f2_model_selection best --eval

# sbatch -J 1eval --partition gpu train.sh --total_steps 100 --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 1 --freeze_pysr_f2 --eval
# sbatch -J 20eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 20 --freeze_pysr_f2 --eval
# sbatch -J 14eval --partition gpu train.sh --total_steps 100 --no_swag --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --freeze_pysr_f2 --eval
# sbatch -J sr --partition ellis --time 12:00:00 sr.sh --time_in_hours 7 --version 43139

# ------------------------------- Mon June 24 -----------------------------------

# sbatch -J sr --partition ellis --time 08:00:00 sr.sh --time_in_hours 7 --version 43139
# sbatch -J sr --partition ellis --time 24:00:00 sr.sh --time_in_hours 23 --version 43139

# ------------------------------- Sun June 23 -----------------------------------

# sbatch -J sr --partition ellis --time 00:05:00 sr.sh --time_in_hours 0.01 --version 24880 --residual

# ------------------------------- Fri June 21 -----------------------------------

# sbatch -J prod3 --partition gpu train.sh --f1_variant products3

# ------------------------------- Wed June 18 -----------------------------------

# sbatch -J sr --partition gpu --time 00:01:00 sr.sh --time_in_hours 0.01 --version 24880
# sbatch -J sr --partition ellis --time 00:01:00 sr.sh --time_in_hours 0.001 --version 24880 --sr_residual --previous_sr_path 'sr_results/8092.pkl'

# ------------------------------- Tue June 18 -----------------------------------

# sbatch -J sr --partition ellis --time 00:15:00 sr.sh --time_in_hours 0.01 --version 24880
# sbatch -J sr --partition ellis --time 00:15:00 sr.sh --time_in_hours 0.01 --version 24880 --sr_residual --previous_sr_path 'sr_results/8092.pkl'

# ------------------------------- Thu June 6 -----------------------------------

# sbatch -J pr10swag prune_train.sh --latent 10 --version 24880

# sbatch -J sr --partition gpu --time 09:00:00 sr.sh --time_in_hours 8 --version 24880
# sbatch -J sr2 --partition gpu --time 09:00:00 sr.sh --time_in_hours 2 --version 24880
# sbatch -J sr1 --partition gpu --time 09:00:00 sr.sh --time_in_hours 1 --version 24880

# sbatch -J s1test --partition gpu prune_train.sh --latent 10 --seed 1 --total_steps 1000
# sbatch -J pr10s1 --partition gpu prune_train.sh --latent 10 --seed 1
# sbatch -J pr10s2 --partition gpu prune_train.sh --latent 10 --seed 2
# sbatch -J prod10 --partition gpu prune_train.sh --latent 10 --f1_variant products2
# sbatch -J prod10s1 --partition gpu prune_train.sh --latent 10 --f1_variant products2 --seed 1
# sbatch -J ifthen10 --partition gpu prune_train.sh --latent 10 --f2_variant ifthen2
# sbatch -J ifthen10s1 --partition gpu prune_train.sh --latent 10 --f2_variant ifthen2 --seed 1

# ------------------------------- Wed June 5 -----------------------------------

# sbatch -J sr --partition gpu sr.sh --time_in_hours 8 --version 24880

# sbatch -J pr10s1 --partition gpu prune_train.sh --latent 10 --seed 1
# sbatch -J pr10s2 --partition gpu prune_train.sh --latent 10 --seed 2
# sbatch -J ifthen10 --partition gpu prune_train.sh --latent 10 --f1_variant products2
# sbatch -J ifthen10s1 --partition gpu prune_train.sh --latent 10 --f1_variant products2 --seed 1
# sbatch -J prod10 --partition gpu prune_train.sh --latent 10 --f2_variant ifthen2
# sbatch -J prod10s1 --partition gpu prune_train.sh --latent 10 --f2_variant ifthen2 --seed 1

# ------------------------------- Fri May 31 -----------------------------------

# sbatch -J pr80_5 --partition gpu prune_train.sh --latent 80 --prune_f1_topn 5
# sbatch -J pr80_10 --partition gpu prune_train.sh --latent 80 --prune_f1_topn 10
# sbatch -J pr80_20 --partition gpu prune_train.sh --latent 80 --prune_f1_topn 20
# sbatch -J pr20_5 --partition gpu prune_train.sh --latent 20 --prune_f1_topn 5
# sbatch -J pr20_10 --partition gpu prune_train.sh --latent 20 --prune_f1_topn 10
# sbatch -J pr20 --partition gpu prune_train.sh --latent 20
# sbatch -J pr10 --partition gpu prune_train.sh --latent 10
# sbatch -J pr5 --partition gpu prune_train.sh --latent 5
# sbatch -J prod20 --partition gpu prune_train.sh --f1_variant products2 --latent 20
# sbatch -J ifthen2 --partition gpu prune_train.sh --f2_variant ifthen2 --latent 20

# ------------------------------- Thu May 30 -----------------------------------

# sbatch -J pr80_5 --partition gpu prune_train.sh --latent 80 --prune_f1_topn 5
# sbatch -J pr80_10 --partition gpu prune_train.sh --latent 80 --prune_f1_topn 10
# sbatch -J pr80_20 --partition gpu prune_train.sh --latent 80 --prune_f1_topn 20
# sbatch -J pr20_5 --partition gpu prune_train.sh --latent 20 --prune_f1_topn 5
# sbatch -J pr20_10 --partition gpu prune_train.sh --latent 20 --prune_f1_topn 10
# sbatch -J pr20 --partition gpu prune_train.sh --latent 20
# sbatch -J pr10 --partition gpu prune_train.sh --latent 10
# sbatch -J pr5 --partition gpu prune_train.sh --latent 5
# sbatch -J prod20 --partition gpu prune_train.sh --f1_variant products2 --latent 20
# sbatch -J ifthen2 --partition gpu prune_train.sh --f2_variant ifthen2 --latent 20

# ------------------------------- Wed May 29 -----------------------------------

# sbatch -J ifthen2 --partition gpu prune_train.sh --f2_variant ifthen2

# sbatch -J pr80_5 --partition gpu prune_train.sh --latent 80 --prune_f1_topn 5
# sbatch -J pr80_10 --partition gpu prune_train.sh --latent 80 --prune_f1_topn 10
# sbatch -J pr80_20 --partition gpu prune_train.sh --latent 80 --prune_f1_topn 20
# sbatch -J pr20_5 --partition gpu prune_train.sh --latent 20 --prune_f1_topn 5
# sbatch -J pr20_10 --partition gpu prune_train.sh --latent 20 --prune_f1_topn 10
# sbatch -J pr20 --partition gpu prune_train.sh --latent 20
# sbatch -J pr10 --partition gpu prune_train.sh --latent 10
# sbatch -J pr5 --partition gpu prune_train.sh --latent 5
# sbatch -J prod20 --partition gpu prune_train.sh --f1_variant products2 --latent 20

# ------------------------------- Tue May 28 -----------------------------------

# sbatch -J pr80_5 --partition gpu prune_train.sh --latent 80 --prune_f1_topn 5
# sbatch -J pr80_10 --partition gpu prune_train.sh --latent 80 --prune_f1_topn 10
# sbatch -J pr80_20 --partition gpu prune_train.sh --latent 80 --prune_f1_topn 20
# sbatch -J pr20_5 --partition gpu prune_train.sh --latent 20 --prune_f1_topn 5
# sbatch -J pr20_10 --partition gpu prune_train.sh --latent 20 --prune_f1_topn 10
# sbatch -J pr20 --partition gpu prune_train.sh --latent 20
# sbatch -J pr10 --partition gpu prune_train.sh --latent 10
# sbatch -J pr5 --partition gpu prune_train.sh --latent 5
# sbatch -J prod20 --partition gpu prune_train.sh --f1_variant products2 --latent 20

# ------------------------------- Mon May 27 -----------------------------------

# sbatch -J pruned --partition gpu prune_train.sh
# sbatch -J pr_prod --partition gpu prune_train.sh --f1_variant products2

# debugging TypeError: non-boolean (Float32) used in boolean context
# parser.add_argument('--target', type=str, default='f1', choices=['f1', 'f2', 'f2_ifthen', 'f2_direct'])
# sbatch -J direct_sr --partition gpu --mem 100G --time 00:30:00 sr.sh --time_in_hours 0.1 --target f2_direct --version 6364

# ---------------- Fri May 24 ------------------

# sbatch -J direct_sr --partition gpu --mem 100G --time 02:00:00 sr.sh --time_in_hours 1 --target f2_direct --version 6364

# sbatch -J pruned --partition gpu prune_train.sh
# sbatch -J pruned_prod --partition gpu prune_train.sh --f1_variant products2

# ---------------- Thu May 16 ------------------

# sbatch -J fixvar_id_lin --partition gpu train.sh --f1_variant identity --f2_variant linear --fix_variance
# sbatch -J fixvar_lin --partition gpu train.sh --f1_variant linear --fix_variance
# sbatch -J fixvar_lin_lin --partition gpu train.sh --f1_variant linear --f2_variant linear --fix_variance

# sbatch -J direct_sr --partition ellis --mem 100G --time 02:00:00 sr.sh --time-in-hours 1 --target direct --version 6364

# ------------------------- Wed Apr 24 --------------------------

# sbatch -J pure_sr --partition gpu --mem 100G --time 00:10:00 sr.sh --time-in-hours 0.1

# sbatch -J linear --partition gpu train.sh --f1_variant linear --run_swag
# sbatch -J mlp --partition gpu train.sh --f1_variant mlp --run_swag
# sbatch -J true_lin --partition gpu train.sh --f1_variant identity --f2_variant linear --run_swag


# ------------------------- Thu Mar 21 --------------------------

# sbatch -J sr_bimt6 --partition ellis --mem 200G --time 08:00:00 sr.sh --version 23219 --target f2 --time_in_hours 1
# sbatch -J sr_bimt1 --partition ellis --mem 200G --time 08:00:00 sr.sh --version 23219 --target f2 --time_in_hours 6

# sbatch -J sr_base6 --partition gpu --mem 200G --time 08:00:00 sr.sh --version 12646 --target f2 --time_in_hours 1
# sbatch -J sr_base1 --partition gpu --mem 200G --time 08:00:00 sr.sh --version 12646 --target f2 --time_in_hours 6

# sbatch -J 14res --partition gpu train.sh --load 21101 --pysr_f2 sr_results/hall_of_fame_f2_21101_0_1.pkl --pysr_model_selection 14 --f2_variant pysr_residual --l1_reg f2_weights --l1_coeff 2 --run_swag

# sbatch -J 14_nores --partition gpu train.sh --load 21101 --pysr_f2 sr_results/hall_of_fame_f2_21101_0_1.pkl --pysr_model_selection 14 --run_swag --total_steps 0


# ---------------------- Wed Mar 20 ---------------------

# sbatch -J bimt --partition ellis train.sh --f1_variant linear --f2_variant bimt

# ---------------------- Tue Mar 19 ---------------------

# sbatch -J bimt --partition ellis train.sh --f1_variant linear --f2_variant bimt
# sbatch -J baseline --partition ellis train.sh --f1_variant linear

# sbatch -J prune2 --partition gpu prune_train.sh --prune_f1_topk 2 --prune_f1_topn 5 --latent 40
# sbatch -J prune2 --partition gpu prune_train.sh --prune_f1_topk 2 --prune_f1_topn 10 --latent 80
# sbatch -J prune2 --partition gpu prune_train.sh --prune_f1_topk 2 --prune_f1_topn 5 --latent 80
# sbatch -J prune2 --partition gpu prune_train.sh --prune_f1_topk 2 --prune_f1_topn 10 --latent 160

# ---------------------- Mon Mar 18 ---------------------

# sbatch -J prune2 --partition gpu prune_train.sh --prune_f1_topk 2 --latent 10
# sbatch -J prune2 --partition gpu prune_train.sh --prune_f1_topk 2 --latent 5
# sbatch -J prune2 --partition gpu prune_train.sh --prune_f1_topk 2 --prune_f1_topn 10 --latent 40
# sbatch -J prune2 --partition gpu prune_train.sh --prune_f1_topk 2 --prune_f1_topn 10 --latent 20
# sbatch -J prune2 --partition gpu prune_train.sh --prune_f1_topk 2 --prune_f1_topn 10
# sbatch -J prune2 --partition gpu prune_train.sh --prune_f1_topk 2 --prune_f1_topn 5

# ---------------------- Fri Mar 15 ---------------------

# sbatch -J prune --partition gpu prune_train.sh --prune_f1_topk 1
# sbatch -J prune --partition gpu prune_train.sh --prune_f1_topk 2
# sbatch -J prune --partition gpu prune_train.sh --prune_f1_topk 3

# ---------------------- Thu Mar 7 ----------------------

# sbatch -J ifthen_pysr2 --partition ellis --mem 200G --time 08:00:00 sr.sh --version 42423 --target f2_ifthen --time_in_hours 1
# sbatch -J ifthen_pysr2 --partition gpu --mem 200G --time 08:00:00 sr.sh --version 42423 --target f2_ifthen --time_in_hours 6

# sbatch -J prod2 --partition gpu train.sh --f1_variant products2 --l1_reg weights --l1_coeff 2

# sbatch -J pruneprod --partition gpu train.sh --load 95944 --prune_f1_topk 2 --f1_variant pruned_products --l1_reg weights --l1_coeff 2

# sbatch -J prune --partition gpu prune_train.sh --total_steps 300000
# sbatch -J prune --partition gpu prune_train.sh --total_steps 150000

# --------------- Fri Mar 1 ------------------------

# sbatch -J ifthen_pysr2 --partition ellis --mem 200G --time 08:00:00 sr.sh --version 42423 --target f2_ifthen --time_in_hours 1
# sbatch -J ifthen_pysr2 --partition gpu --mem 200G --time 08:00:00 sr.sh --version 42423 --target f2_ifthen --time_in_hours 6

# sbatch -J prod2 --partition gpu train.sh --f1_variant products2
# sbatch -J prod2 --partition gpu train.sh --f1_variant products2 --l1_reg weights --l1_coeff 0.2

# sbatch -J pruneprod --partition gpu train.sh --load 95944 --prune_f1_topk 2 --f1_variant pruned_products

# sbatch -J f2l2_3 --partition ellis train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --f2_variant pysr_residual --l1_reg both_weights --l1_coeff 2 --seed 2 --freeze_f1
# sbatch -J f2l2_3 --partition ellis train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --f2_variant pysr_residual --freeze_f1

# ---------------- Thu Feb 22 ------------------------
# got the same standard accuracy
# sbatch -J f2l2_3 --partition ellis train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --f2_variant pysr_residual --l1_reg both_weights --l1_coeff 2 --seed 2

# sbatch -J prod_lr --partition ellis train.sh --f1_variant products --lr 1e-4
# sbatch -J reg_lr --partition ellis train.sh --lr 1e-4
# sbatch -J prod_lr2 --partition gpu train.sh --f1_variant products --lr 1e-5
# sbatch -J reg_lr2 --partition gpu train.sh --lr 1e-5

# sbatch -J it3 --partition gpu train.sh --f2_variant ifthen2 --f2_depth 0
# sbatch -J it3 --partition gpu train.sh --f2_variant ifthen2 --l1_coeff 0.2
# sbatch -J it3 --partition gpu train.sh --f2_variant ifthen2

# sbatch -J it_n --partition gpu train.sh --f2_variant ifthen2 --n_predicates 1 --f2_depth 0
# sbatch -J it_n --partition gpu train.sh --f2_variant ifthen2 --n_predicates 2 --f2_depth 0
# sbatch -J it_n --partition gpu train.sh --f2_variant ifthen2 --n_predicates 5 --f2_depth 0
# sbatch -J it_n --partition gpu train.sh --f2_variant ifthen2 --n_predicates 10 --f2_depth 0
# sbatch -J it_n --partition gpu train.sh --f2_variant ifthen2 --n_predicates 20 --f2_depth 0
# sbatch -J it_n --partition gpu train.sh --f2_variant ifthen2 --n_predicates 100 --f2_depth 0

# sbatch -J it_l --partition gpu train.sh --f2_variant ifthen2 --n_predicates 25 --f2_depth -1
# sbatch -J it_l --partition gpu train.sh --f2_variant ifthen2 --n_predicates 100 --f2_depth -1


# --------------- Wed Feb 21 --------------------------

# sbatch -J reg_h20 --partition gpu train.sh --hidden_dim 20
# sbatch -J f2l2_2 --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --f2_variant pysr_residual --l1_reg both_weights --l1_coeff 2 --seed 1

# sbatch -J prod --partition ellis train.sh --f1_variant products
# sbatch -J prod --partition ellis train.sh --f1_variant products --l1_reg weights --l1_coeff 0.2
# sbatch -J prod --partition ellis train.sh --f1_variant products --l1_reg weights --l1_coeff 2 --seed 1
# sbatch -J prod --partition gpu train.sh --f1_variant products --l1_reg weights --l1_coeff 10

# ------------------- Mon Feb 19 -------------------------
# sbatch -J f2l2_s2 --partition gpu train.sh --l1_reg both_weights --l1_coeff 0.2
# sbatch -J f2l2_s02 --partition gpu train.sh --l1_reg both_weights --l1_coeff 0.2 --f2_depth 0
# sbatch -J f2l2_s-12 --partition gpu train.sh --l1_reg both_weights --l1_coeff 0.2 --f2_depth -1
# sbatch -J f2l2_s0 --partition gpu train.sh --f2_depth 0
# sbatch -J f2l2_s-1 --partition gpu train.sh --f2_depth -1

# sbatch -J new_it2 --partition ellis train.sh --f2_variant ifthen2
# sbatch -J new_it2 --partition ellis train.sh --f2_variant ifthen2 --l1_reg both_weights --l1_coeff 0.2
# sbatch -J new_it2 --partition gpu train.sh --f2_variant ifthen2 --l1_reg both_weights --l1_coeff 2
# sbatch -J new_it2 --partition gpu train.sh --f2_variant ifthen2 --f2_depth 0
# sbatch -J new_it2 --partition gpu train.sh --f2_variant ifthen2 --f2_depth -1

# sbatch -J f2l2_p --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --f2_variant pysr_residual --l1_reg both_weights --l1_coeff 2
# sbatch -J f2l2_p --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --f2_variant pysr_residual --l1_reg both_weights --l1_coeff 0.2
# sbatch -J f2l2_p0 --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --f2_variant pysr_residual --l1_reg both_weights --l1_coeff 0.2 --f2_depth 0
# sbatch -J f2l2_p-1 --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --f2_variant pysr_residual --l1_reg both_weights --l1_coeff 0.2 --f2_depth -1
# sbatch -J f2h20_p --partition gpu train.sh - a-load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --f2_variant pysr_residual --l1_reg both_weights --l1_coeff 0.2 --hidden_dim 20

# ------------------- Fri Feb 16 -------------------------
# sbatch -J ifthen --partition ellis train.sh --f2_variant ifthen
# sbatch -J ifthen2 --partition ellis train.sh --f2_variant ifthen2
# sbatch -J ifthen --partition gpu train.sh --f2_variant ifthen --lr 1e-4
# sbatch -J ifthen --partition gpu train.sh --f2_variant ifthen --lr 1e-4

# sbatch -J f2l2_s --partition gpu train.sh --l1_reg both_weights --l1_coeff 2
# sbatch -J f2l2_s0 --partition gpu train.sh --l1_reg both_weights --l1_coeff 2 --f2_depth 0
# sbatch -J f2l2_s-1 --partition gpu train.sh --l1_reg both_weights --l1_coeff 2 --f2_depth -1


# ------------------- Thu Feb 15 -------------------------
# sbatch -J rand_fr --partition ellis train.sh --f1_variant random_frozen --no_bias --seed 0
# sbatch -J rand_fr --partition ellis train.sh --f1_variant random_frozen --no_bias --seed 1

# sbatch -J prod --partition ellis train.sh --f1_variant products --l1_reg weights --l1_coeff 2

# sbatch -J bimt --partition gpu train.sh --f1_variant bimt
# sbatch -J ifthen --partition gpu train.sh --f2_variant ifthen
# sbatch -J ifthen2 --partition gpu train.sh --f2_variant ifthen2

# sbatch -J 2res --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 2 --f2_variant pysr_residual
# sbatch -J 14res --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --f2_variant pysr_residual
# sbatch -J 25res --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 25 --f2_variant pysr_residual
# sbatch -J 60res --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 60 --f2_variant pysr_residual

# sbatch -J f2l2 --partition gpu train.sh --load 21101 --f2_variant 'new'
# sbatch -J f2l2 --partition gpu train.sh --load 21101 --f2_variant 'new' --l1_reg f2_weights --l1_coeff 0.2
# sbatch -J f2l2 --partition gpu train.sh --load 21101 --f2_variant 'new' --l1_reg f2_weights --l1_coeff 2

# sbatch -J f2l2 --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --f2_variant pysr_residual --l1_reg f2_weights --l1_coeff 0.2
# sbatch -J f2l2 --partition gpu train.sh --load 21101 --pysr_f2 'sr_results/hall_of_fame_f2_21101_0_1.pkl' --pysr_model_selection 14 --f2_variant pysr_residual --l1_reg f2_weights --l1_coeff 2



# -------------------- Tue 1/30 jobs --------------------------

# sbatch -J prune --partition gpu prune_train.sh --no_bias --no_swag --seed 1

# sbatch -J db1 --partition gpu train.sh --load 95944 --prune_f1_threshold -1 --pruned_debug 1 --no_swag --total_steps 50000
# sbatch -J db2 --partition gpu train.sh --load 95944 --prune_f1_threshold -1 --pruned_debug 2 --no_swag --total_steps 50000
# sbatch -J db3 --partition gpu train.sh --load 95944 --prune_f1_threshold -1 --pruned_debug 3 --no_swag --total_steps 50000
# sbatch -J db4 --partition gpu train.sh --load 95944 --prune_f1_threshold -1 --pruned_debug 4 --no_swag --total_steps 50000
# sbatch -J db5 --partition gpu train.sh --load 95944 --prune_f1_threshold -1 --pruned_debug 5 --no_swag --total_steps 50000
# sbatch -J base --partition gpu train.sh --load 95944 --no_swag --total_steps 50000

# sbatch -J 2lat5 --partition gpu train.sh --f1_variant linear --no_bias --l1_reg weights --l1_coeff 2 --latent 5
# sbatch -J 2lat5 --partition gpu train.sh --f1_variant linear --no_bias --l1_reg weights --l1_coeff 2 --latent 5 --seed 1
# sbatch -J 2lat5 --partition gpu train.sh --f1_variant linear --no_bias --l1_reg weights --l1_coeff 0.2 --latent 5

# sbatch -J lin22 --partition gpu train.sh --f1_variant linear --f2_variant linear
# sbatch -J lin22 --partition gpu --mem 100G train.sh --f1_variant mlp --f2_variant linear

# sbatch -J f2_reg02 --partition gpu --mem 100G train.sh --l1_reg f2_weights --l1_coeff 0.2

# sbatch -J rand --partition gpu train.sh --f1_variant random --no_bias
# sbatch -J rand_fr --partition gpu train.sh --f1_variant random_frozen --no_bias


# -------------------- Mon 1/29 jobs --------------------------

# sbatch -J lat5 --partition gpu train.sh --f1_variant linear --no_bias --l1_reg weights --l1_coeff 2 --latent 5
# sbatch -J lat5 --partition gpu train.sh --f1_variant linear --no_bias --l1_reg weights --l1_coeff 2 --latent 5 --seed 1
# sbatch -J lat5 --partition gpu train.sh --f1_variant linear --no_bias --l1_reg weights --l1_coeff 0.2 --latent 5
# sbatch -J lat10 --partition gpu train.sh --f1_variant linear --no_bias --l1_reg weights --l1_coeff 2 --latent 10
# sbatch -J lat10 --partition gpu train.sh --f1_variant linear --no_bias --l1_reg weights --l1_coeff 2 --latent 10 --seed 1
# sbatch -J lat10 --partition gpu train.sh --f1_variant linear --no_bias --l1_reg weights --l1_coeff 0.2 --latent 10

# sbatch -J t_all --partition gpu train.sh --load 95944 --prune_f1_threshold -1
# sbatch -J t_special_all_mask_from_scratch --partition gpu train.sh --load 95944 --prune_f1_threshold -2
# sbatch -J t_fine_tune --partition gpu train.sh --load 95944

# sbatch -J lin2 --partition gpu train.sh --f1_variant linear --f2_variant linear
# sbatch -J lin2 --partition gpu train.sh --f1_variant mlp --f2_variant linear

# sbatch -J f2_reg02 --partition gpu train.sh --l1_reg f2_weights --l1_coeff 0.2
# sbatch -J f2_reg1 --partition gpu train.sh --l1_reg f2_weights --l1_coeff 1
# sbatch -J f2_reg2 --partition gpu train.sh --l1_reg f2_weights --l1_coeff 2

# sbatch -J k1 --partition gpu train.sh --load 32605 --prune_f1_topk 1
# sbatch -J k2 --partition gpu train.sh --load 32605 --prune_f1_topk 2
# sbatch -J k3 --partition gpu train.sh --load 32605 --prune_f1_topk 3
# sbatch -J k5 --partition gpu train.sh --load 32605 --prune_f1_topk 5
# sbatch -J k10 --partition gpu train.sh --load 32605 --prune_f1_topk 10

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
