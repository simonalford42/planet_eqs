#!/usr/bin/env bash

# June 17 2025
sbatch -J f1 --partition ellis sr.sh pure_sr.py --loss_fn ll
sbatch -J f1_mse --partition ellis sr.sh pure_sr.py --loss_fn mse

# June 16 2025
# sbatch -J f2_0 --partition gpu f2_prune_train.sh --prune_f2_topk 0
# sbatch -J f2_1 --partition gpu f2_prune_train.sh --prune_f2_topk 1
# sbatch -J f2_2 --partition gpu f2_prune_train.sh --prune_f2_topk 2
# sbatch -J f2_5 --partition gpu f2_prune_train.sh --prune_f2_topk 5
# sbatch -J f2_10 --partition gpu f2_prune_train.sh --prune_f2_topk 10
# sbatch -J f2_15 --partition gpu f2_prune_train.sh --prune_f2_topk 15
# sbatch -J f2_20 --partition gpu f2_prune_train.sh --prune_f2_topk 20

# June 12 2025
# sbatch -J class_dir --partition gpu sr.sh sr.py --time_in_hours 8 --version 24880 --target f2_direct --loss_fn ll --max_size 45 --seed 5
# sbatch -J class_nn2 --partition gpu sr.sh sr.py --time_in_hours 8 --version 24880 --target f2 --loss_fn ll --max_size 45 --seed 5
# sbatch -J sr15 --partition gpu sr.sh sr.py --time_in_hours 8 --version 76432 --target f2 --max_size 45 --seed 5
# sbatch -J sr14 --partition gpu sr.sh sr.py --time_in_hours 8 --version 24880 --target f2 --max_size 45 --seed 5
# sbatch -J sr13 --partition gpu sr.sh sr.py --time_in_hours 8 --version 24880 --target f2 --max_size 45 --seed 5

# June 11 2025
# sbatch -J t13_2 --partition ellis prune_train.sh --max_pred 13
# sbatch -J t14_2 --partition gpu prune_train.sh --max_pred 14
# sbatch -J t15 --partition gpu prune_train.sh --max_pred 15

# May 22 2025
# sbatch -J f1id_45 --partition gpu -t 01:00:00 run.sh calc_rmse.py --version 28114 --pysr_version 93890 --eval_type pysr --dataset all
# sbatch -J f1id_60 --partition gpu -t 01:00:00 run.sh calc_rmse.py --version 28114 --pysr_version 50620 --eval_type pysr --dataset all
# sbatch -J f1id_45sp --partition gpu -t 01:00:00 run.sh calc_rmse.py --version 28114 --pysr_version 41564 --eval_type pysr --dataset all
# sbatch -J f1id_60sp --partition gpu -t 01:00:00 run.sh calc_rmse.py --version 28114 --pysr_version 32888 --eval_type pysr --dataset all

# May 21 2025
# sbatch -J f1id_45 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 28114 --target f2_direct --seed 42 --max_size 45
# sbatch -J f1id_60 --partition ellis -t 09:00:00 sr.sh --time_in_hours 8 --version 28114 --target f2_direct --seed 42 --max_size 60
# sbatch -J f1id_45sp --partition ellis -t 09:00:00 sr.sh --time_in_hours 8 --version 28114 --target f2_direct --seed 42 --max_size 45 --loss_fn ll
# sbatch -J f1id_60sp --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 28114 --target f2_direct --seed 42 --max_size 60 --loss_fn ll

# May 7 2025
# sbatch -J 20_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 12318 --target f2 --seed 20
# sbatch -J 20_2 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 13535 --target f2 --seed 20
# sbatch -J 20_3 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 10970 --target f2 --seed 20
# sbatch -J 19_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 12318 --target f2 --seed 19
# sbatch -J 19_2 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 13535 --target f2 --seed 19
# sbatch -J 19_3 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 10970 --target f2 --seed 19
# sbatch -J 18_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 12318 --target f2 --seed 18
# sbatch -J 18_2 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 13535 --target f2 --seed 18
# sbatch -J 18_3 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 10970 --target f2 --seed 18
# sbatch -J f1id_60 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --target f2_direct --seed 0 --version 28114 --max_size 60
# sbatch -J 248_45 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --target f2_direct --seed 0 --version 24880 --max_size 45
# sbatch -J 123_45 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --target f2 --seed 0 --version 12318 --max_size 45
# sbatch -J 135_45 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --target f2 --seed 0 --version 13535 --max_size 45
# sbatch -J 109_45 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --target f2 --seed 0 --version 10970 --max_size 45

# run f1_id with

# May 5 2025
# sbatch -J 2_8 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2_direct --seed 2
# sbatch -J 3_8 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2_direct --seed 3
# sbatch -J 4_24 --partition gpu -t 25:00:00 sr.sh --time_in_hours 24 --version 24880 --target f2_direct --seed 4
# sbatch -J 5_24 --partition ellis -t 25:00:00 sr.sh --time_in_hours 24 --version 24880 --target f2_direct --seed 5
# sbatch -J 6_48 --partition gpu -t 49:00:00 sr.sh --time_in_hours 48 --version 24880 --target f2_direct --seed 6
# sbatch -J 7_48 --partition ellis -t 49:00:00 sr.sh --time_in_hours 48 --version 24880 --target f2_direct --seed 7
# sbatch -J 8_8f2 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --seed 8
# sbatch -J 9_8f2 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --seed 9
# sbatch -J 10_8f2 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --seed 10
# sbatch -J 11_8f2 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --seed 11

# May 4 2025
# sbatch -J rmse_k2 --partition ellis -t 02:20:00 run.sh calc_rmse.py --version 22649 --eval_type nn --dataset test
# sbatch -J rmse3 --partition ellis -t 02:20:00 run.sh calc_rmse.py --version 74649 --pysr_version 49636 --eval_type pysr --dataset test
# sbatch -J rmse_k0 --partition gpu -t 02:20:00 run.sh calc_rmse.py --version 4014 --eval_type nn

# May 2 2025

# evaluate rmse for new f2 linear runs
# sbatch -J rmse_k20 --partition gpu -t 02:20:00 run.sh calc_rmse.py --version 8880 --eval_type nn
# sbatch -J rmse_k10 --partition gpu -t 02:20:00 run.sh calc_rmse.py --version 22697 --eval_type nn
# sbatch -J rmse_k5 --partition gpu -t 02:20:00 run.sh calc_rmse.py --version 13523 --eval_type nn

# May 1 2025

# new f2 linear runs
# sbatch -J k20_f2lin train.sh --load_f1_f2 25646 --total_steps 150000 --f2_variant linear --mse_loss
# sbatch -J k10_f2lin train.sh --load_f1_f2 25646 --total_steps 150000 --f2_variant linear --prune_f2_topk 10 --mse_loss
# sbatch -J k5_f2lin train.sh --load_f1_f2 25646 --total_steps 150000 --f2_variant linear --prune_f2_topk 5 --mse_loss
# sbatch -J k2_f2lin --partition gpu train.sh --load_f1_f2 25646 --total_steps 150000 --f2_variant linear --prune_f2_topk 2 --mse_loss
# sbatch -J k0_f2lin --partition gpu train.sh --load_f1_f2 25646 --total_steps 150000 --f2_variant linear --prune_f2_topk 0 --mse_loss

# calc RMSe for topk jobs
# sbatch -J rmse3 --partition ellis -t 02:20:00 run.sh calc_rmse.py --version 74649 --pysr_version 49636 --eval_type pysr
# sbatch -J rmse4 --partition ellis -t 02:20:00 run.sh calc_rmse.py --version 11566 --pysr_version 94842 --eval_type pysr
# sbatch -J rmse5 --partition ellis -t 02:20:00 run.sh calc_rmse.py --version 72646 --pysr_version 42503 --eval_type pysr

# calculate f2 linear results
# sbatch -J f2_lin20 --partition gpu -t 01:20:00 run.sh calc_rmse.py --version 2702 --eval_type nn
# sbatch -J f2_lin10 --partition gpu -t 01:20:00 run.sh calc_rmse.py --version 13529 --eval_type nn
# sbatch -J f2_lin5 --partition gpu -t 01:20:00 run.sh calc_rmse.py --version 7307 --eval_type nn
# sbatch -J f2_lin2 --partition gpu -t 01:20:00 run.sh calc_rmse.py --version 22160 --eval_type nn

# April 30 2025

# new topk sr runs
# sbatch -J topk_3 --partition ellis -t 09:00:00 sr.sh --time_in_hours 8 --version 74649 --target f2_direct --seed 0
# sbatch -J topk_4 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 11566 --target f2_direct --seed 0
# sbatch -J topk_5 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 72646 --target f2_direct --seed 0

# April 28, 2025

# direct mse SR for ft networks
# sbatch -J ftfv_dir --partition ellis -t 09:00:00 sr.sh --time_in_hours 8 --version 10970 --target f2_direct --seed 0
# sbatch -J ftfv_f2 --partition ellis -t 09:00:00 sr.sh --time_in_hours 8 --version 10970 --target f2 --seed 0

#### calculating RMSE values for different things
# nn's
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --version 24880 --eval_type nn
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --version 4590 --eval_type nn
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --version 86952 --eval_type nn
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --version 12318 --eval_type nn
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --version 10970 --eval_type nn

# petit
# sbatch -J rmse --partition gpu -t 12:00:00 run.sh calc_rmse.py --eval_type petit

# pure sr
# sbatch -J rmse --partition gpu -t 12:00:00 run.sh calc_rmse.py --eval_type pure_sr --pysr_version 83941
# sbatch -J rmse --partition gpu -t 12:00:00 run.sh calc_rmse.py --eval_type pure_sr --pysr_version 72420

# f1 identity
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --eval_type f1_identity --version 28114 --pysr_version 9054
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --eval_type f1_identity --version 28114 --pysr_version 23758

### possible outcomes
# mse nn's
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --eval_type pysr --version 4590 --pysr_version 65599
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --eval_type pysr --version 86952 --pysr_version 55106
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --eval_type pysr --version 4590 --pysr_version 40403
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --eval_type pysr --version 86952 --pysr_version 66953

# fine-tuned mse nn's
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --eval_type pysr --version 12318 --pysr_version 86055
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --eval_type pysr --version 10970 --pysr_version 4929
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --eval_type pysr --version 12318 --pysr_version 22271
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --eval_type pysr --version 10970 --pysr_version 39675

# special NN but mse/direct
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --eval_type pysr --version 24880 --pysr_version 93102

# special loss for everything
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --eval_type pysr --version 24880 --pysr_version 11003
# sbatch -J rmse --partition gpu -t 02:20:00 run.sh calc_rmse.py --eval_type pysr --version 24880 --pysr_version 79364


# April 28, 2025

# pure SR with mse or special loss
# sbatch -J pure --partition ellis -t 09:00:00 run.sh pure_sr.py --time-in-hours 8 --seed 0 --loss_fn mse
# sbatch -J pure2 --partition gpu -t 09:00:00 run.sh pure_sr.py --time-in-hours 8 --seed 0 --loss_fn special

# new f1 identity runs
# sbatch -J f1id_ll --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --target f2_direct --loss_fn ll --seed 0 --version 28114

# fine-tuned fixvar mse loss
# sbatch -J ftfv_24880 --partition gpu train.sh --load_f1 24880 --total_steps 150000 --seed 0 --fix_variance

# running SR for fine-tuned NN's
# sbatch -J ft_mse_f2_mse --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 12318 --target f2 --seed 0
# sbatch -J ft_fv_f2_mse --partition ellis -t 09:00:00 sr.sh --time_in_hours 8 --version 10970 --target f2 --seed 0
# sbatch -J ft_mse_direct_mse --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 12318 --target f2_direct --seed 0
# sbatch -J ft_fv_direct_mse --partition ellis -t 09:00:00 sr.sh --time_in_hours 8 --version 10970 --target f2_direct --seed 0

# April 23, 2025

# fine tune 24880 for mse loss
# sbatch -J ft_24880 --partition ellis train.sh --load_f1 24880 --total_steps 150000 --seed 0 --mse_loss
# sbatch -J ftf_24880 --partition ellis train.sh --load_f1 24880 --total_steps 150000 --seed 0 --mse_loss --freeze_f1

# run sr.py with special loss function
# sbatch -J ll --partition ellis -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2_direct --seed 0 --loss_fn ll

# versions=(4590 4590 84963 84963 86952 86952)
# pysr_versions=(40403 65599 77075 57131 66953 55106)

# for i in "${!versions[@]}"; do
#     version=${versions[$i]}
#     pysr_version=${pysr_versions[$i]}
#     sbatch -J c --partition gpu -t 00:20:00 run.sh calc_rmse.py --version $version --pysr_version $pysr_version --train
#     sbatch -J c --partition gpu -t 00:20:00 run.sh calc_rmse.py --version $version --pysr_version $pysr_version
# done

# running pure SR on the mse-loss or fix-variance trained NN's
# sbatch -J fixv --partition ellis -t 09:00:00 sr.sh --time_in_hours 8 --version 86952 --target f2_direct --seed 0
# sbatch -J mse2 --partition ellis -t 09:00:00 sr.sh --time_in_hours 8 --version 84963 --target f2_direct --seed 0
# sbatch -J mse --partition ellis -t 09:00:00 sr.sh --time_in_hours 8 --version 4590 --target f2_direct --seed 0
# sbatch -J fixv --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 86952 --target f2 --seed 0
# sbatch -J mse2 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 84963 --target f2 --seed 0
# sbatch -J mse --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 4590 --target f2 --seed 0

# April 22, 2025

# training NN with mse loss, or fix variance, two seeds
# sbatch -J mse --partition ellis prune_train.sh --seed 0 --mse_loss
# sbatch -J mse2 --partition gpu prune_train.sh --seed 1 --mse_loss
# sbatch -J fix_var --partition gpu prune_train.sh --seed 0 --fix_variance

# sr with f1 identity, target f2_direct or f2
# sbatch -J f1id --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --target f2_direct --seed 0 --version 28114
# sbatch -J f1id_f2 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --target f2 --seed 0 --version 28114

# SR with f2 direct
# sbatch -J f2_dir --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2_direct --seed 1
# sbatch -J f2_dir --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2_direct --seed 0

# pure SR with NN targets
# sbatch -J pure --partition ellis -t 09:00:00 run.sh pure_sr.py --time-in-hours 8 --seed 0 --nn_targets True
# sbatch -J pure --partition ellis -t 09:00:00 run.sh pure_sr.py --time-in-hours 8 --seed 1 --nn_targets True

# pure SR with different seeds
# sbatch -J pure --partition gpu -t 09:00:00 run.sh pure_sr.py --time-in-hours 8 --seed 2
# sbatch -J pure --partition gpu -t 09:00:00 run.sh pure_sr.py --time-in-hours 8 --seed 3

# ------------------------------------------------------------------------

# SR with k=3 and k=4
# sbatch -J k3 --time 09:00:00 --partition gpu sr.sh --time_in_hours 8 --version 24880 --target f2
# sbatch -J k3_24h --partition ellis --time 25:00:00 sr.sh --time_in_hours 24 --version 24880 --target f2
# sbatch -J k4 --time 09:00:00 --partition gpu sr.sh --time_in_hours 8 --version 24880 --target f2
# sbatch -J k3s --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 74649 --target f2 --max_size 30
# sbatch -J k4s --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 11566 --target f2 --max_size 30

# March 21 2025
# sbatch -J no_sin --time 09:00:00 --partition gpu sr.sh --time_in_hours 8 --version 24880 --target f2
# sbatch -J clipped --time 01:30:00 --partition gpu sr.sh --time_in_hours 1 --version 24880 --target f2
# sbatch -J clipped2 --time 09:00:00 --partition gpu sr.sh --time_in_hours 8 --version 24880 --target f2

# Tuesday January 14 2025
# sbatch -J srf1id3 --partition gpu sr.sh --time_in_hours 8 --version 95292 --target f2
# sbatch -J srf1id2 --partition gpu sr.sh --time_in_hours 8 --version 75178 --target f2
# sbatch -J srf1id1 --partition gpu sr.sh --time_in_hours 8 --version 95292 --target f2

# Sunday December 21
# sbatch -J 10f1id_sr --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 14763 --target f2
# sbatch -J f1id1 --partition gpu prune_train.sh --seed 1 --f1_variant identity --prune_f1_topk 10
# sbatch -J f1id2 --partition gpu prune_train.sh --seed 2 --f1_variant identity --prune_f1_topk 10
# sbatch -J f1id3 --partition gpu prune_train.sh --seed 3 --f1_variant identity --prune_f1_topk 10


# Saturday December 21
# sbatch -J 10f1idL1reginputs --partition gpu train.sh --f1_variant identity --load_f1_f2 26929 --prune_f1_topk 10 --total_steps 150000

# Friday December 20
# sbatch -J f1idL1reginputs --partition gpu train.sh --f1_variant identity --l1_reg inputs --l1_coeff 2
# sbatch -J f1idL1reginputs --partition gpu train.sh --f1_variant identity --l1_reg inputs --l1_coeff 2 --total_steps 150000

# Wednedsay December 18
# sbatch -J p1sr --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 92130 --target f2 --max_size 30
# sbatch -J p2sr --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 10777 --target f2 --max_size 30
# sbatch -J p3sr --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 43317 --target f2 --max_size 30

# Tuesday December 17
# sbatch -J f1id1 --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 854 --target f2 --max_size 30
# sbatch -J f1id2 --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 11237 --target f2 --max_size 30
# sbatch -J f1id3 --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 28114 --target f2 --max_size 30
# sbatch -J prune1 --partition gpu prune_train.sh --seed 1
# sbatch -J prune2 --partition gpu prune_train.sh --seed 2
# sbatch -J prune3 --partition gpu prune_train.sh --seed 3
# sbatch -J sr_ll --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2_direct --loss_fn ll
# sbatch -J f1_id_ll --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 12370 --target f2_direct --loss_fn ll

# Monday December 16
# try different seeds for f1 identity and baseline
# sbatch -J f1id1 --partition gpu train.sh --f1_variant identity --seed 1
# sbatch -J f1id2 --partition gpu train.sh --f1_variant identity --seed 2
# sbatch -J f1id3 --partition gpu train.sh --f1_variant identity --seed 3
# sbatch -J prune1 --partition gpu prune_train.sh --seed 1
# sbatch -J prune2 --partition gpu prune_train.sh --seed 2
# sbatch -J prune3 --partition gpu prune_train.sh --seed 3

# Thursday December 12
# sbatch -J k3s --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 74649 --target f2 --max_size 30
# sbatch -J k4s --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 11566 --target f2 --max_size 30
# sbatch -J k5s --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 72646 --target f2 --max_size 30
# sbatch -J k0_f2lin train.sh --load_f1_f2 25646 --total_steps 150000 --f2_variant linear --prune_f2_topk 0
# sbatch -J k1_f2lin train.sh --load_f1_f2 25646 --total_steps 150000 --f2_variant linear --prune_f2_topk 1

# Wednesday December 11
# sbatch -J id_sr --partition gpu -t 02:00:00 sr.sh --time_in_hours 2 --version 12370 --target f2
# sbatch -J f2lin --partition train.sh --load_f1 24880 --f2_variant linear --freeze_f1

# f1+f2 linear comparison
# sbatch -J f2lin train.sh --total_steps 150000 --f2_variant linear --load_f1 24880 --l1_reg f2_weights --l1_coeff 2

# now apply prune and fine tune f2
# sbatch -J k20_f2lin train.sh --load_f1_f2 25646 --total_steps 150000 --f2_variant linear
# sbatch -J k10_f2lin train.sh --load_f1_f2 25646 --total_steps 150000 --f2_variant linear --prune_f2_topk 10
# sbatch -J k5_f2lin train.sh --load_f1_f2 25646 --total_steps 150000 --f2_variant linear --prune_f2_topk 5
# sbatch -J k2_f2lin train.sh --load_f1_f2 25646 --total_steps 150000 --f2_variant linear --prune_f2_topk 2
# sbatch -J k0_f2lin train.sh --load_f1_f2 25646 --total_steps 150000 --f2_variant linear --prune_f2_topk 0

# Tuesday December 10
# sbatch -J f1id --partition gpu train.sh --f1_variant identity
# sbatch -J f1id --partition gpu train.sh --f1_variant identity --fix_variance
# sbatch -J fix_var --partition gpu train.sh --fix_variance
# sbatch -J fix_var --partition gpu prune_train.sh --fix_variance
# sbatch -J fix_var --partition gpu train.sh --fix_variance --f1_variant mlp

# Monday December 9
# sbatch -J pure --partition gpu -t 02:00:00 run.sh pure_sr.py --time-in-hours 1
# sbatch -J pure --partition gpu -t 09:00:00 run.sh pure_sr.py --time-in-hours 9
# sbatch -J pure --partition gpu -t 25:00:00 run.sh pure_sr.py --time-in-hours 24
# sbatch -J eq_pred --partition gpu train.sh --predict_eq_uncertainty --load_f1 24880 --total_steps 30000 --seed 1
# sbatch -J eq_pred --partition gpu train.sh --predict_eq_uncertainty --load_f1 24880 --total_steps 60000 --seed 2
# sbatch -J eq_pred --partition gpu train.sh --predict_eq_uncertainty --load_f1 24880 --total_steps 90000 --seed 3

# Thursday December 5
# sbatch -J eq_pred --partition gpu train.sh --predict_eq_uncertainty --load_f1 24880 --total_steps 60000
# sbatch -J eq_pred --partition gpu train.sh --predict_eq_uncertainty --load_f1 24880 --total_steps 60000

# Sunday November 24
# sbatch -J 1_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold 1
# sbatch -J 5_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold 5
# sbatch -J .1_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold .1
# sbatch -J 1_1p --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold 1 --loss_fn perceptron
# sbatch -J pure --partition gpu -t 01:00:00 run.sh pure_sr.py --time-in-hours 0.01
# sbatch -J eq_pred --partition gpu train.sh --predict_eq_uncertainty --load_f1 24880

# Friday November 22
# sbatch -J test_classification --partition gpu -t 01:00:00 run.sh sr_classification.py
# sbatch -J 1_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold 1
# sbatch -J 5_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold 5
# sbatch -J .1_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold .1
# sbatch -J 1_1p --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold 1 --loss_fn perceptron

# Tuesday November 19
# sbatch -J pure --partition gpu -t 01:00:00 run.sh pure_sr.py --time-in-hours 0.01

# Monday November 18
# sbatch -J pure1 --partition gpu sr2.sh --time-in-hours 1
# sbatch -J pure1 --partition gpu -t 09:00:00 sr2.sh --time-in-hours 8
# sbatch -J fixvar --partition gpu train.sh --fix_variance

# sbatch -J eq_1_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold 1
# sbatch -J eq5_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold 5
# sbatch -J eq.1_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold .1
# sbatch -J ll_loss --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2_direct --loss_fn ll

# Friday November 15
# sbatch -J pure1 --partition gpu sr2.sh --time-in-hours 1
# sbatch -J pure8 --partition gpu sr2.sh --time-in-hours 8

# Thursday November 13

# sbatch -J testeq1_10 --partition gpu sr.sh --time_in_hours 0.01 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold 1 --eq_bound_oob_target 10

# sbatch -J eq_1_1 --partition gpu sr.sh --time_in_hours 0.01 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold 1
# sbatch -J eq5_1 --partition gpu sr.sh --time_in_hours 1 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold 5
# sbatch -J eq.1_1 --partition gpu sr.sh --time_in_hours 1 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold .1
# sbatch -J eq.01_1 --partition gpu sr.sh --time_in_hours 1 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold .01

# sbatch -J eq_1_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold 1
# sbatch -J eq5_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold 5
# sbatch -J eq.1_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold .1
# sbatch -J eq.01_1 --partition gpu -t 09:00:00 sr.sh --time_in_hours 8 --version 24880 --previous_sr_path 'sr_results/11003.pkl' --target 'equation_bounds' --eq_bound_mse_threshold .01


# Wednesday November 13
# sbatch -J pure --partition ellis sr.sh --time-in-hours 1
# sbatch -J test --partition ellis sr.sh --time_in_hours 1 --version 24880 --target f2
# sbatch -J test --partition gpu sr.sh --time_in_hours 1 --version 24880 --target f2
# sbatch -J pure --partition gpu sr.sh --time-in-hours 1
# sbatch -J pure2 --partition gpu sr.sh --time-in-hours 1

# Friday Oct 16

# evaluate the linear comparisons
# sbatch -J eval2 --partition gpu eval_eqs.sh --version 64336
# sbatch -J eval5 --partition gpu eval_eqs.sh --version 47272
# sbatch -J eval10 --partition gpu eval_eqs.sh --version 94348
# sbatch -J eval15 --partition gpu eval_eqs.sh --version 51791
# sbatch -J eval20 --partition gpu eval_eqs.sh --version 53392

# Friday Oct 4
# to see how much results are affected by seed
# sbatch -J copy --partition gpu prune_train.sh --seed 10
# sbatch -J copy --partition gpu prune_train.sh --seed 11

# f2 linear comparison
# sbatch -J nc_linf2_k2 --partition gpu f2_prune_train.sh --prune_f2_topk 2
# sbatch -J nc_linf2_k5 --partition gpu f2_prune_train.sh --prune_f2_topk 5
# sbatch -J nc_linf2_k10 --partition gpu f2_prune_train.sh --prune_f2_topk 10
# sbatch -J nc_linf2_k15 --partition gpu f2_prune_train.sh --prune_f2_topk 15
# sbatch -J nc_linf2_k20 --partition gpu f2_prune_train.sh --prune_f2_topk 20

# Friday Sep 26
#
# sbatch -J M --partition gpu prune_train.sh --combined_mass_feature
# sbatch -J k3 --partition gpu prune_train.sh --prune_f1_topk 3
# sbatch -J k4 --partition gpu prune_train.sh --prune_f1_topk 4
# sbatch -J k5 --partition gpu prune_train.sh --prune_f1_topk 5
# sbatch -J M-3 --partition gpu prune_train.sh --combined_mass_feature --prune_f1_topk 3
# sbatch -J M-4 --partition gpu prune_train.sh --combined_mass_feature --prune_f1_topk 4
# sbatch -J M-5 --partition gpu prune_train.sh --combined_mass_feature --prune_f1_topk 5

# sbatch -J k3s --partition gpu --time 25:00:00 sr.sh --time_in_hours 24 --version 74649 --target f2 --max_size 60
# sbatch -J k4s --partition gpu --time 25:00:00 sr.sh --time_in_hours 24 --version 11566 --target f2 --max_size 60
# sbatch -J k5s --partition gpu --time 25:00:00 sr.sh --time_in_hours 24 --version 72646 --target f2 --max_size 60
# sbatch -J Ms --partition gpu --time 25:00:00 sr.sh --time_in_hours 22 --version 21622 --target f2 --max_size 60
# accidentally killed the M job :(
# sbatch -J M --partition gpu train.sh --version 21622 --total_steps 150000 --load_f1_f2 31796 --prune_f1_topk 2 --combined_mass_feature

# Friday Sep 13

# compare model trained deterministically vs normally.
# sbatch -J det0 --partition gpu train.sh --deterministic_summary_stats --seed 0
# sbatch -J det1 --partition gpu train.sh --deterministic_summary_stats --seed 1
# sbatch -J nondet0 --partition gpu train.sh --seed 0
# sbatch -J nondet1 --partition gpu train.sh --seed 1


# try fine-tuning

# Mon Sep 10

# debug NaN's when training nn for residual/variance prediction
# bash train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection best --freeze_f1 --total_steps 100
# bash train.sh --total_steps 100

# Fri Sep 6

# learning a sparse linear f1, given that f2 is linear
# sbatch -J linf2 --partition gpu f2_prune_train.sh
# much simpler f2 linear comparison: just load 24880 for f1 and then train a sparse linear f2
# sbatch -J linf22 --partition gpu f2_prune_train2.sh

# eval the new pysr equations from sr 3 days
# sbatch -J evalsr3 eval_eqs.sh --version 24880 --pysr_version 51173

# fine tune f2 with autosimplified f1
# sbatch --partition gpu -J e.001 train.sh --load_f1_f2 24880 --load_f1_feature_nn models/24880_feature_nn_simplified_v3_eps=0.001.pt --freeze_f1
# sbatch --partition gpu -J e.1 train.sh --load_f1_f2 24880 --load_f1_feature_nn models/24880_feature_nn_simplified_v3_eps=0.1.pt --freeze_f1

# try a regular prune train, make sure things are still working?
# sbatch --partition gpu -J debug prune_train.sh

# nn pred std
# sbatch -J nn_pred_std --partition gpu train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection accuracy --nn_pred_std --freeze_f1 --total_steps 500
# bash train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection accuracy --nn_pred_std --freeze_f1 --total_steps 500

# once those are done, learn pysr equations for them.
# sbatch -J as.001 --partition gpu --time 25:00:00 sr.sh --time_in_hours 23 --version 10290 --target f2 --max_size 60
# sbatch -J as.1 --partition gpu --time 25:00:00 sr.sh --time_in_hours 23 --version 9259 --target f2 --max_size 60


# sbatch -J f2_l2 --partition gpu f2_prune_train.sh --latent 10 --load_f1 24880 --prune_f2_topk 2 --freeze_f1 --total_steps 500
# bash f2_prune_train.sh --latent 10 --load_f1 24880 --prune_f2_topk 2 --freeze_f1 --total_steps 500
# sbatch -J f2_l5 --partition gpu f2_prune_train.sh --latent 10 --load_f1 24880 --prune_f2_topk 5 --freeze_f1
# sbatch -J f2_l10 --partition gpu f2_prune_train.sh --latent 10 --load_f1 24880 --prune_f2_topk 10 --freeze_f1
# sbatch -J f2_l20 --partition gpu f2_prune_train.sh --latent 10 --load_f1 24880 --prune_f2_topk 20 --freeze_f1
# sbatch -J f2_l30 --partition gpu f2_prune_train.sh --latent 10 --load_f1 24880 --prune_f2_topk 30 --freeze_f1

# Wed Sep 4

# sbatch -J eval --partition gpu eval_eqs.sh --version 10290 --pysr_version 69083


# compare model trained deterministically vs normally.
# sbatch -J det0 --partition gpu train.sh --deterministic_summary_stats --seed 0
# sbatch -J det1 --partition gpu train.sh --deterministic_summary_stats --seed 1
# sbatch -J det2 --partition gpu train.sh --deterministic_summary_stats --seed 2
# sbatch -J nondet0 --partition gpu train.sh --seed 0
# sbatch -J nondet1 --partition gpu train.sh --seed 1
# sbatch -J nondet2 --partition gpu train.sh --seed 2

# evaluate the autosimplified pysr equations


# Mon September 2

# sbatch -J autosimp0.001 --partition gpu --time 25:00:00 sr.sh --time_in_hours 23 --version 10290 --target f2 --max_size 60
# sbatch -J autosimp0.1 --partition gpu --time 25:00:00 sr.sh --time_in_hours 23 --version 9259 --target f2 --max_size 60

# sbatch -J sr3 --partition ellis --mem 300G --time 3-00:00:00 sr.sh --time_in_hours 70 --version 24880 --target f2 --max_size 60

# TODO need to avoid NaN's
# sbatch -J f2res --partition gpu train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection accuracy --f2_residual mlp --freeze_f1
# sbatch -J nn_pred_std --partition gpu train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection accuracy --nn_pred_std --freeze_f1

# sbatch -J sr7 --partition ellis --time 7-00:00:00 sr.sh --time_in_hours 160 --version 1021 --target f2 --max_size 60

# Thu August 29

# model_selection_values=(5 7 9 11 14 18 20 27 29)
# # model_selection_values=(3)
# for val in "${model_selection_values[@]}"
# do
#     bash train.sh --load_f1 24880 --pysr_f2 "sr_results/11003.pkl" --pysr_f2_model_selection "$val" --eval
# done

# sbatch -J eval7 --partition gpu train.sh --load_f1 24880 --pysr_f2 "sr_results/51897.pkl" --pysr_f2_model_selection accuracy --eval
# sbatch -J eval3 --partition gpu train.sh --load_f1 24880 --pysr_f2 "sr_results/3535.pkl" --pysr_f2_model_selection accuracy --eval
# sbatch -J eval7_2 --partition gpu train.sh --load_f1 24880 --pysr_f2 "sr_results/78515.pkl" --pysr_f2_model_selection accuracy --eval

# sbatch -J sr7 --partition ellis --time 7-00:00:00 sr.sh --time_in_hours 160 --version 1021 --target f2 --max_size 60
# sbatch -J sr7 --partition gpu --time 7-00:00:00 sr.sh --time_in_hours 160 --version 1021 --target f2 --max_size 60

# Thu August 22

# sbatch -J evaleqs --partition gpu train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection accuracy --calc_scores
# sbatch -J evaleqs --partition gpu train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection 1 --calc_scores
# sbatch -J evaleqs --partition gpu train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection 3 --calc_scores
# sbatch -J evaleqs --partition gpu train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection 7 --calc_scores
# sbatch -J evaleqs --partition gpu train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection 14 --calc_scores
# sbatch -J evaleqs --partition gpu train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection 21 --calc_scores
# sbatch -J evaleqs --partition gpu train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection 28 --calc_scores
# bash train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection accuracy --calc_scores

# Thu August 14

# 3.35 # sbatch -J 50-1k --partition gpu --time 12:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --batch_size 50 --n 1000
# 2.80 # sbatch -J 100-1k --partition gpu --time 12:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --batch_size 100 --n 1000

# 2.31 # sbatch -J 50-5k --partition gpu --time 12:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --batch_size 50 --n 5000
# 2.30 # sbatch -J 100-5k --partition gpu --time 12:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --batch_size 100 --n 5000
# 2.36 sbatch -J 500-5k --partition gpu --time 12:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --batch_size 500 --n 5000
# 2.28 sbatch -J 1000-5k --partition gpu --time 12:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --batch_size 1000 --n 5000

# 2.36 # sbatch -J 50-10k --partition gpu --time 12:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --batch_size 50 --n 10000
# 2.52 # sbatch -J 100-10k --partition gpu --time 12:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --batch_size 100 --n 10000
# 2.30 # sbatch -J 500-10k --partition gpu --time 12:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --batch_size 500 --n 10000
# 2.45 # sbatch -J 1k-10k --partition gpu --time 12:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --batch_size 1000 --n 10000

# 2.32 # sbatch -J 100-20k --partition gpu --time 12:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --batch_size 100 --n 20000
# 2.39 # sbatch -J 500-20k --partition gpu --time 12:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --batch_size 500 --n 20000
# 2.31 # sbatch -J 1k-20k --partition gpu --time 12:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2 --batch_size 1000 --n 20000


# sbatch -J f2_l2 --partition gpu f2_prune_train.sh --latent 10 --load_f1 24880 --prune_f2_topk 2
# sbatch -J f2_l5 --partition gpu f2_prune_train.sh --latent 10 --load_f1 24880 --prune_f2_topk 5
# sbatch -J f2_l10 --partition gpu f2_prune_train.sh --latent 10 --load_f1 24880 --prune_f2_topk 10
# sbatch -J f2_l20 --partition gpu f2_prune_train.sh --latent 10 --load_f1 24880 --prune_f2_topk 20
# sbatch -J f2_l30 --partition gpu f2_prune_train.sh --latent 10 --load_f1 24880 --prune_f2_topk 30

# sbatch --partition gpu -J nzsimp_e0.001 train.sh --load_f1_f2 24880 --load_f1_feature_nn models/24880_feature_nn_simplified_v2_norm_nozero_eps=0.001.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nzsimp_e0.005 train.sh --load_f1_f2 24880 --load_f1_feature_nn models/24880_feature_nn_simplified_v2_norm_nozero_eps=0.005.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nzsimp_e0.01 train.sh --load_f1_f2 24880 --load_f1_feature_nn models/24880_feature_nn_simplified_v2_norm_nozero_eps=0.01.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nzsimp_e0.05 train.sh --load_f1_f2 24880 --load_f1_feature_nn models/24880_feature_nn_simplified_v2_norm_nozero_eps=0.05.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nzsimp_e0.1 train.sh --load_f1_f2 24880 --load_f1_feature_nn models/24880_feature_nn_simplified_v2_norm_nozero_eps=0.1.pt --freeze_f1 --latent 10

# sbatch -J e0.1 --partition gpu --time 25:00:00 sr.sh --time_in_hours 24 --version 7652 --target f2
# sbatch -J e0.01 --partition gpu --time 25:00:00 sr.sh --time_in_hours 24 --version 30941 --target f2
# sbatch -J e0.001 --partition gpu --time 25:00:00 sr.sh --time_in_hours 24 --version 17401 --target f2

# sbatch -J sr7 --partition ellis --time 7-00:00:00 sr.sh --time_in_hours 160 --version 24880 --target f2
# sbatch -J sr3 --partition ellis --time 3-00:00:00 sr.sh --time_in_hours 65 --version 24880 --target f2


# Wed August 14
# sbatch --partition gpu -J nsimp_e0.001 train.sh --load_f1_f2 24880 --load_f1_feature_nn models/24880_feature_nn_simplified_v2_norm_eps=0.001.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nsimp_e0.005 train.sh --load_f1_f2 24880 --load_f1_feature_nn models/24880_feature_nn_simplified_v2_norm_eps=0.005.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nsimp_e0.01 train.sh --load_f1_f2 24880 --load_f1_feature_nn models/24880_feature_nn_simplified_v2_norm_eps=0.01.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nsimp_e0.05 train.sh --load_f1_f2 24880 --load_f1_feature_nn models/24880_feature_nn_simplified_v2_norm_eps=0.05.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nsimp_e0.1 train.sh --load_f1_f2 24880 --load_f1_feature_nn models/24880_feature_nn_simplified_v2_norm_eps=0.1.pt --freeze_f1 --latent 10


# Mon August 12

# comparing normalized vs unnormalized f1 features
# sbatch --partition gpu -J norm train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_norm_eps=0.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J norm_newf2 train.sh --load_f1 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_norm_eps=0.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J unnorm train.sh --load_f1_f2 24880 --freeze_f1 --latent 10
# sbatch --partition gpu -J unnorm_newf2 train.sh --load_f1 24880 --freeze_f1 --latent 10

# trying the different epsilon values, normalized and unnormalized
# unnormalized
# sbatch --partition gpu -J simp_e0.01 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_eps=0.01.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J simp_e0.1 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_eps=0.1.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J simp_e0.5 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_eps=0.5.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J simp_e1 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_eps=1.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J simp_e2 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_eps=2.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J simp_e5 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_eps=5.pt --freeze_f1 --latent 10

# normalized
# sbatch --partition gpu -J nsimp_e0.001 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_norm_eps=0.001.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nsimp_e0.01 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_norm_eps=0.01.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nsimp_e0.1 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_norm_eps=0.1.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nsimp_e0.5 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_norm_eps=0.5.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nsimp_e1 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_norm_eps=1.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nsimp_e2 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_norm_eps=2.pt --freeze_f1 --latent 10

# didn't run, not worth it TODO still need to change the feature_nn versions before running
# sbatch --partition gpu -J nsimp_e0.05 train.sh --load_f1_f2 24880 --load_f1_feature_nn fix24880_feature_nn_simplified_v2_eps=0.5.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nsimp_e0.2 train.sh --load_f1_f2 24880 --load_f1_feature_nn fix24880_feature_nn_simplified_v2_eps=2.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nsimp_e0.75 train.sh --load_f1_f2 24880 --load_f1_feature_nn fix24880_feature_nn_simplified_v2_norm_eps=5.pt --freeze_f1 --latent 10
# sbatch --partition gpu -J nsimp_e0.95 train.sh --load_f1_f2 24880 --load_f1_feature_nn fix24880_feature_nn_simplified_v2_eps=5.pt --freeze_f1 --latent 10

# didn't run, not worth it
# sbatch --partition ellis -J simp_e0.75 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_eps=0.75.pt --freeze_f1 --latent 10
# sbatch --partition ellis -J simp_e0.2 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_eps=0.2.pt --freeze_f1 --latent 10
# sbatch --partition ellis -J simp_e0.05 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_v2_eps=0.05.pt --freeze_f1 --latent 10


# Friday August 9

# sbatch --partition ellis -J eps0.1 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_eps=0.1.pt --freeze_f1 --latent 10
# sbatch --partition ellis -J eps0.2 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_eps=0.2.pt --freeze_f1 --latent 10
# sbatch --partition ellis -J eps0.5 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_eps=0.5.pt --freeze_f1 --latent 10
# sbatch --partition ellis -J eps1 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified_eps=1.pt --freeze_f1 --latent 10

# sbatch -J sr --partition gpu --time 25:00:00 sr.sh --time_in_hours 24 --version 1021 --target f2
# sbatch -J sr7 --partition gpu --time 7-00:00:00 sr.sh --time_in_hours 160 --version 1021 --target f2

# ------------------------------- Thu August 8 -----------------------------------

# sbatch -J f1simple train.sh --load_f1_feature_nn 24880_feature_nn_simplified.pt --freeze_f1 --latent 10
# sbatch -J f1unsimple train.sh --load_f1 24880 --freeze_f1 --latent 10
# sbatch -J f1_f2 train.sh --load_f1_f2 24880 --freeze_f1
# sbatch -J f1simplef2 train.sh --load_f1_f2 24880 --load_f1_feature_nn 24880_feature_nn_simplified.pt --freeze_f1 --latent 10
# sbatch -J f1_f2_unfrozen train.sh --load_f1_f2 24880

# ------------------------------- Thu August 1 -----------------------------------


# bash train.sh --load_f1_f2 24880 --calc_scores
# bash train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection best --calc_scores
# bash train.sh --load_f1_f2 43139 --calc_scores
# bash train.sh --load_f1 43139 --pysr_f2 'sr_results/33060.pkl' --pysr_f2_model_selection best --calc_scores

# ------------------------------- Mon July 8 -----------------------------------

# sbatch -J swag --partition ellis run_swag.sh --version 24880 --eval
# bash run_swag.sh --version 24880 --eval

# ------------------------------- Mon July 2 -----------------------------------

# sbatch -J sr --partition gpu --time 2:00:00 sr.sh --time_in_hours 1 --version 24880 --target f2
# sbatch -J sr --partition gpu --time 9:00:00 sr.sh --time_in_hours 8 --version 24880 --target f2
# sbatch -J sr --partition gpu --time 25:00:00 sr.sh --time_in_hours 24 --version 24880 --target f2

# sbatch -J 14ft_f2froz --partition gpu train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection 14 --freeze_f2 --total_steps 50000
# bash train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection 5 --freeze_f2 --total_steps 100
# sbatch -J 14ft --partition gpu train.sh --load_f1 24880 --pysr_f2 'sr_results/11003.pkl' --pysr_f2_model_selection 14 --total_steps 50000
# sbatch -J lincomp --partition ellis train.sh --load_f1 24880 --f2_variant linear --total_steps 50000

# ------------------------------- Fri June 28 -----------------------------------

# bash train.sh --load_f1 43139 --pysr_f2 'sr_results/33060.pkl' --pysr_f2_model_selection best --eval

# complexity_values=(1 3 4 5 7 9 10 11 12 13 14 15 20 25 30)

# for complexity in "${complexity_values[@]}"; do
#     echo "Running for complexity: $complexity"
#     bash train.sh --load_f1 43139 --pysr_f2 'sr_results/33060.pkl' --pysr_f2_model_selection "$complexity" --eval
# done


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

# sbatch -J sr --partition gpu sr.sh --time_in_hours 0.05 --version 43139 --target f2
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
