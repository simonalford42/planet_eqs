#!/usr/bin/env bash


# -------------------- Monday jobs -------------------

# sbatch -J lw015 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.15
# sbatch -J lw005 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.005
# sbatch -J lw02 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.2
# sbatch -J lw05 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.5
# sbatch -J lw07 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.7

# -------------------- Friday jobs -------------------

# sbatch -J sr --partition gpu sr.sh --version 21101
# sbatch -J f2_d2_h40 --partition gpu train.sh --f1_variant linear # default: depth 2, hidden 40
# sbatch -J f2_d2_h20 --partition gpu train.sh --f1_variant linear --f2_depth 2 --hidden 20
# sbatch -J f2_d1_h40 --partition gpu train.sh --f1_variant linear --f2_depth 1
# sbatch -J f2_d1_h80 --partition gpu train.sh --f1_variant linear --f2_depth 1 --hidden 80
# sbatch -J f2_d1_h20 --partition gpu train.sh --f1_variant linear --f2_depth 1 --hidden 20

# sbatch -J lw01 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 0.1
# sbatch -J lw1 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 1.0
# sbatch -J lw5 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 5.0
# sbatch -J lw10 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 10
# sbatch -J lw100 --partition gpu train.sh --f1_variant linear --l1_reg weights --l1_coeff 100

# sbatch -J l25 --partition gpu train.sh --f1_variant linear --f2_reg 2.5
# sbatch -J l30 --partition gpu train.sh --f1_variant linear --f2_reg 3.0
# sbatch -J l35 --partition gpu train.sh --f1_variant linear --f2_reg 3.5
# sbatch -J l40 --partition gpu train.sh --f1_variant linear --f2_reg 4.0
# sbatch -J l50 --partition gpu train.sh --f1_variant linear --f2_reg 5.0
# sbatch -J l60 --partition gpu train.sh --f1_variant linear --f2_reg 6.0
# sbatch -J l100 --partition gpu train.sh --f1_variant linear --f2_reg 10.0

# sbatch -J mean_var --partition ellis train.sh --f1_variant identity --no_summary_sample --seed 5
# sbatch -J mean_var --partition ellis train.sh --f1_variant identity --no_summary_sample --seed 6
# sbatch -J mean_var --partition ellis train.sh --f1_variant mean_cov --seed 5 --mean_var
# sbatch -J mean_var --partition ellis train.sh --f1_variant mean_cov --seed 6 --mean_var

# ------------ Wednesday jobs ----------------

# sbatch -J fr_mean_cov --partition ellis train.sh --f1_variant mean_cov --no_summary_sample --no_swag --total_steps 50000 --init_special

# sbatch -J l15 --partition ellis train.sh --f1_variant linear --f2_reg 1.5
# sbatch -J l19 --partition ellis train.sh --f1_variant linear --f2_reg 1.9
# sbatch -J l23 --partition ellis train.sh --f1_variant linear --f2_reg 2.3

# --------------------- Tuesday jobs --------------------

# sbatch -J fr_lin --partition gpu train.sh --f1_variant linear --no_summary_sample --no_swag --total_steps 50000 --load_f1 'version=21101'
# sbatch -J fr_mean_cov --partition gpu train.sh --f1_variant mean_cov --no_summary_sample --no_swag --total_steps 50000 --init_special

# sbatch -J mean_var --partition gpu train.sh --f1_variant mean_cov --seed 0
# sbatch -J mean_var --partition gpu train.sh --f1_variant mean_cov --seed 1

# sbatch -J e1 --partition gpu train.sh --version 1 --seed 0
# sbatch -J e2 --partition gpu train.sh --version 1 --seed 1
# sbatch -J e3 --partition gpu train.sh --version 1 --seed 2
# sbatch -J e4 --partition gpu train.sh --version 1 --seed 3
# sbatch -J e5 --partition gpu train.sh --version 1 --seed 4
# sbatch -J e6 --partition gpu train.sh --version 1 --seed 5
# sbatch -J e7 --partition gpu train.sh --version 1 --seed 6
# sbatch -J e8 --partition gpu train.sh --version 1 --seed 7
# sbatch -J e9 --partition gpu train.sh --version 1 --seed 8
# sbatch -J e10 --partition gpu train.sh --version 1 --seed 9
# sbatch -J e11 --partition gpu train.sh --version 1 --seed 10
# sbatch -J e12 --partition gpu train.sh --version 1 --seed 11
# sbatch -J e13 --partition gpu train.sh --version 1 --seed 12
# sbatch -J e14 --partition gpu train.sh --version 1 --seed 13
# sbatch -J e15 --partition gpu train.sh --version 1 --seed 14
# sbatch -J e16 --partition gpu train.sh --version 1 --seed 15
# sbatch -J e17 --partition gpu train.sh --version 1 --seed 16
# sbatch -J e18 --partition gpu train.sh --version 1 --seed 17
# sbatch -J e19 --partition gpu train.sh --version 1 --seed 18
# sbatch -J e20 --partition gpu train.sh --version 1 --seed 19
# sbatch -J e21 --partition gpu train.sh --version 1 --seed 20
# sbatch -J e22 --partition gpu train.sh --version 1 --seed 21
# sbatch -J e23 --partition gpu train.sh --version 1 --seed 22
# sbatch -J e24 --partition gpu train.sh --version 1 --seed 23
# sbatch -J e25 --partition gpu train.sh --version 1 --seed 24
# sbatch -J e26 --partition gpu train.sh --version 1 --seed 25
# sbatch -J e27 --partition gpu train.sh --version 1 --seed 26
# sbatch -J e28 --partition gpu train.sh --version 1 --seed 27
# sbatch -J e29 --partition gpu train.sh --version 1 --seed 28
# sbatch -J e30 --partition gpu train.sh --version 1 --seed 29

# sbatch -J l13 --partition ellis train.sh --f1_variant linear --f2_reg 1.3
# sbatch -J l9 --partition ellis train.sh --f1_variant linear --f2_reg 0.9
# sbatch -J l11 --partition ellis train.sh --f1_variant linear --f2_reg 1.1

# sbatch -J earlymc --partition ellis --mem=100G train.sh --f1_variant mean_cov --total_steps 40000
