#!/usr/bin/env bash

# sbatch --time=28:00:00 sr_slurm.sh --version 5171 --seed 0 --time_in_hours 24 --max_size 60

# sbatch -J mc80 --partition ellis train.sh --f1_variant mean_cov --hidden 80
# sbatch -J mc160 --partition ellis train.sh --f1_variant mean_cov --hidden 160
# sbatch -J mc80_2 --partition gpu train.sh --f1_variant mean_cov --hidden 80 --f2_depth 2
# sbatch -J mc160_3 --partition gpu train.sh --f1_variant mean_cov --hidden 160 --f2_depth 3

# sbatch -J nosum --partition gpu train.sh --no_summary_sample --seed 1 --no_log

# sbatch -J zt1 --partition gpu train.sh --zero_theta 1
# sbatch -J zt2 --partition gpu train.sh --zero_theta 2
# sbatch -J zt3 --partition gpu train.sh --zero_theta 3
# sbatch -J zt4 --partition gpu train.sh --zero_theta 4
# sbatch -J zt5 --partition gpu train.sh --zero_theta 5
# sbatch -J zt6 --partition gpu train.sh --zero_theta 6
# sbatch -J zt12 --partition gpu train.sh --zero_theta 1 2
# sbatch -J zt34 --partition gpu train.sh --zero_theta 3 4
# sbatch -J zt56 --partition gpu train.sh --zero_theta 5 6
# sbatch -J ztall --partition gpu train.sh --zero_theta 1 2 3 4 5 6

# sbatch -J l1_3 --partition gpu train.sh --l1_reg --l1_coeff 0.3 --version 16095
# sbatch -J l1_4 --partition gpu train.sh --l1_reg --l1_coeff 0.4 --version 15610
# sbatch -J l1_5 --partition gpu train.sh --l1_reg --l1_coeff 0.5 --version 17643
# sbatch -J l1_6 --partition gpu train.sh --l1_reg --l1_coeff 0.6 --version 2586
# sbatch -J l1_7 --partition gpu train.sh --l1_reg --l1_coeff 0.7 --version 32369
# sbatch -J l1_10 --partition gpu train.sh --l1_reg --l1_coeff 0.10 --version 8758
# sbatch -J l1_15 --partition gpu train.sh --l1_reg --l1_coeff 0.15 --version 19756
