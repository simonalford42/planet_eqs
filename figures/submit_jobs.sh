#!/usr/bin/env bash

# compute and plot for BNN predictions
# python period_ratio_figure.py --Ngrid 4 --version 24880 --compute
# python period_ratio_figure.py --Ngrid 4 --version 24880 --plot

# compute and plot for pysr f2 (need to have computed for BNN before running this)
# python period_ratio_figure.py --Ngrid 4 --version 24880 --pysr_path '../sr_results/33060.pkl' --compute
# python period_ratio_figure.py --Ngrid 4 --version 24880 --pysr_path '../sr_results/33060.pkl' --plot

# compute using 4 parallel jobs
# python period_ratio_figure.py --Ngrid 400 --version 24880 --compute --parallel_ix 0 --parallel_total 4
# python period_ratio_figure.py --Ngrid 400 --version 24880 --compute --parallel_ix 1 --parallel_total 4
# python period_ratio_figure.py --Ngrid 400 --version 24880 --compute --parallel_ix 2 --parallel_total 4
# python period_ratio_figure.py --Ngrid 400 --version 24880 --compute --parallel_ix 3 --parallel_total 4

# collate the parallel results and save
# python period_ratio_figure.py --Ngrid 400 --version 24880 --collate --parallel_total 4

# python period_ratio_figure.py --Ngrid 400 --version 24880 --pysr_path 'sr_results/33060.pkl' --plot
