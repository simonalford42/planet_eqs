#!/usr/bin/env bash

# baseline models with different seeds
# python main_figures.py --version 92130 --pysr_version 68813 --plot_random --just_rmse
# python main_figures.py --version 10777 --pysr_version 86627 --plot_random --just_rmse
# python main_figures.py --version 43317 --pysr_version 14367 --plot_random --just_rmse

# f2 identity models with different seeds
# python main_figures.py --version 854 --pysr_version 98177 --plot_random --just_rmse
# python main_figures.py --version 11237 --pysr_version 24004 --plot_random --just_rmse
# python main_figures.py --version 28114 --pysr_version 62431 --plot_random --just_rmse

python multiswag_5_planet.py --N 100 --turbo --version 92130 --pysr_version 68813
python multiswag_5_planet.py --N 100 --turbo --version 10777 --pysr_version 86627
python multiswag_5_planet.py --N 100 --turbo --version 43317 --pysr_version 14367

python multiswag_5_planet.py --N 100 --turbo --version 854 --pysr_version 98177
python multiswag_5_planet.py --N 100 --turbo --version 11237 --pysr_version 24004
python multiswag_5_planet.py --N 100 --turbo --version 28114 --pysr_version 62431

# calculating the val losses of the models
# python find_minima.py --eval --load_f1 24880 --pysr_f2 "sr_results/11003.pkl" --pysr_f2_model_selection accuracy --no_plot
# python find_minima.py --eval --load_f1 12370 --pysr_f2 "sr_results/22943.pkl" --pysr_f2_model_selection accuracy --no_plot
# python find_minima.py --eval --load_f1_f2 24880 --no_plot
# python find_minima.py --eval --load_f1_f2 12370 --no_plot

# Hard-coded version pairs (version, pysr_version)
# declare -a pairs=(
#     "92130 68813"
#     "10777 86627"
#     "43317 14367"
#     "854 98177"
#     "11237 24004"
#     "28114 62431"
# )

# for pair in "${pairs[@]}"; do
#     read version pysr_version <<< "$pair"
#     echo "Processing version pair: $version, $pysr_version"
#     python find_minima.py --eval --load_f1 "$version" --pysr_f2 "sr_results/${pysr_version}.pkl" --pysr_f2_model_selection accuracy --no_plot
#     # python find_minima.py --eval --load_f1_f2 "$version" --no_plot
#     echo "----------------------------------------"
# done
