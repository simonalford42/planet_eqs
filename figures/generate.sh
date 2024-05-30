#!/bin/zsh

gen_5p_figures=false
gen_scatterplot_figures=false
gen_comparison_figures=false
gen_likelihood_example=false
gen_orbital_series_for_schematic=false
gen_feature_importances=false

# gen_5p_figures=true
gen_scatterplot_figures=true
# gen_comparison_figures=true
# gen_likelihood_example=true
# gen_orbital_series_for_schematic=true
# gen_feature_importances=true


# 3948 worked
# 6428 doesnt work 
# # 1409 doesnt work
if [ "$gen_scatterplot_figures" = true ] ; then
    python main_figures.py --version 3948 --total_steps 300000 --swa_steps 50000 --angles --no_mmr --no_nan --no_eplusminus --seed 0 --plot "$@"
    # the test set... wait until the end to use
    # python main_figures.py --version 3948 --total_steps 300000 --swa_steps 50000 --angles --no_mmr --no_nan --no_eplusminus --seed 0 --plot --plot_random "$@"
fi

if [ "$gen_5p_figures" = true ] ; then
    python multiswag_5_planet.py 50
fi

if [ "$gen_comparison_figures" = true ] ; then
    python comparison_figures.py && \
    python comparison_figures.py --random
fi

if [ "$gen_likelihood_example" = true ] ; then
    python likelihood.py
fi

if [ "$gen_orbital_series_for_schematic" = true ] ; then
    python orbital_series.py
fi

if [ "$gen_feature_importances" = true ] ; then
    python feature_importance.py 50
fi
