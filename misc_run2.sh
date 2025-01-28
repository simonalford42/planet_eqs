#!/usr/bin/env bash

declare -a pairs=(
    # "92130 68813"
    # "10777 86627"
    # "43317 14367"
    "854 98177"
    "11237 24004"
    "28114 62431"
)

for pair in "${pairs[@]}"; do
    read version pysr_version <<< "$pair"
    echo "Processing version pair: $version, $pysr_version"
    python find_minima.py --eval --load_f1 "$version" --pysr_f2 "sr_results/${pysr_version}.pkl" --pysr_f2_model_selection accuracy --no_plot
    # python find_minima.py --eval --load_f1_f2 "$version" --no_plot
    echo "----------------------------------------"
done
