# Distilling neural networks into equations that predict instability of planetary systems

## Installation
1. Clone repo and `cd` into it
2. Create conda environment: `conda env create -f environment.yml` and activate it with `conda activate planet_eqs`.
3. Run `python -c "import pysr; pysr.install()"` to make sure Julia is installed.
4. Run `bash setup_data.sh` to download the large data and results files from Zenodo. Or, if you wish to proceed manually:
    - Download data from Zenodo: https://zenodo.org/records/15724986. Extract and verify the checksum.
    - Move the `data/` folder into the main repo folder, and the `period_results/` folder into the `figures` folder.

## Reproducing experiments
Commands that generate the figures from already run experiments are in `official_plots.sh`.
Commands for rerunning the experiments and generating figures for the paper are in `experiments.sh`.

## Minimal Predictor

Installing the dependencies of the old environment can be challenging, so we created a minimal predictor implementation that predicts with our equations using only numpy, rebound, and matplotlib as dependencies. Made with Codex, verified to match original calculations and plots.

Examples of predictions on the resonant/random splits or a generic rebound simulation is in `minimal/2d_plot.py` and `minimal/planet_stability.py`.

## Miscellaneous
This repository originated as a fork of https://github.com/MilesCranmer/bnn_chaos_model.
