# Distilling neural networks into equations that predict instability of planetary systems

## Installation
1. Clone repo and `cd` into it
2. Create conda environment: `conda env create -f environment.yml` and activate it with `conda activate planet_eqs`.
3. Run `python -c "import pysr; pysr.install()"` to make sure Julia is installed.
4. Run `bash setup_data.sh` to download the large data and results files from Zenodo. Or, if you wish to proceed manually:
    - Download data from Zenodo: https://zenodo.org/records/15724986. Extract and verify the checksum.
    - Move the `data/` folder into the main repo folder, and the `period_results/` folder into the `figures` folder.

Typical install time is between 10 minutes and 1 hour, depending on how quickly you can download the data.

## System requirements
See `environment.yml` for all software dependencies.

This software was developed on Ubuntu 24.04.4 LTS (Noble Numbat) on a SLURM-managed Linux compute node, using Python 3.7.12, Julia 1.12.5, and GCC 13.3.0.

Installation of software dependencies, reproduction of the main figures, and execution of the code example was additionally tested using Python 3.10.19.

Reproduction of the main figures that do not require GPU was reproduced on MacOS 26.3 with arm64 hardware. 


## Reproducing experiments
Commands that reproduce the figures from already-run experiments are in `official_plots.sh` and takes 15–60 minutes (some of the evaluations are computed from scratch the first time this command is run.)

Commands for rerunning all experiments (training, evaluation, comparisons/ablations) and generating figures for the paper are in `experiments.sh`.

End-to-end reproduction of training of the main distilled equations (two-stage NN training and pruning, distillation into equations via PySR) takes around 15 hours: 7 hours to train and prune the neural network, and 8 hours to distill equations. 

Running the full set of ablations, comparisons, and evaluations takes longer, but can be done in parallel.

## Demos
Installing the dependencies of the old environment can be challenging, so we created a minimal predictor implementation that predicts with our equations using only numpy, rebound, and matplotlib as dependencies. Made with Codex, verified to match original calculations and plots.

Demonstrations of predictions on the resonant/random splits or a generic rebound simulation is in `minimal/2d_plot.py` and `minimal/planet_stability.py`.

## Miscellaneous
This repository originated as a fork of https://github.com/MilesCranmer/bnn_chaos_model.

## License
Apache license
