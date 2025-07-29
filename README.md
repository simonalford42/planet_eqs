# Distilling neural networks into equations that predict instability of planetary systems

## Installation
1. Clone repo and `cd` into it
2. Create python environment:
    ```bash
    conda create python=3.7.12 -n planet_eqs
    conda activate planet_eqs
    pip install -r requirements.txt
    ```
3. Run `bash setup_data.sh` to download the large data and results files from Zenodo. Or, if you wish to proceed manually:
    - Download data from Zenodo: https://zenodo.org/records/15724986. Extract and verify the checksum.
    - Move the `data/` folder into the main repo folder, and the `period_results/` folder into the `figures` folder.

## Reproducing experiments
Commands for rerunning the experiments and generating figures for the paper are in `experiments.sh`.
Commands that generate the figures from already run experiments are in `official.sh`.

## Miscellaneous
This repository originated as a fork of https://github.com/MilesCranmer/bnn_chaos_model.
