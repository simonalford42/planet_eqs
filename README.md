# Distilling neural networks into equations that predict instability of planetary systems

## Installation
1. Clone repo
2. Create python environment:
```bash
conda create python=3.7.12 planet_eqs
conda activate planet_eqs
pip install -r requirements.txt
```
3. Download data from Zenodo: https://zenodo.org/records/15724986. Extract and verify the checksum.
4. Move the `data/` folder into the main repo folder, and the `period_results/` folder into the `figures` folder.

Steps 3 and 4 can be done automatically by running `bash setup_data.sh` from the main folder.

## Reproducing experiments
Commands for rerunning the experiments and generating figures for the paper are in `experiments.sh`.
Commands that generate the figures from already run experiments are in `official.sh`.

## Miscellaneous
This repository originated as a fork of https://github.com/MilesCranmer/bnn_chaos_model.
