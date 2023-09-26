#!/usr/bin/env bash

python simon.py --total_steps 200000 --swa_steps 50000 --version 7649 --seed 0 --angles --no_mmr --no_nan --no_eplusminus "$@"
