#!/usr/bin/env bash

python simon.py --total_steps 300000 --swa_steps 50000 --version 7955 --seed 5 --angles --no_mmr --no_nan --no_eplusminus "$@"
