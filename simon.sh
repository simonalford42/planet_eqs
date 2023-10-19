#!/usr/bin/env bash

python simon.py --total_steps 300000 --swa_steps 50000 --version 4995 --angles --no_mmr --no_nan --no_eplusminus "$@"
