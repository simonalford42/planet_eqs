
# train NN with sparse linear psi
sbatch prune_train.sh
# run symbolic regression to distill phi into equations
sbatch sr.sh --time_in_hours 8 --target f2 --max_size 30 --version 24880
