
# train NN with sparse linear psi
version=$(python versions.py)
python -u find_minima.py --version $version --total_steps 150000 --l1_reg weights --l1_coeff 2 "$@"
version2=$(python versions.py)
python -u find_minima.py --version $version2 --total_steps 150000 --load_f1_f2 $version --prune_f1_topk 2 "$@"

# run symbolic regression to distill phi into equations
sbatch sr.sh --version $version2

# calculate rmse scores for neural network
python calc_rmse.py --version $version2 --eval_type nn --dataset all
# calculate rmse scores for equations
python calc_rmse.py --version $version2 --pysr_version --eval_type pysr --dataset all
# calculate rmse scores for Petit
python calc_rmse.py --eval_type petit --dataset all

# view all the results with interpret.ipynb

# calculate five planet results

