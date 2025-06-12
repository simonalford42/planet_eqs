
# train NN with sparse linear psi
version=$(python versions.py)
python -u find_minima.py --version $version --total_steps 150000 --l1_reg weights --l1_coeff 2 "$@"
version2=$(python versions.py)
python -u find_minima.py --version $version2 --total_steps 150000 --load_f1_f2 $version --prune_f1_topk 2 "$@"

# run symbolic regression to distill phi into equations
# we ran with 32 cores for 8 hours
python sr.py --version $version2
# TODO: replace with pysr version ('id') from the run
pysr_v=1

# view all the results with interpret.ipynb
# replace pysr_c with best complexity from interpret.ipynb
pysr_c=1

# run pure sr
# we ran with 32 cores for 8 hours
python pure_sr.py
# TODO: replace with pure sr version and highest complexity reached
puresr_v=2
puresr_c=1

# run direct SR (imitates neural network
version3=$(python versions.py)
python -u find_minima.py --version $version3 --f1_variant identity

# calculate rmse scores for neural network
python calc_rmse.py --version $version2 --eval_type nn --dataset all
# calculate rmse scores for equations
python calc_rmse.py --version $version2 --pysr_version $pysr_v --eval_type pysr --dataset all
# calculate rmse scores for Petit
python calc_rmse.py --eval_type petit --dataset all
# calculate rmse scores for Pure SR
python calc_rmse.py --eval_type pure_sr --dataset all --pysr_version $puresr_v --pysr_model_selection $puresr_c


# calculate five planet results
p five_planet.py --version $version2 --paper-ready --turbo --extrapolate
p five_planet.py --version $version2 --pysr_version $pysr_v --pysr_model_selection $pysr_c --paper-ready --turbo --extrapolate

