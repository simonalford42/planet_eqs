
# train NN with sparse linear psi
python -u find_minima.py --version 1 --total_steps 150000 --l1_reg weights --l1_coeff 2
python -u find_minima.py --version 2 --total_steps 150000 --load_f1_f2 1 --prune_f1_topk 2

# run symbolic regression to distill phi into equations
# we ran with 32 cores for 8 hours
python sr.py --nn_version 2 --version 3

# run pure sr (w/ or w/o psi)
# we ran with 32 cores for 8 hours
python pure_sr.py --f1 --version 4
python pure_sr.py --no-f1 --version 5

# calculate rmse scores for neural network
python calc_rmse.py --version 2 --eval_type nn --dataset all
# calculate rmse scores for equations
python calc_rmse.py --version 2 --pysr_version 4 --eval_type pysr --dataset all
# calculate rmse scores for Petit
python calc_rmse.py --eval_type petit --dataset all
# calculate rmse scores for Pure SR
python calc_rmse.py --eval_type pure_sr --dataset all --pysr_version 4
python calc_rmse.py --eval_type pure_sr --dataset all --pysr_version 5

# store best complexity for each sr/pure sr version
# sr_c=TODO
# puresr_c=TODO
# puresr2_c=TODO

cd figures

# calculate five planet results
p five_planet.py --version 2 --pysr_version 3 --pysr_model_selection $sr_c --paper-ready --turbo --extrapolate
p five_planet.py --pure_sr --pysr_version 4 --pysr_model_selection $puresr_c --paper-ready --turbo --extrapolate
p five_planet.py --pure_sr --pysr_version 5 --pysr_model_selection $puresr2_c --paper-ready --turbo --extrapolate

# calculate and plot period ratio results
p period_ratio_figure.py --Ngrid 300 --version 2 --compute
p period_ratio_figure.py --Ngrid 300 --version 2 --pysr_version 3 --compute
p period_ratio_figure.py --Ngrid 300 --version 2 --pysr_version 3 --plot
p period_ratio_figure.py --special 4way --Ngrid 300 --version 2 --pysr_version 3 --pysr_model_selection $sr_c
p period_ratio_figure.py --Ngrid 300 --pure_sr --pysr_version 4 --pysr_model_selection $puresr_c --compute
p period_ratio_figure.py --Ngrid 300 --pure_sr --pysr_version 4 --pysr_model_selection $puresr2_c --compute
p period_ratio_figure.py --Ngrid 300 --pure_sr --pysr_version 4 --pysr_model_selection $puresr_c --plot
p period_ratio_figure.py --Ngrid 300 --pure_sr --pysr_version 4 --pysr_model_selection $puresr2_c --plot

# topk comparisons
for k in {3..5} do
    python -u find_minima.py --version $(10+k) --total_steps 150000 --l1_reg weights --l1_coeff 2
    python -u find_minima.py --version k --total_steps 150000 --load_f1_f2 $(10+k) --prune_f1_topk $k
end

# f2 linear models
for k in (0 2 5 10 20) do
    version=$(python versions.py)
    python -u find_minima.py --version $version --total_steps 150000 --f2_variant linear --load_f1 24880 --freeze_f1 --l1_reg f2_weights --l1_coeff 2
    version2=$(python versions.py)
    python -u find_minima.py --version $version2 --total_steps 150000 --f2_variant linear --load_f1_f2 $version --prune_f2_topk $k
end

# then run topk_plots.ipynb to make the last two plots.


