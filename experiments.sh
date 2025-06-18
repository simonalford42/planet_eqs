
# train NN with sparse linear psi
python -u find_minima.py --version 0 --total_steps 150000 --l1_reg weights --l1_coeff 2
python -u find_minima.py --version 1 --total_steps 150000 --load_f1_f2 0 --prune_f1_topk 2

# run symbolic regression to distill phi into equations
# we ran with 32 cores for 8 hours (results will be worse if fewer cores/hours are used)
python sr.py --nn_version 1 --version 1

# we ran with 32 cores for 8 hours
# run pure sr w/ learned psi)
python pure_sr.py --version 2
# run pure sr w/ mean/std of input variables
# 28114 is a NN that has psi be identity, so features passed in are mean/std of input variables
python sr.py --target f2_direct --version 3 --nn_version 28114

# calculate rmse scores for neural network
python calc_rmse.py --eval_type nn --dataset all --version 1
# calculate rmse scores for equations
python calc_rmse.py --eval_type pysr --dataset all --version 1 --pysr_version 1
# calculate rmse scores for Petit
python calc_rmse.py --eval_type petit --dataset all
# calculate rmse scores for Pure SR
python calc_rmse.py --eval_type pure_sr --dataset all --pysr_version 2
python calc_rmse.py --eval_type pysr --dataset all --version 28114 --pysr_version 3

# store best complexity for each sr/pure sr version
sr_c=$(python calc_rmse.py --version 1 --pysr_version 1 --best_complexity)
puresr_c=$(python calc_rmse.py --pure_sr --pysr_version 2 --best_complexity)
puresr2_c=$(python calc_rmse.py --version 28114 --pysr_version 3 --best_complexity)

cd figures

# calculate five planet results
python five_planet.py --version 1 --pysr_version 1 --pysr_model_selection $sr_c --N 5000 --turbo --extrapolate
python five_planet.py --pure_sr --pysr_version 2 --pysr_model_selection $puresr_c --N 5000 --turbo --extrapolate
python five_planet.py --version 28114 --pysr_version 3 --pysr_model_selection $puresr2_c --N 5000 --turbo --extrapolate

# create five planet plots
main_path='five_planet_figures/v1_pysr1_ms=('$sr_c')_N=5000_turbo_extrapolate.csv'
pure_sr_path='five_planet_figures/v24880_pysr2_ms=('$puresr_c')_N=5000_turbo_extrapolate.csv'
pure_sr2_path='five_planet_figures/v28114_pysr3_ms=('$puresr_c')_N=5000_turbo_extrapolate.csv'
# TODO: figure out what we're doing with puresr2
python five_planet_plot.py $main_path $pure_sr_path

# calculate and plot period ratio results
python period_ratio_figure.py --Ngrid 300 --version 1 --compute --plot --rmse
# need to compute for NN before computing for SR, even though we don't care about this NN
python period_ratio_figure.py --Ngrid 300 --version 28114 --compute
python period_ratio_figure.py --Ngrid 300 --version 1 --pysr_version 1 --compute --plot --rmse
python period_ratio_figure.py --special 4way --Ngrid 300 --version 1 --pysr_version 1 --pysr_model_selection $sr_c
python period_ratio_figure.py --Ngrid 300 --pure_sr --pysr_version 2 --pysr_model_selection $puresr_c --compute --plot --rmse
python period_ratio_figure.py --Ngrid 300 --version 28114 --pysr_version 3 --pysr_model_selection $puresr2_c --compute --plot --rmse

cd ..

# topk comparisons. uses nn versions 3, 4, 5, 13, 14, 15 and sr versions 13, 14, 15
for k in {3..5} do
    python -u find_minima.py --version $k --total_steps 150000 --l1_reg weights --l1_coeff 2
    python -u find_minima.py --version $(10+k) --total_steps 150000 --load_f1_f2 $k --prune_f1_topk $k
    python sr.py --nn_version $(10+k) --version $(10+k)
    python calc_rmse.py --version $(10+kh) --pysr_version $(10+k) --eval_type pysr --dataset all
end

# f2 linear models. uses nn versions 6, 20, 22, 25, 30, 40
# first train model with linear f2, allowing f1 weights to be modified (but still sparse)
python -u find_minima.py --version 6 --total_steps 150000 --f2_variant linear --load_f1 2 --l1_reg f2_weights --l1_coeff 2
for k in (0 2 5 10 20) do
    python -u find_minima.py --version $(20+k) --total_steps 150000 --f2_variant linear --load_f1_f2 6  --prune_f2_topk 2 --mse_loss
    python calc_rmse.py --version $(20+k) --eval_type nn --dataset all
end

# plot the pareto comparisons
python ablations.py --version_json ablation_versions2.json --path graphics/pareto_comparison.pdf

