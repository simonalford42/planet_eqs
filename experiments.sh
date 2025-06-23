################################################################################
### This script has all the commands to rerun the experiments in the paper.
### Actually running the experiments with this script will take a long time.
### Instead you should run them in parallel using a job scheduler like slurm.
### To generate plots using existing data from the paper, use official.sh
################################################################################

# exit on error
set -e

# train NN with sparse linear psi
python -u find_minima.py --version 0 --total_steps 150000 --l1_reg weights --l1_coeff 2
python -u find_minima.py --version 1 --total_steps 150000 --load_f1_f2 0 --prune_f1_topk 2

# run symbolic regression to distill phi into equations. we ran with 32 cores for 8 hours
python sr.py --nn_version 1 --version 1 --target f2 --time_in_hours 8

# run pure sr w/ learned psi. we ran with 32 cores for 8 hours
python pure_sr.py --version 2 --loss_fn ll --time_in_hours 8

# run pure sr w/ mean/std of input variables
# 28114 is a NN that has psi be the identity function, so features passed in are mean/std of input variables
python sr.py --target f2_direct --version 3 --nn_version 28114 --loss_fn ll --time_in_hours 8 --max_size 45

# train mse optimized variants of the models
python -u find_minima.py --version 2 --total_steps 150000 --load_f1_f2 1 --mse_loss
python sr.py --nn_version 1 --version 4 --target f2_direct --loss_fn mse --time_in_hours 8
python pure_sr.py --version 5 --loss_fn mse --time_in_hours 8
python sr.py --target f2_direct --version 6 --nn_version 28114 --loss_fn mse --time_in_hours 8

# topk comparisons. uses nn versions 3, 4, 5, 13, 14, 15 and sr versions 13, 14, 15
for k in {3..5}; do
    python -u find_minima.py --version $k --total_steps 150000 --l1_reg weights --l1_coeff 2 --mse_loss
    python -u find_minima.py --version $((10+k)) --total_steps 150000 --load_f1_f2 $k --prune_f1_topk $k --mse_loss
    python sr.py --nn_version $((10+k)) --version $((10+k)) --target f2 --time_in_hours 8
    python calc_rmse.py --version $((10+k)) --pysr_version $((10+k)) --eval_type pysr --dataset all
done

# f2 linear models. uses nn versions 6, 20, 22, 25, 30, 40
# first train model with linear f2, allowing f1 weights to be modified (but still sparse)
python -u find_minima.py --version 6 --total_steps 150000 --f2_variant linear --load_f1 2 --l1_reg f2_weights --l1_coeff 2 --mse_loss
for k in 0 2 5 10 20; do
    python -u find_minima.py --version $((20+k)) --total_steps 150000 --f2_variant linear --load_f1_f2 6  --prune_f2_topk 2 --mse_loss
    python calc_rmse.py --version $((20+k)) --eval_type nn --dataset all
done

# calculate rmse scores for everything, and store best complexity for each sr/pure sr version
python calc_rmse.py --eval_type nn --dataset all --version 1
python calc_rmse.py --eval_type pysr --dataset all --version 1 --pysr_version 1
python calc_rmse.py --eval_type petit --dataset all
python calc_rmse.py --eval_type pure_sr --dataset all --pysr_version 2
python calc_rmse.py --eval_type pysr --dataset all --version 28114 --pysr_version 3
# mse versions
python calc_rmse.py --eval_type nn --dataset all --version 2
python calc_rmse.py --eval_type pysr --dataset all --version 1 --pysr_version 4
python calc_rmse.py --eval_type pure_sr --dataset all --pysr_version 5
python calc_rmse.py --eval_type pysr --dataset all --version 28114 --pysr_version 6
# store the best complexity for each model
sr_c=$(python calc_rmse.py --best_complexity --eval_type pysr --dataset all --version 1 --pysr_version 1)
puresr_c=$(python calc_rmse.py --best_complexity --eval_type pure_sr --dataset all --pysr_version 2)
puresr2_c=$(python calc_rmse.py --best_complexity --eval_type pysr --dataset all --version 28114 --pysr_version 3)
sr_mse_c=$(python calc_rmse.py --best_complexity --eval_type pysr --dataset all --version 1 --pysr_version 4)
puresr_mse_c=$(python calc_rmse.py --best_complexity --eval_type pure_sr --dataset all --pysr_version 5)
puresr2_mse_c=$(python calc_rmse.py --best_complexity --eval_type pysr --dataset all --version 28114 --pysr_version 6)

# store all of the model versions and complexities in a json file
cat <<EOF > model_versions.json
{
  "nn_version": 1,
  "pysr_version": 1,
  "pysr_model_selection": $sr_c,
  "pure_sr_version": 2,
  "pure_sr_model_selection": $puresr_c,
  "pure_sr2_version": 3,
  "pure_sr2_model_selection": $puresr2_c,

  "mse_nn_version": 2,
  "mse_pysr_version": 4,
  "mse_pysr_model_selection": $sr_mse_c,
  "mse_pure_sr_version": 5,
  "mse_pure_sr_model_selection": $puresr_mse_c,
  "mse_pure_sr2_version": 6,
  "mse_pure_sr2_model_selection": $puresr2_mse_c,

  "k": {
    "2":  { "version": 1,  "pysr_version": 1  },
    "3":  { "version": 13, "pysr_version": 13 },
    "4":  { "version": 14, "pysr_version": 14 },
    "5":  { "version": 15, "pysr_version": 15 }
  },
  "f2_linear": {
    "0":  20,
    "2":  22,
    "5":  25,
    "10": 30,
    "20": 40
  }
}
EOF

cd figures

# calculate five planet results
python five_planet.py --version 1 --pysr_version 1 --pysr_model_selection $sr_c --N 5000 --turbo --extrapolate
python five_planet.py --pure_sr --pysr_version 2 --pysr_model_selection $puresr_c --N 5000 --turbo --extrapolate
python five_planet.py --version 28114 --pysr_version 3 --pysr_model_selection $puresr2_c --N 5000 --turbo --extrapolate

# create the five planet plots
python five_planet_plot.py --version_json ../model_versions.json

# calculate and plot period ratio results
python period_ratio_figure.py --version 1 --compute
python period_ratio_figure.py --version 1 --pysr_version 1 --compute --plot
python period_ratio_figure.py --pure_sr --pysr_version 2 --pysr_model_selection $puresr_c --compute
python period_ratio_figure.py --version 28114 --pysr_version 3 --pysr_model_selection $puresr2_c --compute
python period_ratio_figure.py --special 4way --version_json ../model_versions.json --pdf
python period_ratio_figure.py --special pure_sr --version_json ../model_versions.json --pdf
python period_ratio_figure.py --special rmse --version_json ../model_versions.json

cd ..
# table 3c, print equations, etc.
python interpret.py --version_json model_versions.json

# plot the pareto comparisons
python ablations.py --version_json model_versions.json

