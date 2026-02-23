python interpret.py --version_json official_versions.json
cd figures
python five_planet_plot.py --version_json ../official_versions.json
python period_ratio_figure.py --special main --version_json ../official_versions.json
python period_ratio_figure.py --special pure_sr --version_json ../official_versions.json
python period_ratio_figure.py --special f1_features --version_json ../official_versions.json
python period_ratio_figure.py --special rmse --version_json ../official_versions.json

cd ..
python ablations.py --version_json official_versions.json

# for calculating metrics for the official versions, which are used in interpret.py in the printed table
# python calc_rmse.py --eval_type nn --dataset all --version 24880
# python calc_rmse.py --eval_type pysr --dataset all --version 24880 --pysr_version 11003
# python calc_rmse.py --eval_type petit --dataset all
# python calc_rmse.py --eval_type pure_sr --dataset all --pysr_version 72420
# python calc_rmse.py --eval_type pysr --dataset all --version 28114 --pysr_version 41564

# python calc_rmse.py --eval_type nn --dataset test --version 12318
# python calc_rmse.py --eval_type nn --dataset random --version 12318
# python calc_rmse.py --eval_type pysr --dataset all --version 24880 --pysr_version 11003
# python calc_rmse.py --eval_type petit --dataset all
# python calc_rmse.py --eval_type pure_sr --dataset all --pysr_version 72420
# python calc_rmse.py --eval_type pysr --dataset all --version 28114 --pysr_version 41564
