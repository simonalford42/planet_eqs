python interpret.py --version_json official_versions.json
cd figures
python five_planet_plot.py --version_json ../official_versions.json
python period_ratio_figure.py --special main --version_json ../official_versions.json
python period_ratio_figure.py --special pure_sr --version_json ../official_versions.json
python period_ratio_figure.py --special f1_features --version_json ../official_versions.json
python period_ratio_figure.py --special rmse --version_json ../official_versions.json

cd ..
python ablations.py --version_json official_versions.json

# python calc_rmse.py --eval_type nn --version 12318 --calculate_all
# python calc_rmse.py --eval_type pysr --version 12318 --pysr_version 93102 --pysr_model_selection 29 --calculate_all
# python calc_rmse.py --eval_type petit --calculate_all
# python calc_rmse.py --eval_type pure_sr --pysr_version 83941 --calculate_all --pysr_model_selection 40
# python calc_rmse.py --eval_type pysr --version 28114 --pysr_version 93890 --calculate_all --pysr_model_selection 43

# python calc_rmse.py --eval_type nn --calculate_all --version 24880
# python calc_rmse.py --eval_type pysr --calculate_all --version 24880 --pysr_version 11003 --pysr_model_selection 26
# python calc_rmse.py --eval_type pure_sr --calculate_all --pysr_version 72420 --pysr_model_selection 50
# python calc_rmse.py --eval_type pysr --calculate_all --version 28114 --pysr_version 41564 --pysr_model_selection 35

# python calc_rmse.py --eval_type nn --version 12318 --calculate_all
# python calc_rmse.py --eval_type pysr --version 12318 --pysr_version 93102 --pysr_model_selection 29 --calculate_all
# python calc_rmse.py --eval_type petit --calculate_all
# python calc_rmse.py --eval_type pure_sr --pysr_version 83941 --calculate_all --pysr_model_selection 40
# python calc_rmse.py --eval_type pysr --version 28114 --pysr_version 93890 --calculate_all --pysr_model_selection 43

# python calc_rmse.py --eval_type nn --calculate_all --version 24880
# python calc_rmse.py --eval_type pysr --calculate_all --version 24880 --pysr_version 11003 --pysr_model_selection 26
# python calc_rmse.py --eval_type pure_sr --calculate_all --pysr_version 72420 --pysr_model_selection 50
# python calc_rmse.py --eval_type pysr --calculate_all --version 28114 --pysr_version 41564 --pysr_model_selection 35
