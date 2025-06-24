python interpret.py --version_json official_versions.json
cd figures
python five_planet_plot.py
python period_ratio_figure.py --plot --version 24880 --pysr_version 11003
python period_ratio_figure.py --special 4way --version_json ../official_versions.json
python period_ratio_figure.py --special pure_sr --version_json ../official_versions.json
python period_ratio_figure.py --special rmse --version_json ../official_versions.json
python period_ratio_figure.py --plot --version 24880 --pysr_version 11003 --pysr_model_selection 26 --rmse_diff

cd ..
python ablations.py --version_json official_versions.json
