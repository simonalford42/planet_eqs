python interpret.py --version_json official_versions.json
cd figures
python five_planet_plot.py --version_json ../official_versions.json
python period_ratio_figure.py --special main --version_json ../official_versions.json
python period_ratio_figure.py --special pure_sr --version_json ../official_versions.json
python period_ratio_figure.py --special f1_features --version_json ../official_versions.json
python period_ratio_figure.py --special rmse --version_json ../official_versions.json

cd ..
python ablations.py --version_json official_versions.json
