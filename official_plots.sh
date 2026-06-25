python interpret.py --version_json official_versions.json
cd figures
python five_planet_plot.py --version_json ../official_versions.json
python resonant_figure.py
# note: requires torch/gpu to load trained models
python period_ratio_figure.py --official --version_json ../official_versions.json

cd ..
# this will take 15-45 minutes to run the first time as the predictions are computed.
# after that, the predictions are cached
python evaluation.py --official --version_json official_versions.json
python ablations.py --version_json official_versions.json
