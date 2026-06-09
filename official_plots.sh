python interpret.py --version_json official_versions.json
cd figures
python five_planet_plot.py --version_json ../official_versions.json
python period_ratio_figure.py --special official_plots --version_json ../official_versions.json
python period_ratio_figure.py --special official_metrics --version_json ../official_versions.json
python resonant_figure.py

cd ..
python evaluation.py --official
python evaluation.py --official_figures
python ablations.py --version_json official_versions.json
