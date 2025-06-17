cd figures
python five_planet_plot.py
python period_ratio_figure.py --special 4way --Ngrid 300 --version 24880 --pysr_version 11003 --pysr_model_selection 26
# pot period ratio figure predictions for increasing equation complexity
python period_ratio_figure.py --plot --Ngrid 300 --version 24880 --pysr_version 11003
python interpret.py
