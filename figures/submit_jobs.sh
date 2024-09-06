#!/usr/bin/env bash


sbatch -J as_eq_plot1 --partition gpu run.sh --paper-ready --version 9259 --pysr_version 89776
sbatch -J as_eq_plot2 --partition gpu run.sh --paper-ready --version 10290 --pysr_version 69083

# sbatch -J eq_plot_pysr1 --partition gpu run.sh --paper-ready --version 24880 --pysr_model_selection 1 --pysr_version 11003
# sbatch -J eq_plot_pysr3 --partition gpu run.sh --paper-ready --version 24880 --pysr_model_selection 3 --pysr_version 11003
# sbatch -J eq_plot_pysr5 --partition gpu run.sh --paper-ready --version 24880 --pysr_model_selection 5 --pysr_version 11003
# sbatch -J eq_plot_pysr7 --partition gpu run.sh --paper-ready --version 24880 --pysr_model_selection 7 --pysr_version 11003
# sbatch -J eq_plot_pysr9 --partition gpu run.sh --paper-ready --version 24880 --pysr_model_selection 9 --pysr_version 11003
# sbatch -J eq_plot_pysr11 --partition gpu run.sh --paper-ready --version 24880 --pysr_model_selection 11 --pysr_version 11003
# sbatch -J eq_plot_pysr14 --partition gpu run.sh --paper-ready --version 24880 --pysr_model_selection 14 --pysr_version 11003
# sbatch -J eq_plot_pysr18 --partition gpu run.sh --paper-ready --version 24880 --pysr_model_selection 18 --pysr_version 11003
# sbatch -J eq_plot_pysr20 --partition gpu run.sh --paper-ready --version 24880 --pysr_model_selection 20 --pysr_version 11003
# sbatch -J eq_plot_pysr27 --partition gpu run.sh --paper-ready --version 24880 --pysr_model_selection 27 --pysr_version 11003
# sbatch -J eq_plot_pysr29 --partition gpu run.sh --paper-ready --version 24880 --pysr_model_selection 29 --pysr_version 11003

# sbatch -J p3 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 3 --parallel_total 80
# sbatch -J p4 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 4 --parallel_total 80
# sbatch -J p17 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 17 --parallel_total 80
# sbatch -J p72 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 72 --parallel_total 80

# sbatch -J p0 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 0 --parallel_total 80
# sbatch -J p1 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 1 --parallel_total 80
# sbatch -J p2 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 2 --parallel_total 80
# sbatch -J p3 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 3 --parallel_total 80
# sbatch -J p4 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 4 --parallel_total 80
# sbatch -J p5 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 5 --parallel_total 80
# sbatch -J p6 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 6 --parallel_total 80
# sbatch -J p7 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 7 --parallel_total 80
# sbatch -J p8 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 8 --parallel_total 80
# sbatch -J p9 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 9 --parallel_total 80
# sbatch -J p10 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 10 --parallel_total 80
# sbatch -J p11 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 11 --parallel_total 80
# sbatch -J p12 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 12 --parallel_total 80
# sbatch -J p13 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 13 --parallel_total 80
# sbatch -J p14 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 14 --parallel_total 80
# sbatch -J p15 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 15 --parallel_total 80
# sbatch -J p16 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 16 --parallel_total 80
# sbatch -J p17 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 17 --parallel_total 80
# sbatch -J p18 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 18 --parallel_total 80
# sbatch -J p19 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 19 --parallel_total 80
# sbatch -J p20 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 20 --parallel_total 80
# sbatch -J p21 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 21 --parallel_total 80
# sbatch -J p22 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 22 --parallel_total 80
# sbatch -J p23 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 23 --parallel_total 80
# sbatch -J p24 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 24 --parallel_total 80
# sbatch -J p25 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 25 --parallel_total 80
# sbatch -J p26 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 26 --parallel_total 80
# sbatch -J p27 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 27 --parallel_total 80
# sbatch -J p28 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 28 --parallel_total 80
# sbatch -J p29 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 29 --parallel_total 80
# sbatch -J p30 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 30 --parallel_total 80
# sbatch -J p31 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 31 --parallel_total 80
# sbatch -J p32 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 32 --parallel_total 80
# sbatch -J p33 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 33 --parallel_total 80
# sbatch -J p34 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 34 --parallel_total 80
# sbatch -J p35 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 35 --parallel_total 80
# sbatch -J p36 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 36 --parallel_total 80
# sbatch -J p37 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 37 --parallel_total 80
# sbatch -J p38 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 38 --parallel_total 80
# sbatch -J p39 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 39 --parallel_total 80
# sbatch -J p40 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 40 --parallel_total 80
# sbatch -J p41 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 41 --parallel_total 80
# sbatch -J p42 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 42 --parallel_total 80
# sbatch -J p43 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 43 --parallel_total 80
# sbatch -J p44 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 44 --parallel_total 80
# sbatch -J p45 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 45 --parallel_total 80
# sbatch -J p46 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 46 --parallel_total 80
# sbatch -J p47 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 47 --parallel_total 80
# sbatch -J p48 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 48 --parallel_total 80
# sbatch -J p49 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 49 --parallel_total 80
# sbatch -J p50 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 50 --parallel_total 80
# sbatch -J p51 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 51 --parallel_total 80
# sbatch -J p52 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 52 --parallel_total 80
# sbatch -J p53 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 53 --parallel_total 80
# sbatch -J p54 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 54 --parallel_total 80
# sbatch -J p55 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 55 --parallel_total 80
# sbatch -J p56 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 56 --parallel_total 80
# sbatch -J p57 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 57 --parallel_total 80
# sbatch -J p58 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 58 --parallel_total 80
# sbatch -J p59 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 59 --parallel_total 80
# sbatch -J p60 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 60 --parallel_total 80
# sbatch -J p61 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 61 --parallel_total 80
# sbatch -J p62 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 62 --parallel_total 80
# sbatch -J p63 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 63 --parallel_total 80
# sbatch -J p64 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 64 --parallel_total 80
# sbatch -J p65 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 65 --parallel_total 80
# sbatch -J p66 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 66 --parallel_total 80
# sbatch -J p67 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 67 --parallel_total 80
# sbatch -J p68 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 68 --parallel_total 80
# sbatch -J p69 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 69 --parallel_total 80
# sbatch -J p70 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 70 --parallel_total 80
# sbatch -J p71 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 71 --parallel_total 80
# sbatch -J p72 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 72 --parallel_total 80
# sbatch -J p73 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 73 --parallel_total 80
# sbatch -J p74 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 74 --parallel_total 80
# sbatch -J p75 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 75 --parallel_total 80
# sbatch -J p76 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 76 --parallel_total 80
# sbatch -J p77 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 77 --parallel_total 80
# sbatch -J p78 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 78 --parallel_total 80
# sbatch -J p79 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 79 --parallel_total 80
# sbatch -J p80 --partition gpu run.sh --Ngrid 1600 --create_input_cache --parallel_ix 80 --parallel_total 80

# # sbatch -J p0 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 0 --parallel_total 40
# # sbatch -J p1 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 1 --parallel_total 40
# # sbatch -J p2 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 2 --parallel_total 40
# # sbatch -J p3 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 3 --parallel_total 40
# # sbatch -J p4 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 4 --parallel_total 40
# # sbatch -J p5 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 5 --parallel_total 40
# # sbatch -J p6 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 6 --parallel_total 40
# # sbatch -J p7 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 7 --parallel_total 40
# # sbatch -J p8 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 8 --parallel_total 40
# # sbatch -J p9 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 9 --parallel_total 40
# # sbatch -J p10 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 10 --parallel_total 40
# # sbatch -J p11 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 11 --parallel_total 40
# # sbatch -J p12 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 12 --parallel_total 40
# # sbatch -J p13 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 13 --parallel_total 40
# # sbatch -J p14 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 14 --parallel_total 40
# # sbatch -J p15 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 15 --parallel_total 40
# # sbatch -J p16 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 16 --parallel_total 40
# # sbatch -J p17 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 17 --parallel_total 40
# # sbatch -J p18 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 18 --parallel_total 40
# # sbatch -J p19 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 19 --parallel_total 40
# # sbatch -J p20 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 20 --parallel_total 40
# # sbatch -J p21 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 21 --parallel_total 40
# # sbatch -J p22 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 22 --parallel_total 40
# # sbatch -J p23 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 23 --parallel_total 40
# # sbatch -J p24 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 24 --parallel_total 40
# # sbatch -J p25 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 25 --parallel_total 40
# # sbatch -J p26 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 26 --parallel_total 40
# # sbatch -J p27 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 27 --parallel_total 40
# # sbatch -J p28 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 28 --parallel_total 40
# # sbatch -J p29 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 29 --parallel_total 40
# # sbatch -J p30 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 30 --parallel_total 40
# # sbatch -J p31 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 31 --parallel_total 40
# # sbatch -J p32 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 32 --parallel_total 40
# # sbatch -J p33 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 33 --parallel_total 40
# # sbatch -J p34 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 34 --parallel_total 40
# # sbatch -J p35 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 35 --parallel_total 40
# # sbatch -J p36 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 36 --parallel_total 40
# # sbatch -J p37 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 37 --parallel_total 40
# # sbatch -J p38 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 38 --parallel_total 40
# # sbatch -J p39 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 39 --parallel_total 40
