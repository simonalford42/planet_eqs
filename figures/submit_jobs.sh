#!/usr/bin/env bash

# sbatch -J pysr300 --partition ellis run.sh --Ngrid 300 --version 24880 --pysr_version 93102 --compute

# sbatch -J five29 --partition ellis five_planet.sh --version 24880 --pysr_version 58106 --paper-ready --pysr_model_selection 29
# sbatch -J five27 --partition ellis five_planet.sh --version 24880 --pysr_version 58106 --paper-ready --pysr_model_selection 27
# sbatch -J five25 --partition ellis five_planet.sh --version 24880 --pysr_version 58106 --paper-ready --pysr_model_selection 25
# sbatch -J five23 --partition ellis five_planet.sh --version 24880 --pysr_version 58106 --paper-ready --pysr_model_selection 23
# sbatch -J five21 --partition ellis five_planet.sh --version 24880 --pysr_version 58106 --paper-ready --pysr_model_selection 21
# sbatch -J five21 --partition ellis five_planet.sh --version 24880 --pysr_version 58106 --paper-ready --pysr_model_selection 19
# sbatch -J five21 --partition ellis five_planet.sh --version 24880 --pysr_version 58106 --paper-ready --pysr_model_selection 17
# sbatch -J five21 --partition ellis five_planet.sh --version 24880 --pysr_version 58106 --paper-ready --pysr_model_selection 15
# sbatch -J five21 --partition ellis five_planet.sh --version 24880 --pysr_version 58106 --paper-ready --pysr_model_selection 13
# sbatch -J five21 --partition ellis five_planet.sh --version 24880 --pysr_version 58106 --paper-ready --pysr_model_selection 11


# sbatch --time=48:00:00 --array=0-3 -J petit16 --partition gpu run.sh --Ngrid 16 --compute --petit --job_array
# sbatch --time=48:00:00 --array=0-24 -J petit300 --partition gpu run.sh --Ngrid 300 --compute --petit --job_array

# sbatch -J five --partition ellis five_planet.sh --version 24880 --pysr_version 11003 --paper-ready
# sbatch -J five --partition gpu five_planet.sh --version 24880 --pysr_version 11003 --paper-ready --turbo --pysr_model_selection 1
# ...
# sbatch -J five --partition gpu five_planet.sh --version 24880 --pysr_version 11003 --paper-ready --turbo --pysr_model_selection 30

# python period_ratio_figure.py --collate --Ngrid 300 --ground_truth --max_t 1e9

# sbatch -J bnn300 --partition gpu run.sh --Ngrid 300 --compute

# sbatch --time=20-00:00:00 --array=0-899%100 job_array.sh --Ngrid 300 --max_t 1e9 --compute

# testing running more than 100 at once using multiple job arrays
# sbatch --array=0-899%100 job_array.sh --Ngrid 100 --max_t 1e6 --compute --parallel_ix 1 --parallel_total 2
# sbatch --array=0-999%100 job_array.sh --Ngrid 100 --max_t 1e6 --compute --parallel_ix 2 --parallel_total 10
# sbatch --array=0-999%100 job_array.sh --Ngrid 100 --max_t 1e6 --compute --parallel_ix 3 --parallel_total 10
# sbatch --array=0-999%100 job_array.sh --Ngrid 100 --max_t 1e6 --compute --parallel_ix 4 --parallel_total 10
# sbatch --array=0-999%100 job_array.sh --Ngrid 100 --max_t 1e6 --compute --parallel_ix 5 --parallel_total 10
# sbatch --array=0-999%100 job_array.sh --Ngrid 100 --max_t 1e6 --compute --parallel_ix 6 --parallel_total 10
# sbatch --array=0-999%100 job_array.sh --Ngrid 100 --max_t 1e6 --compute --parallel_ix 7 --parallel_total 10
# sbatch --array=0-999%100 job_array.sh --Ngrid 100 --max_t 1e6 --compute --parallel_ix 8 --parallel_total 10
# sbatch --array=0-999%100 job_array.sh --Ngrid 100 --max_t 1e6 --compute --parallel_ix 9 --parallel_total 10

# sbatch --array=0-899%100 job_array.sh --Ngrid 42 --max_t 1e5 --compute --parallel_ix 0
# sbatch --array=0-899%100 job_array.sh --Ngrid 42 --max_t 1e5 --compute --parallel_ix 0
# sbatch --array=0-899%100 job_array.sh --Ngrid 42 --max_t 1e5 --compute --parallel_ix 1
# sbatch --array=0-89%50 job_array.sh --Ngrid 13 --max_t 1e5 --compute --parallel_ix 0 --parallel_total 2
# sbatch --array=0-89%50 job_array.sh --Ngrid 13 --max_t 1e5 --compute --parallel_ix 1 --parallel_total 2
# sbatch --array=0-8%4 job_array.sh --Ngrid 11 --max_t 1e5 --compute --parallel_ix 0 --parallel_total 2
# sbatch --array=0-8%4 job_array.sh --Ngrid 11 --max_t 1e5 --compute --parallel_ix 1 --parallel_total 2

# testing to see if I can run more than 100 jobs at once
# sbatch --array=0-499%100 job_array.sh --Ngrid 26 --max_t 1e5 --compute

# sbatch -J five --partition ellis five_planet.sh --turbo --paper-ready
# sbatch --array=0-899%100 job_array.sh --Ngrid 100 --max_t 1e7 --compute

# command for 100x100 1e8 photo
# job_id=""
# for ix in {0..10}; do
#     job_id=$(sbatch ${job_id:+"--dependency=afterany:$job_id"} --array=0-999%100 job_array.sh --Ngrid 100 --max_t 1e8 --compute --parallel_ix $ix | cut -d ' ' -f 4)
# done

# sbatch --array=0-899%100 job_array.sh --Ngrid 100 --max_t 1e8 --compute --parallel_ix $ix | cut -d ' ' -f 4)

# testing on a much smaller size to start
# sbatch --array=0-25%10 job_array.sh --Ngrid 9 --max_t 1e6 --compute
# sbatch --array=0-899%100 job_array.sh --Ngrid 41 --max_t 1e6 --compute
# sbatch --array=0-35%10 job_array.sh --Ngrid 12 --max_t 1e6 --compute

# sbatch --array=0-899%100 job_array.sh --Ngrid 100 --max_t 1e6 --compute
# sbatch --array=0-899%100 job_array.sh --Ngrid 100 --max_t 1e7 --compute

# command for 400x400 1e8 photo
# job_id=""
# for ix in {0..159}; do
#     job_id=$(sbatch ${job_id:+"--dependency=afterany:$job_id"} --array=0-999%100 job_array.sh --Ngrid 400 --max_t 1e8 --compute --parallel_ix $ix | cut -d ' ' -f 4)
# done

# sbatch -J five --partition ellis five_planet.sh --turbo
# sbatch --array=0-8 job_array.sh --Ngrid 3 --max_t 100000000 --compute
# sbatch --array=0-99 job_array.sh --Ngrid 10 --max_t 1000000 --compute
# sbatch --array=0-15 job_array.sh --Ngrid 4 --max_t 100000 --compute

# sbatch -J five --partition ellis five_planet.sh --paper-ready --version 12370 --pysr_version 22943
# sbatch -J five --partition ellis five_planet.sh --paper-ready --pysr_model_selection 14

# sbatch -J five --partition ellis five_planet.sh --paper-ready --extrapolate
# sbatch -J five --partition ellis five_planet.sh --paper-ready --extrapolate --turbo
# sbatch -J five --partition ellis five_planet.sh --paper-ready --turbo
# sbatch -J five --partition ellis five_planet.sh --paper-ready

# sbatch -J five --partition ellis five_planet.sh --version 24880 --pysr_version 11003 --paper-ready
# sbatch -J five --partition ellis five_planet.sh --version 24880 --pysr_version 11003 --fix_variance --paper-ready
# sbatch -J five --partition ellis five_planet.sh --version 24880 --pysr_version 11003 --paper-ready
# sbatch -J five --partition gpu five_planet.sh --version 24880 --fix_variance
# sbatch -J five --partition gpu five_planet.sh --version 24880 --fix_variance --pysr_version 11003

# sbatch -J p35 --partition gpu run.sh --Ngrid 1600 --compute --megno --parallel_ix 35 --parallel_total 40

# computing using cached is really fast: should only take about 15 minutes
# sbatch -J nn --partition gpu run.sh --Ngrid 1600 --compute --version 43139

# sbatch -J cache100 --partition gpu run.sh --create_input_cache --Ngrid 100
# sbatch -J cache200 --partition gpu run.sh --create_input_cache --Ngrid 200
# sbatch -J cache400 --partition gpu run.sh --create_input_cache --Ngrid 400
# sbatch -J cache800 --partition gpu run.sh --create_input_cache --Ngrid 800

# sbatch -J as_eq_plot1 --partition gpu run.sh --paper-ready --version 9259 --pysr_version 89776
# sbatch -J as_eq_plot2 --partition gpu run.sh --paper-ready --version 10290 --pysr_version 69083

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

# Missing files: [29, 30, 31, 32, 33, 46, 50, 51, 52, 53, 69, 70, 71, 72, 77, 78, 79]
# sbatch -J p29 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 29 --parallel_total 80
# sbatch -J p30 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 30 --parallel_total 80
# sbatch -J p31 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 31 --parallel_total 80
# sbatch -J p32 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 32 --parallel_total 80
# sbatch -J p33 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 33 --parallel_total 80
# sbatch -J p46 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 46 --parallel_total 80
# sbatch -J p50 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 50 --parallel_total 80
# sbatch -J p51 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 51 --parallel_total 80
# sbatch -J p52 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 52 --parallel_total 80
# sbatch -J p53 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 53 --parallel_total 80
# sbatch -J p69 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 69 --parallel_total 80
# sbatch -J p70 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 70 --parallel_total 80
# sbatch -J p71 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 71 --parallel_total 80
# sbatch -J p72 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 72 --parallel_total 80
# sbatch -J p77 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 77 --parallel_total 80
# sbatch -J p78 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 78 --parallel_total 80
# sbatch -J p79 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 79 --parallel_total 80

# sbatch -J p0 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 0 --parallel_total 80
# sbatch -J p1 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 1 --parallel_total 80
# sbatch -J p2 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 2 --parallel_total 80
# sbatch -J p3 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 3 --parallel_total 80
# sbatch -J p4 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 4 --parallel_total 80
# sbatch -J p5 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 5 --parallel_total 80
# sbatch -J p6 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 6 --parallel_total 80
# sbatch -J p7 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 7 --parallel_total 80
# sbatch -J p8 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 8 --parallel_total 80
# sbatch -J p9 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 9 --parallel_total 80
# sbatch -J p10 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 10 --parallel_total 80
# sbatch -J p11 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 11 --parallel_total 80
# sbatch -J p12 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 12 --parallel_total 80
# sbatch -J p13 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 13 --parallel_total 80
# sbatch -J p14 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 14 --parallel_total 80
# sbatch -J p15 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 15 --parallel_total 80
# sbatch -J p16 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 16 --parallel_total 80
# sbatch -J p17 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 17 --parallel_total 80
# sbatch -J p18 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 18 --parallel_total 80
# sbatch -J p19 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 19 --parallel_total 80
# sbatch -J p20 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 20 --parallel_total 80
# sbatch -J p21 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 21 --parallel_total 80
# sbatch -J p22 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 22 --parallel_total 80
# sbatch -J p23 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 23 --parallel_total 80
# sbatch -J p24 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 24 --parallel_total 80
# sbatch -J p25 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 25 --parallel_total 80
# sbatch -J p26 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 26 --parallel_total 80
# sbatch -J p27 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 27 --parallel_total 80
# sbatch -J p28 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 28 --parallel_total 80
# sbatch -J p29 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 29 --parallel_total 80
# sbatch -J p30 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 30 --parallel_total 80
# sbatch -J p31 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 31 --parallel_total 80
# sbatch -J p32 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 32 --parallel_total 80
# sbatch -J p33 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 33 --parallel_total 80
# sbatch -J p34 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 34 --parallel_total 80
# sbatch -J p35 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 35 --parallel_total 80
# sbatch -J p36 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 36 --parallel_total 80
# sbatch -J p37 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 37 --parallel_total 80
# sbatch -J p38 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 38 --parallel_total 80
# sbatch -J p39 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 39 --parallel_total 80
# sbatch -J p40 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 40 --parallel_total 80
# sbatch -J p41 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 41 --parallel_total 80
# sbatch -J p42 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 42 --parallel_total 80
# sbatch -J p43 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 43 --parallel_total 80
# sbatch -J p44 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 44 --parallel_total 80
# sbatch -J p45 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 45 --parallel_total 80
# sbatch -J p46 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 46 --parallel_total 80
# sbatch -J p47 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 47 --parallel_total 80
# sbatch -J p48 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 48 --parallel_total 80
# sbatch -J p49 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 49 --parallel_total 80
# sbatch -J p50 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 50 --parallel_total 80
# sbatch -J p51 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 51 --parallel_total 80
# sbatch -J p52 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 52 --parallel_total 80
# sbatch -J p53 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 53 --parallel_total 80
# sbatch -J p54 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 54 --parallel_total 80
# sbatch -J p55 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 55 --parallel_total 80
# sbatch -J p56 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 56 --parallel_total 80
# sbatch -J p57 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 57 --parallel_total 80
# sbatch -J p58 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 58 --parallel_total 80
# sbatch -J p59 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 59 --parallel_total 80
# sbatch -J p60 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 60 --parallel_total 80
# sbatch -J p61 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 61 --parallel_total 80
# sbatch -J p62 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 62 --parallel_total 80
# sbatch -J p63 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 63 --parallel_total 80
# sbatch -J p64 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 64 --parallel_total 80
# sbatch -J p65 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 65 --parallel_total 80
# sbatch -J p66 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 66 --parallel_total 80
# sbatch -J p67 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 67 --parallel_total 80
# sbatch -J p68 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 68 --parallel_total 80
# sbatch -J p69 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 69 --parallel_total 80
# sbatch -J p70 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 70 --parallel_total 80
# sbatch -J p71 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 71 --parallel_total 80
# sbatch -J p72 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 72 --parallel_total 80
# sbatch -J p73 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 73 --parallel_total 80
# sbatch -J p74 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 74 --parallel_total 80
# sbatch -J p75 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 75 --parallel_total 80
# sbatch -J p76 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 76 --parallel_total 80
# sbatch -J p77 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 77 --parallel_total 80
# sbatch -J p78 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 78 --parallel_total 80
# sbatch -J p79 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 79 --parallel_total 80
# sbatch -J p80 --partition gpu run.sh --Ngrid 1600 --petit --compute --parallel_ix 80 --parallel_total 80

# sbatch -J p0 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 0 --parallel_total 40
# sbatch -J p1 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 1 --parallel_total 40
# sbatch -J p2 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 2 --parallel_total 40
# sbatch -J p3 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 3 --parallel_total 40
# sbatch -J p4 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 4 --parallel_total 40
# sbatch -J p5 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 5 --parallel_total 40
# sbatch -J p6 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 6 --parallel_total 40
# sbatch -J p7 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 7 --parallel_total 40
# sbatch -J p8 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 8 --parallel_total 40
# sbatch -J p9 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 9 --parallel_total 40
# sbatch -J p10 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 10 --parallel_total 40
# sbatch -J p11 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 11 --parallel_total 40
# sbatch -J p12 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 12 --parallel_total 40
# sbatch -J p13 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 13 --parallel_total 40
# sbatch -J p14 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 14 --parallel_total 40
# sbatch -J p15 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 15 --parallel_total 40
# sbatch -J p16 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 16 --parallel_total 40
# sbatch -J p17 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 17 --parallel_total 40
# sbatch -J p18 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 18 --parallel_total 40
# sbatch -J p19 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 19 --parallel_total 40
# sbatch -J p20 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 20 --parallel_total 40
# sbatch -J p21 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 21 --parallel_total 40
# sbatch -J p22 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 22 --parallel_total 40
# sbatch -J p23 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 23 --parallel_total 40
# sbatch -J p24 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 24 --parallel_total 40
# sbatch -J p25 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 25 --parallel_total 40
# sbatch -J p26 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 26 --parallel_total 40
# sbatch -J p27 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 27 --parallel_total 40
# sbatch -J p28 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 28 --parallel_total 40
# sbatch -J p29 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 29 --parallel_total 40
# sbatch -J p30 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 30 --parallel_total 40
# sbatch -J p31 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 31 --parallel_total 40
# sbatch -J p32 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 32 --parallel_total 40
# sbatch -J p33 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 33 --parallel_total 40
# sbatch -J p34 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 34 --parallel_total 40
# sbatch -J p35 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 35 --parallel_total 40
# sbatch -J p36 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 36 --parallel_total 40
# sbatch -J p37 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 37 --parallel_total 40
# sbatch -J p38 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 38 --parallel_total 40
# sbatch -J p39 --partition gpu run.sh --Ngrid 1600 --compute --petit --parallel_ix 39 --parallel_total 40

# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 20, 35, 36, 39]
# sbatch -J p0 --partition gpu run.sh --Ngrid 1600 --compute --parallel_ix 0 --parallel_total 40
# sbatch -J p1 --partition gpu run.sh --Ngrid 1600 --compute --parallel_ix 1 --parallel_total 40
# sbatch -J p2 --partition gpu run.sh --Ngrid 1600 --compute --parallel_ix 2 --parallel_total 40
# sbatch -J p3 --partition gpu run.sh --Ngrid 1600 --compute --parallel_ix 3 --parallel_total 40
# sbatch -J p4 --partition gpu run.sh --Ngrid 1600 --compute ---parallel_ix 4 --parallel_total 40
# sbatch -J p5 --partition gpu run.sh --Ngrid 1600 --compute ---parallel_ix 5 --parallel_total 40
# sbatch -J p6 --partition gpu run.sh --Ngrid 1600 --compute ---parallel_ix 6 --parallel_total 40
# sbatch -J p7 --partition gpu run.sh --Ngrid 1600 --compute ---parallel_ix 7 --parallel_total 40
# sbatch -J p8 --partition gpu run.sh --Ngrid 1600 --compute ---parallel_ix 8 --parallel_total 40
# sbatch -J p9 --partition gpu run.sh --Ngrid 1600 --compute ---parallel_ix 9 --parallel_total 40
# sbatch -J p10 --partition gpu run.sh --Ngrid 1600 --compute --parallel_ix 10 --parallel_total 40
# sbatch -J p11 --partition gpu run.sh --Ngrid 1600 --compute --parallel_ix 11 --parallel_total 40
# sbatch -J p12 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 12 --parallel_total 40
# sbatch -J p13 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 13 --parallel_total 40
# sbatch -J p14 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 14 --parallel_total 40
# sbatch -J p15 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 15 --parallel_total 40
# sbatch -J p16 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 16 --parallel_total 40
# sbatch -J p17 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 17 --parallel_total 40
# sbatch -J p18 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 18 --parallel_total 40
# sbatch -J p19 --partition gpu run.sh --Ngrid 1600 --compute --parallel_ix 19 --parallel_total 40
# sbatch -J p20 --partition gpu run.sh --Ngrid 1600 --compute --parallel_ix 20 --parallel_total 40
# sbatch -J p21 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 21 --parallel_total 40
# sbatch -J p22 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 22 --parallel_total 40
# sbatch -J p23 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 23 --parallel_total 40
# sbatch -J p24 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 24 --parallel_total 40
# sbatch -J p25 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 25 --parallel_total 40
# sbatch -J p26 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 26 --paralle5915574l_total 40
# sbatch -J p27 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 27 --parallel_total 40
# sbatch -J p28 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 28 --parallel_total 40
# sbatch -J p29 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 29 --parallel_total 40
# sbatch -J p30 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 30 --parallel_total 40
# sbatch -J p31 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 31 --parallel_total 40
# sbatch -J p32 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 32 --parallel_total 40
# sbatch -J p33 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 33 --parallel_total 40
# sbatch -J p34 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 34 --parallel_total 40
# sbatch -J p35 --partition gpu run.sh --Ngrid 1600 --compute --parallel_ix 35 --parallel_total 40
# sbatch -J p36 --partition gpu run.sh --Ngrid 1600 --compute --parallel_ix 36 --parallel_total 40
# sbatch -J p37 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 37 --parallel_total 40
# sbatch -J p38 --partition gpu run.sh --Ngrid 1600 --compute --version 43139 --parallel_ix 38 --parallel_total 40
# sbatch -J p39 --partition gpu run.sh --Ngrid 1600 --compute --parallel_ix 39 --parallel_total 40
