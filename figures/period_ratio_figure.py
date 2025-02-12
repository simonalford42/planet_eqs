import pysr
import warnings
# import spock_reg_model
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
sys.path.append('../')

import rebound
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import modules
import spock
import utils2
import pickle
from utils2 import assert_equal
import multiprocessing as mp
import argparse
import time

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

INSTABILITY_TIME_LABEL = r"$\log_{10}(T_{\rm inst})$"
MEGNO_LABEL = r"$\log_{10}(\rm MEGNO-2)$"

GROUND_TRUTH_MAX_T = None # Assigned in get_args function


'''
Example commands:

# compute and plot for BNN predictions
python period_ratio_figure.py --Ngrid 4 --version 24880 --compute
python period_ratio_figure.py --Ngrid 4 --version 24880 --plot

# compute and plot for pysr f2 (need to have computed for BNN before running this)
python period_ratio_figure.py --Ngrid 4 --version 24880 --pysr_version 33060 --compute
python period_ratio_figure.py --Ngrid 4 --version 24880 --pysr_version 33060 --plot

# compute using 4 parallel jobs
python period_ratio_figure.py --Ngrid 400 --version 24880 --compute --parallel_ix 0 --parallel_total 4
python period_ratio_figure.py --Ngrid 400 --version 24880 --compute --parallel_ix 1 --parallel_total 4
python period_ratio_figure.py --Ngrid 400 --version 24880 --compute --parallel_ix 2 --parallel_total 4
python period_ratio_figure.py --Ngrid 400 --version 24880 --compute --parallel_ix 3 --parallel_total 4

# collate the parallel results and save
python period_ratio_figure.py --Ngrid 400 --version 24880 --collate

# compute and plot for petit predictions
python period_ratio_figure.py --Ngrid 4 --petit --compute
python period_ratio_figure.py --Ngrid 4 --petit --plot

# compute and plot for megno predictions
python period_ratio_figure.py --Ngrid 4 --megno --compute
python period_ratio_figure.py --Ngrid 4 --megno --plot

# compute and plot for ground truth predictions using job array
python period_ratio_figure.py --Ngrid 100 --compute --ground_truth --max_t 1e9 --job_array
python period_ratio_figure.py --Ngrid 100 --collate --ground_truth
python period_ratio_figure.py --Ngrid 100 --plot --ground_truth

# create input cache for Ngrid=4
python period_ratio_figure.py --Ngrid 4 --create_input_cache

# copy needed files to local so we can plot
scopy bnn_chaos_model/figures/period_results/v=43139_ngrid=6_pysr_f2_v=33060/ ~/code/bnn_chaos_model/figures/period_results/
scopy bnn_chaos_model/figures/period_results/v=43139_ngrid=6.pkl ~/code/bnn_chaos_model/figures/period_results/
'''


def load_model(version):
    # model = spock_reg_model.load(version, seed=0)
    # return spock.NonSwagFeatureRegressor(model)
    return spock.NonSwagFeatureRegressor(version)


def get_simulation(par):
    P12, P23 = par # unpack parameters
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.add(m=1.) # Star
    sim.add(m=1e-4, P=1, theta='uniform')
    sim.add(m=1e-4, P=1/P12, theta='uniform')
    sim.add(m=1e-4, P=1/P12/P23, theta='uniform')
    sim.move_to_com()

    return sim


def get_ground_truth(sim):
    sim = sim.copy()
    sim.dt = 0.05
    sim.exit_max_distance = 20.
    sim.exit_min_distance = 0
    T = GROUND_TRUTH_MAX_T

    try:
        sim.integrate(T)
    except rebound.Escape:
        # get the current time
        return sim.t
    except rebound.Encounter:
        return sim.t

    return T

def get_ground_truth_wrapper(par):
    sim = get_simulation(par)
    out = get_ground_truth(sim)
    return {'ground_truth': out}


def get_model_prediction(sim, model, use_petit=False, use_megno=False, create_input_cache=False, ground_truth=False):
    '''
    cache: maps simulation id to X.
    '''

    if ground_truth:
        return {'ground_truth': get_ground_truth(sim)}

    sim = sim.copy()
    sim.dt = 0.05
    sim.init_megno()
    sim.exit_max_distance = 20.

    if use_megno:
        try:
            sim.integrate(1e4)
            megno = sim.calculate_megno()
            return {'megno': megno}
        except rebound.Escape:
            return None

    try:
        if create_input_cache:
            return model.predict_up_to_cached_input(sim)
        else:
            out_dict = model.predict(sim, use_petit=use_petit)
            if use_petit:
                assert_equal(out_dict['petit'].shape, (1,))
                return {
                    'petit': out_dict['petit'][0].detach().cpu().numpy(),
                }
            else:
                return {
                    'mean': out_dict['mean'][0,0].detach().cpu().numpy(),
                    'std': out_dict['std'][0,0].detach().cpu().numpy(),
                    'f1': out_dict['summary_stats'][0].detach().cpu().numpy(),
                }
    except rebound.Escape:
        return None
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return None


def get_centered_grid(xlist, ylist, probs):
    # assumes uniformly spaced values in x and y (can have different lengths)
    dx = xlist[1]-xlist[0]
    dy = ylist[1]-ylist[0]

    xgrid = [x - dx/2 for x in xlist] + [xlist[-1]+dx/2]
    ygrid = [y - dy/2 for y in ylist] + [ylist[-1]+dy/2]

    X, Y = np.meshgrid(xgrid, ygrid)
    Z = np.array(probs).reshape(len(ylist),len(xlist))

    return X,Y,Z


def get_period_ratios(Ngrid):
    P12s = np.linspace(0.55, 0.76, Ngrid)
    P23s = np.linspace(0.55, 0.76, Ngrid)
    return P12s, P23s


def get_parameters(P12s, P23s):
    parameters = []
    for P12 in P12s:
        for P23 in P23s:
            parameters.append((P12,P23))

    return parameters


def compute_results_for_parameters(parameters, model, use_petit=False, use_megno=False, create_input_cache=False, ground_truth=False):

    if ground_truth:
        # ground truth only requires cpu, so we can parallelize across cores
        num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", mp.cpu_count()))
        print(f'Parallelizing {len(parameters)} sims across {num_cpus} cores')

        with mp.Pool(processes=num_cpus) as pool:
            results = pool.map(get_ground_truth_wrapper, parameters)

    else:
        simulations = [get_simulation(par) for par in parameters]
        results = [get_model_prediction(sim, model, use_petit=use_petit, use_megno=use_megno, create_input_cache=create_input_cache, ground_truth=ground_truth) for sim in simulations]

    return results


def load_input_cache(Ngrid):
    path = get_results_path(Ngrid, input_cache=True)

    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return None


def get_list_chunk(lst, ix, total):
    '''
    split list into total chunks and return the ix-th chunk
    example: get_list_chunk([0,1,2,3,4,5,6,7,8,9], 0, 3) -> [0,1,2]
             get_list_chunk([0,1,2,3,4,5,6,7,8,9], 1, 3) -> [2,3,4]
             get_list_chunk([0,1,2,3,4,5,6,7,8,9], 2, 3) -> [5,6,7,8,9]
    '''
    assert ix < total
    chunk_size = len(lst) // total
    start = ix * chunk_size
    end = start + chunk_size if ix < total - 1 else len(lst)
    return lst[start:end]


def predict_from_cached_input(model, cached_input):
    if cached_input is None:
        # the simulation messed up somehow, return None just like in get_model_prediction
        return None
    out_dict = model.model(cached_input, noisy_val=False, return_intermediates=True, deterministic=True)

    return {
        'mean': out_dict['mean'][0,0].detach().cpu().numpy(),
        'std': out_dict['std'][0,0].detach().cpu().numpy(),
        'f1': out_dict['summary_stats'][0].detach().cpu().numpy(),
    }


def compute_results(args):
    if not args.ground_truth:
        model = load_model(args.version)
    else:
        model = None

    P12s, P23s = get_period_ratios(args.Ngrid)
    parameters = get_parameters(P12s, P23s)

    input_cache = None
    if not (args.petit or args.megno or args.ground_truth):
        input_cache = load_input_cache(args.Ngrid)
        if input_cache:
            assert len(input_cache) == len(parameters), f'Input cache length {len(input_cache)} does not match parameters length {len(parameters)}'
            input_cache = {params: input for params, input in zip(parameters, input_cache)}

    if args.parallel_ix is not None:
        # if there are already results for this ix, return
        path = get_results_path(args.Ngrid, args.version, args.parallel_ix, args.parallel_total, args.pysr_version, args.pysr_model_selection, args.petit, args.megno, input_cache=args.create_input_cache, ground_truth=args.ground_truth)

        if os.path.exists(path):
            print(f'File {path} already exists')
            print(f'Already computed results for ix={args.parallel_ix}, skipping')
            return

        parameters = get_list_chunk(parameters, args.parallel_ix, args.parallel_total)

    if input_cache:
        print('Computing using cached input')

        for par in parameters:
            assert par in input_cache, f'Input cache missing for {par}'

        results = [predict_from_cached_input(model, input_cache[par]) for par in parameters]

    else:
        results = compute_results_for_parameters(parameters, model, use_petit=args.petit, use_megno=args.megno, create_input_cache=args.create_input_cache, ground_truth=args.ground_truth)

    # save the results
    path = get_results_path(args.Ngrid, args.version, args.parallel_ix, args.parallel_total, args.pysr_version, args.pysr_model_selection, args.petit, args.megno, input_cache=args.create_input_cache, ground_truth=args.ground_truth)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(results, f)

    print('Saved results to', path)
    return results


def get_results_path(Ngrid, version=None, parallel_ix=None, parallel_total=None, pysr_version=None, pysr_model_selection=None, use_petit=False, use_megno=False, input_cache=False, ground_truth=False):
    if use_petit:
        path = f'period_results/petit/petit_ngrid={Ngrid}'
    elif use_megno:
        path = f'period_results/megno/megno_ngrid={Ngrid}'
    elif input_cache:
        path = f'period_results/caches/cache_ngrid={Ngrid}'
    elif ground_truth:
        T = GROUND_TRUTH_MAX_T
        T_str = f'{T:.0e}'.replace('e+0', 'e')
        path = f'period_results/ground_truth/ground_truth_ngrid={Ngrid}_T={T_str}'
    else:
        path = f'period_results/v={version}/v={version}_ngrid={Ngrid}'

        if pysr_model_selection is not None:
            if pysr_model_selection == 'accuracy':
                model = pickle.load(open(f'../sr_results/{pysr_version}.pkl', 'rb'))
                if type(model.equations_) == list:
                    pysr_model_selection = max(model.equations_[0]['complexity'])
                else:
                    pysr_model_selection = max(model.equations_['complexity'])

            path += f'_pysr_f2_v={pysr_version}/{pysr_model_selection}'

    if parallel_ix is not None:
        path += f'/{parallel_ix}-{parallel_total}'

    path += '.pkl'
    return path


def load_results(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def collate_parallel_results(args):
    '''load the parallel results and save as one big list'''
    results = []
    if args.parallel_total is None:
        # try to detect the total
        path = get_results_path(args.Ngrid, args.version, use_petit=args.petit, input_cache=args.create_input_cache, use_megno=args.megno, ground_truth=args.ground_truth)

        files = os.listdir(get_results_path(args.Ngrid, args.version, use_petit=args.petit, input_cache=args.create_input_cache, use_megno=args.megno, ground_truth=args.ground_truth)[:-4])
        # filter to those of form f'{ix}-{total}.pkl'
        files = [file for file in files if file.endswith('.pkl')]
        # get the total. use the largest possible total
        assert len(files) > 0
        total = int(files[0].split('-')[1].split('.')[0])
        print('Detected parallel total as', total)
        print(f'Found {len(files)} files to collate')
        if len(files) != total:
            ixs_present = [int(file.split('-')[0]) for file in files]
            missing = sorted(list(set(range(total)) - set(ixs_present)))
            print('Missing files:', missing)
    else:
        total = args.parallel_total

    missing = False
    for ix in range(total):
        # print(ix)
        path = get_results_path(args.Ngrid, args.version, ix, total, use_petit=args.petit, input_cache=args.create_input_cache, use_megno=args.megno, ground_truth=args.ground_truth)
        # print(f'path={path}')

        try:
            sub_results = load_results(path)

            if args.create_input_cache:
                sub_results = [t.cpu() if t is not None else None for t in sub_results]
        except FileNotFoundError:
            sub_results = None
            print('Missing results for ix', ix)
            missing = True

        results.append(sub_results)

    if not missing:
        print('All subresults files found!')

    length = None
    for sub_results in results:
        if sub_results is not None:
            length = len(sub_results)
            break

    if length is None:
        print('No results found')
        return

    results = [sub_result if sub_result is not None
               else [None for _ in range(length)]
               for sub_result in results]

    # concatenate into one big list of results, maintaining ordering
    results = [result for sub_results in results for result in sub_results]
    path = get_results_path(args.Ngrid, args.version, use_petit=args.petit, use_megno=args.megno, ground_truth=args.ground_truth)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print('Saved results to', path)


def plot_results(args, metric=None):
    if (not (args.petit or args.megno or args.ground_truth or args.equation_bounds)) and metric is None:
        for metric in ['mean', 'std']:
            plot_results(args, metric)
        return

    results = load_results(get_results_path(args.Ngrid, args.version, pysr_version=args.pysr_version, pysr_model_selection=args.pysr_model_selection, use_petit=args.petit, use_megno=args.megno, ground_truth=args.ground_truth))
    P12s, P23s = get_period_ratios(args.Ngrid)

    fig, ax = plt.subplots(figsize=(5,4.5))
    ax.set_aspect('equal', adjustable='box')

    if not args.minimal_plot:
        ticks = [0.55, 0.60, 0.65, 0.70, 0.75]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    # get the results for the specific metric
    if args.petit:
        results = [d['petit'] if d is not None else np.nan for d in results]
    elif args.megno:
        results = [d['megno'] if d is not None else np.nan for d in results]
    elif args.equation_bounds:
        results = [d['bound'] if d is not None else np.nan for d in results]
    elif args.ground_truth:
        results = [d['ground_truth'] if d is not None else np.nan for d in results]
    elif metric == 'mean':
        results = [d['mean'] if d is not None else np.nan for d in results]
    elif metric == 'mean2':
        pysr_results = results
        base_results = load_results(get_results_path(args.Ngrid, args.version))
        results = [(pysr_d['mean'] - d['mean'])**2 if d is not None and pysr_d is not None else np.nan for pysr_d, d in zip(pysr_results, base_results)]
    elif metric == 'std':
        results = [d['std'] if d is not None else np.nan for d in results]
    elif metric == 'std2':
        results = [d['std'] / d['mean'] if d is not None else np.nan for d in results]

    results = np.array(results)
    X,Y,Z = get_centered_grid(P12s, P23s, results)

    if args.petit:
        cmap = plt.cm.inferno.copy().reversed()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, Z, cmap=cmap)
        label = INSTABILITY_TIME_LABEL
    if args.megno:
        Zfilt = Z
        Zfilt[Zfilt <= 2] = 2.01
        cmap = plt.cm.inferno.copy()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, np.log10(Zfilt-2), vmin=-4, vmax=4, cmap=cmap)
        label = MEGNO_LABEL
    if args.ground_truth:
        cmap = plt.cm.inferno.copy().reversed()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, Z, cmap=cmap)
        label = INSTABILITY_TIME_LABEL
    elif metric == 'std':
        cmap = plt.cm.inferno.copy().reversed()
        cmap.set_bad(color='white')
        m = Z[~np.isnan(Z)].max()
        im = ax.pcolormesh(X, Y, Z, vmin=0, vmax=m, cmap=cmap)
        label = "std(" + INSTABILITY_TIME_LABEL + ")"
    elif args.equation_bounds:
        cmap = plt.cm.inferno.copy().reversed()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, Z, vmin=0, vmax=1, cmap=cmap)
        label = 'Equation bounds'

    elif metric == 'mean' or metric == 'mean2':
        cmap = plt.cm.inferno.copy().reversed()
        cmap.set_bad(color='white')
        # zmax = Z[~np.isnan(Z)].max()
        # zmin = Z[~np.isnan(Z)].min()
        zmax = 12
        zmin = 4

        im = ax.pcolormesh(X, Y, Z, vmin=zmin, vmax=zmax, cmap=cmap)
        label = INSTABILITY_TIME_LABEL

    if args.minimal_plot:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label)
        ax.set_xlabel("P1/P2")
        ax.set_ylabel("P2/P3")

    plt.tight_layout()

    path = get_results_path(args.Ngrid, args.version, use_petit=args.petit, use_megno=args.megno, ground_truth=args.ground_truth)
    # get rid of .pkl
    path = path[:-4]
    path += '_plot'

    if metric == 'std':
        path += '_std'
    if metric == 'std2':
        path += '_std2'
    if metric == 'mean2':
        path += '_mean2'

    if args.pysr_model_selection is not None:
        path += f'_pysr_f2_v={args.pysr_version}/{args.pysr_model_selection}'

    path += '.png'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=800)
    print('Saved figure to', path)
    plt.close(fig)


def get_pysr_module(sr_results_path, residual_sr_results_path=None, model_selection=None, residual_model_selection=None):
    # TODO: edward
    if residual_sr_results_path is None:
        # nonresidual sr
        regress_nn = modules.PySRNet(sr_results_path, model_selection).cuda()
        return regress_nn
    else:
        # ... do it for residual.
        pass


def compute_pysr_f2_results(args):
    try:
        results = load_results(get_results_path(args.Ngrid, args.version))
    except FileNotFoundError:
        print('Results not found. Make sure you run --compute with the same --version, but without --pysr, before you compute with pysr f2')
        import sys
        sys.exit(0)

    if args.pysr_model_selection is None:
        reg = pickle.load(open(args.pysr_path, 'rb'))
        model_selections = list(reg.equations_[0]['complexity'])
    else:
        model_selections = [args.pysr_model_selection]

    f1_results = [d['f1'] if d is not None else None for d in results]
    good_ixs = np.array([i for i in range(len(f1_results)) if f1_results[i] is not None])

    batch = np.array([d for d in f1_results if d is not None])
    # if cuda is not available, quit
    if not torch.cuda.is_available():
        print('CUDA not available, do this on the cluster!')
        return
    batch = torch.tensor(batch).float().cuda()

    for model_selection in model_selections:
        regress_nn = modules.PySRNet(args.pysr_path, model_selection).cuda()
        pred = regress_nn(batch).detach().cpu().numpy()
        results = np.full((len(f1_results), pred.shape[1]), np.NaN)
        results[good_ixs] = pred

        if args.equation_bounds:
            assert_equal(results.shape[1], 1)
            # convert back to dictionary of 'mean': mean, 'std': std, for compatibility with the other results
            results2 = []
            for result in results:
                if np.isnan(result).any():
                    results2.append(None)
                else:
                    results2.append({
                        'bound': result[0],
                    })
        else:
            assert_equal(results.shape[1], 2)
            # convert back to dictionary of 'mean': mean, 'std': std, for compatibility with the other results
            results2 = []
            for result in results:
                if np.isnan(result).any():
                    results2.append(None)
                else:
                    results2.append({
                        'mean': result[0],
                        'std': result[1]
                    })


        results = results2

        # save the results
        path = get_results_path(args.Ngrid, args.version, pysr_version=args.pysr_version, pysr_model_selection=model_selection)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(results, f)

        print('Saved results to', path)

    return results


def plot_results_pysr_f2(args):
    # get the model selections by grepping for the files
    path = get_results_path(args.Ngrid, args.version, pysr_version=args.pysr_version, pysr_model_selection='*')
    files = os.listdir(os.path.dirname(path))

    # filter to those of form f'{i}.pkl'
    files = [file for file in files if file.endswith('.pkl')]

    # go from f'{model_selection}.pkl' to model_selection
    model_selections = sorted([int(f.split('.')[0]) for f in files])
    # model_selections = [1, 3, 5, 7, 9, 11, 14, 18, 20, 27, 29]

    # if model selection is provided, filter to those
    if args.pysr_model_selection is not None:
        if args.pysr_model_selection == 'accuracy':
            model_selections = [model_selections[-1]]
        else:
            files = [file for file in files if file.startswith(args.pysr_model_selection)]

    # go from f'{model_selection}.pkl' to model_selection
    model_selections = [int(f.split('.')[0]) for f in files]
    model_selections = sorted(model_selections)

    original_model_selection = args.pysr_model_selection
    for model_selection in model_selections:
        args.pysr_model_selection = model_selection
        args.title = f'Equation complexity = {model_selection}'
        plot_results(args)
    args.pysr_model_selection = original_model_selection


def plot_4way_comparison(args):
    # Create the figure and axes
    # fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 8))
    axs = axs.flatten()

    for ax in axs:
        ax.set_aspect('equal', adjustable='box')

    ticks = [0.55, 0.60, 0.65, 0.70, 0.75]
    for ax in axs:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    nn_results = load_results(get_results_path(args.Ngrid, args.version))
    eq_results = load_results(get_results_path(args.Ngrid, args.version, pysr_version=args.pysr_version, pysr_model_selection='accuracy'))
    petit_results = load_results(get_results_path(args.Ngrid, use_petit=True))
    megno_results = load_results(get_results_path(args.Ngrid, use_megno=True))

    model_results = [nn_results, eq_results, petit_results, megno_results]
    names = ['nn', 'eq', 'petit', 'megno']

    for i in range(4):
        results = model_results[i]
        name = names[i]

        P12s, P23s = get_period_ratios(args.Ngrid)

        # get the results for the specific metric
        if name == 'petit':
            results = [d['petit'] if d is not None else np.nan for d in results]
        elif name == 'megno':
            results = [d['megno'] if d is not None else np.nan for d in results]
        else:
            results = [d['mean'] if d is not None else np.nan for d in results]

        results = np.array(results)
        X,Y,Z = get_centered_grid(P12s, P23s, results)

        if name == 'petit':
            cmap = plt.cm.inferno.copy().reversed()
            cmap.set_bad(color='white')
            im = axs[i].pcolormesh(X, Y, Z, cmap=cmap)
            label = INSTABILITY_TIME_LABEL
        elif name == 'megno':
            Zfilt = Z
            Zfilt[Zfilt <= 2] = 2.01
            cmap = plt.cm.inferno.copy()
            cmap.set_bad(color='white')
            im = axs[i].pcolormesh(X, Y, np.log10(Zfilt-2), vmin=-4, vmax=4, cmap=cmap)
            label = MEGNO_LABEL
        else:
            cmap = plt.cm.inferno.copy().reversed()
            cmap.set_bad(color='white')
            zmax = 12
            zmin = 4
            im = axs[i].pcolormesh(X, Y, Z, vmin=zmin, vmax=zmax, cmap=cmap)
            label = INSTABILITY_TIME_LABEL

        cb = plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
        cb.set_label(label)
        axs[i].set_xlabel("P1/P2")
        axs[i].set_ylabel("P2/P3")
        title = ['Neural network', 'Distilled equations', 'Petit 2020', 'MEGNO'][i]
        axs[i].set_title(title)
        plt.tight_layout()


    # fig.set_constrained_layout(True)

    path = get_results_path(args.Ngrid, args.version)[:-4] + '_comparison2'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path + '.png', dpi=800)
    print('Saved figure to', path + '.png')
    plt.close(fig)


def plot_f1_features(args):
    results = load_results(get_results_path(args.Ngrid, args.version))
    n_features = results[0]['f1'].shape[0] // 2

    P12s, P23s = get_period_ratios(args.Ngrid)

    for feature in range(n_features):
        for plot_std in [True, False]:
            values = [d['f1'][feature + n_features if plot_std else feature] if d is not None else np.nan for d in results]
            X,Y,Z = get_centered_grid(P12s, P23s, values)

            fig, ax = plt.subplots(figsize=(8,6))

            cmap = plt.cm.inferno.copy().reversed()
            cmap.set_bad(color='white')
            im = ax.pcolormesh(X, Y, Z, cmap=cmap)
            ax.set_title(f'Feature {feature} std' if plot_std else f'Feature {feature} mean')
            ax.set_xlabel("P1/P2")
            ax.set_ylabel("P2/P3")

            cb = plt.colorbar(im, ax=ax)
            cb.set_label('Feature std' if plot_std else 'Feature mean')
            ax.set_xlabel("P1/P2")
            ax.set_ylabel("P2/P3")
            plt.tight_layout()

            path = get_results_path(args.Ngrid, args.version)
            # get rid of .pkl
            path = path[:-4]
            path += f'_plot_feature{feature}'
            path += '_std' if plot_std else '_mean'
            path += '.png'

            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=800)
            print('Saved figure to', path)
            plt.close(fig)


def plot_4way_pysr_comparison(args):
    # get the model selections by grepping for the files
    path = get_results_path(args.Ngrid, args.version, pysr_version=args.pysr_version, pysr_model_selection='*')
    files = os.listdir(os.path.dirname(path))

    # filter to those of form f'{i}.pkl'
    files = [file for file in files if file.endswith('.pkl')]

    # go from f'{model_selection}.pkl' to model_selection
    model_selections = sorted([int(f.split('.')[0]) for f in files])

    # Define the complexities you want to plot
    desired_complexities = [3, 5, 14, 29]

    # Filter the model_selections to only include the desired complexities
    model_selections = [ms for ms in model_selections if ms in desired_complexities]

    # Create the figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(9, 8))
    axs = axs.flatten()

    for ax in axs:
        ax.set_aspect('equal', adjustable='box')

    ticks = [0.55, 0.60, 0.65, 0.70, 0.75]
    for ax in axs:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    for i, model_selection in enumerate(model_selections):
        args.pysr_model_selection = model_selection
        results = load_results(get_results_path(args.Ngrid, args.version, pysr_version=args.pysr_version, pysr_model_selection=str(model_selection)))

        P12s, P23s = get_period_ratios(args.Ngrid)

        # get the results for the mean
        results = [d['mean'] if d is not None else np.nan for d in results]

        X,Y,Z = get_centered_grid(P12s, P23s, results)

        cmap = plt.cm.inferno.copy().reversed()
        cmap.set_bad(color='white')
        zmax = 12
        zmin = 4
        im = axs[i].pcolormesh(X, Y, Z, vmin=zmin, vmax=zmax, cmap=cmap)
        axs[i].set_title(f'Complexity {model_selection}')
        axs[i].set_xlabel("P1/P2")
        axs[i].set_ylabel("P2/P3")

    # Create a single colorbar
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
    cb.set_label(INSTABILITY_TIME_LABEL)
    fig.set_constrained_layout(True)
    # fig.subplots_adjust(wspace=0.05, hspace=0.05)

    path = get_results_path(args.Ngrid, args.version, pysr_version=args.pysr_version, pysr_model_selection='comparison')[:-4]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path + '.png', dpi=800)
    print('Saved figure to', path + '.png')
    plt.close(fig)


def calculate_mse(Ngrid, version, pysr_version):
    # since we don't have a ground truth, use the BNN predictions as ground truth
    # compare petit and pysr f2
    bnn_results = load_results(get_results_path(Ngrid, version))
    petit_results = load_results(get_results_path(Ngrid, use_petit=True))
    pysr_f2_results = load_results(get_results_path(Ngrid, version, pysr_version=pysr_version, pysr_model_selection='accuracy'))

    bnn_results = [d['mean'] if d is not None else np.nan for d in bnn_results]
    petit_results = [d['petit'] if d is not None else np.nan for d in petit_results]
    pysr_f2_results = [d['mean'] if d is not None else np.nan for d in pysr_f2_results]

    bnn_results, petit_results, pysr_f2_results = np.array(bnn_results), np.array(petit_results), np.array(pysr_f2_results)

    assert_equal(bnn_results.shape, petit_results.shape)
    assert_equal(bnn_results.shape, pysr_f2_results.shape)
    assert_equal(len(bnn_results.shape), 1)

    # ignore entries where bnn_results is nan
    nan_ixs = np.isnan(bnn_results)
    bnn_results = bnn_results[~nan_ixs]
    petit_results = petit_results[~nan_ixs]
    pysr_f2_results = pysr_f2_results[~nan_ixs]

    # ignore entries where petit_results is nan
    nan_ixs = np.isnan(petit_results)
    bnn_results = bnn_results[~nan_ixs]
    petit_results = petit_results[~nan_ixs]
    pysr_f2_results = pysr_f2_results[~nan_ixs]

    # ignore entries where pysr_f2 is nan
    nan_ixs = np.isnan(pysr_f2_results)
    bnn_results = bnn_results[~nan_ixs]
    petit_results = petit_results[~nan_ixs]
    pysr_f2_results = pysr_f2_results[~nan_ixs]

    # calculate mse
    mse_petit = np.mean((bnn_results - petit_results)**2)
    mse_pysr_f2 = np.mean((bnn_results - pysr_f2_results)**2)
    print('MSE Petit:', mse_petit)
    print('MSE PySR F2:', mse_pysr_f2)


def get_args():
    print(utils2.get_script_execution_command())
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ngrid', '-n', type=int, default=1600)
    parser.add_argument('--version', '-v', type=int, default=24880)

    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--compute', action='store_true')
    parser.add_argument('--collate', action='store_true')
    parser.add_argument('--petit', action='store_true')
    parser.add_argument('--megno', action='store_true')
    parser.add_argument('--ground_truth', action='store_true')
    parser.add_argument('--create_input_cache', action='store_true')

    parser.add_argument('--pysr_version', type=str, default=None) # sr_results/11003.pkl
    parser.add_argument('--pysr_dir', type=str, default='../sr_results/')  # folder containing pysr results pkl

    parser.add_argument('--pysr_model_selection', type=str, default=None, help='"best", "accuracy", "score", or an integer of the pysr equation complexity. If not provided, will do all complexities. If plotting, has to be an integer complexity')

    # parallel processing args
    # ix should be in [0, total)
    parser.add_argument('--parallel_ix', '-i', type=int, default=None)
    parser.add_argument('--parallel_total', '-t', type=int, default=None)

    parser.add_argument('--equation_bounds', action='store_true')
    parser.add_argument('--job_array', action='store_true')
    parser.add_argument('--max_t', type=float, default=1e9, help='Maximum integration time for ground truth')
    parser.add_argument('--plot_special', type=str, default=None, choices=['4way', '4way_pysr'])
    parser.add_argument('--minimal_plot', action='store_true')

    args = parser.parse_args()

    if args.pysr_version is not None:
        args.pysr_path = os.path.join(args.pysr_dir, f'{args.pysr_version}.pkl')
    else:
        args.pysr_path = None

    if args.create_input_cache and not args.collate:
        args.compute = True

    if args.job_array:
        try:
            array_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
            array_total = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
            # get the
        except KeyError:
            print("Error: --job_array specified but SLURM array variables not found")
            sys.exit(1)

        # sometimes, we want more jobs than job array slots possible
        # in this case, we overload parallel_ix to be the index in the different job array submissions.
        if args.parallel_ix is not None:
            args.parallel_ix = args.parallel_ix * array_total + array_id
            args.parallel_total = args.parallel_total * array_total
            if args.parallel_ix >= args.Ngrid ** 2:
                print('Exceeded total number of jobs, this one is not needed.')
                sys.exit(0)
        else:
            args.parallel_ix = array_id
            args.parallel_total = array_total

    if args.ground_truth:
        global GROUND_TRUTH_MAX_T
        GROUND_TRUTH_MAX_T = args.max_t

    return args


if __name__ == '__main__':
    args = get_args()

    start = time.time()

    if args.create_input_cache:
        # check if cache already exists for this size!
        path = get_results_path(args.Ngrid, input_cache=True)
        if os.path.exists(path):
            print('Input cache already exists for this size!')
            import sys; sys.exit(0)

    if args.compute:
        if args.pysr_path:
            compute_pysr_f2_results(args)
        else:
            results = compute_results(args)

    if args.collate:
        collate_parallel_results(args)

    if args.plot:
        if args.pysr_path:
            plot_results_pysr_f2(args)
            # pysr_error_analysis(args)
        else:
            plot_results(args)
            # plot_summary_stats(args)

    if args.plot_special:
        if args.plot_special == '4way':
            plot_4way_comparison(args)
        if args.plot_special == '4way_pysr':
            plot_4way_pysr_comparison(args)

    end = time.time()
    formatted_time = time.strftime('%H:%M:%S', time.gmtime(end - start))
    print(f'Done (time taken: {formatted_time})')
