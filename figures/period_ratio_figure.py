import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
sys.path.append('../')

import rebound
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['mathtext.fontset']='dejavusans'

import torch
import modules
try:
    import spock
except ImportError:
    import figures.spock as spock
import pickle
import math
from utils2 import assert_equal, load_pickle, get_script_execution_command, load_json, truncate_cmap, save_pickle
import einops
import multiprocessing as mp
import argparse
import time
import copy
import spock_reg_model
from petit20_survival_time import Tsurv
import cmasher as cmr
import warnings
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score, roc_curve
from resonant_period_ratio_overlay import (
    add_period_ratio_inner32_slice_overlay,
    add_period_ratio_resonant_overlay,
)

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("default", category=UserWarning)

INSTABILITY_TIME_LABEL = r"$\log_{10}(T_{\rm inst})$"
MEGNO_LABEL = r"$\log_{10}(\rm MEGNO-2)$"

GROUND_TRUTH_MAX_T = 1e9 # Assigned in get_args function

COLOR_MAP = plt.cm.plasma
DIVERGING_CMAP = cmr.viola.reversed()
RMSE_CMAP = truncate_cmap(cmr.sunburst.reversed(), 0.0, 0.65)


USE_SUBFOLDERS = False

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


def load_model(args):
    if args.pure_sr:
        pure_sr_net = modules.PureSRNet.from_path(args.pysr_path, args.pysr_model_selection)
        return spock.NonSwagFeatureRegressor(pure_sr_net)
    else:
        model = spock_reg_model.load(args.version, args.seed)
        return spock.NonSwagFeatureRegressor(model)


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

def get_petit_prediction(par):
    p12, p23 = par
    t = Tsurv(
        p12, p23,
        [1e-4, 1e-4, 1e-4],
    )
    t = np.clip(t, 0, 1e9)
    return np.log10(np.nan_to_num(t, posinf=1e9, neginf=1e9, nan=1e9))


def get_model_prediction(sim, par, model, use_petit=False, use_megno=False, create_input_cache=False, ground_truth=False):
    '''
    cache: maps simulation id to X.
    '''

    if ground_truth:
        return {'ground_truth': get_ground_truth(sim)}
    elif use_petit:
        return {'mean': get_petit_prediction(par)}

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
            out_dict = model.predict(sim)
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
    # Z = np.array(probs).reshape(len(ylist),len(xlist))
    # plotting expects Y then X
    Z = einops.rearrange(np.array(probs), '(x y) -> y x', x=len(xlist), y=len(ylist))

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
        results = [get_model_prediction(sim, par, model, use_petit=use_petit, use_megno=use_megno, create_input_cache=create_input_cache, ground_truth=ground_truth) for sim, par in zip(simulations, parameters)]

    return results


def load_input_cache(Ngrid):
    path = get_results_path(Ngrid, input_cache=True)

    if not os.path.exists(path):
        path = 'figures/' + path

        if not os.path.exists(path):
            return None

    with open(path, 'rb') as f:
        return pickle.load(f)


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


def predict_from_cached_input(model, cached_input, pure_sr=False):
    if cached_input is None:
        # the simulation messed up somehow, return None just like in get_model_prediction
        return None

    if pure_sr:
        out = model.model(cached_input)
        return {
            'mean': out[0, 0].detach().cpu().numpy(),
        }

    out_dict = model.model(cached_input, noisy_val=False, return_intermediates=True, deterministic=True)

    return {
        'mean': out_dict['mean'][0,0].detach().cpu().numpy(),
        'std': out_dict['std'][0,0].detach().cpu().numpy(),
        'f1': out_dict['summary_stats'][0].detach().cpu().numpy(),
    }


def compute_results(args):
    if not args.ground_truth:
        # need model for petit currently... because the logic for setting up the sim is intertwined a lot
        model = load_model(args)
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

        results = [predict_from_cached_input(model, input_cache[par], pure_sr=args.pure_sr) for par in parameters]

    else:
        results = compute_results_for_parameters(parameters, model, use_petit=args.petit, use_megno=args.megno, create_input_cache=args.create_input_cache, ground_truth=args.ground_truth)

    # save the results
    path = get_results_path(args.Ngrid, args.version, args.parallel_ix, args.parallel_total, args.pysr_version, args.pysr_model_selection, args.petit, args.megno, input_cache=args.create_input_cache, ground_truth=args.ground_truth, pure_sr=args.pure_sr)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(results, f)

    print('Saved results to', path)
    return results


def _result_exists(args):
    path = get_results_path(
        args.Ngrid,
        args.version,
        args.parallel_ix,
        args.parallel_total,
        args.pysr_version,
        args.pysr_model_selection,
        args.petit,
        args.megno,
        input_cache=args.create_input_cache,
        ground_truth=args.ground_truth,
        pure_sr=args.pure_sr,
    )
    return os.path.exists(path), path


def _run_compute_job(args, **updates):
    job_args = copy.copy(args)
    job_args.compute = True
    job_args.plot = False
    job_args.collate = False
    job_args.special = None
    job_args.petit = False
    job_args.megno = False
    job_args.ground_truth = False
    job_args.pure_sr = False
    job_args.create_input_cache = False
    job_args.pysr_version = None
    job_args.pysr_model_selection = None
    job_args.pysr_path = None
    job_args.version = None

    for key, value in updates.items():
        setattr(job_args, key, value)

    if job_args.pysr_version is not None:
        job_args.pysr_path = os.path.join(job_args.pysr_dir, f'{job_args.pysr_version}.pkl')

    exists, path = _result_exists(job_args)
    if exists:
        print('Already computed, skipping', path)
        return

    print('Computing', path)
    if job_args.pysr_path and not job_args.pure_sr:
        compute_pysr_f2_results(job_args)
    else:
        compute_results(job_args)


def compute_official_results(args):
    v = load_json(args.version_json)

    cache_path = get_results_path(args.Ngrid, input_cache=True)
    if not os.path.exists(cache_path):
        _run_compute_job(args, version=v['nn_version'], create_input_cache=True)
    else:
        print('Input cache already exists, skipping', cache_path)

    _run_compute_job(args, version=v['nn_version'])
    _run_compute_job(
        args,
        version=v['nn_version'],
        pysr_version=str(v['pysr_version']),
        pysr_model_selection=str(v['pysr_model_selection']),
    )

    for complexity in [3, 7, 14, 26]:
        _run_compute_job(
            args,
            version=v['nn_version'],
            pysr_version=str(v['pysr_version']),
            pysr_model_selection=str(complexity),
        )

    _run_compute_job(args, petit=True)
    _run_compute_job(args, ground_truth=True)
    _run_compute_job(
        args,
        pure_sr=True,
        pysr_version=str(v['pure_sr_version']),
        pysr_model_selection=str(v['pure_sr_model_selection']),
    )
    _run_compute_job(args, version=v['pure_sr2_nn_version'])
    _run_compute_job(
        args,
        version=v['pure_sr2_nn_version'],
        pysr_version=str(v['pure_sr2_version']),
        pysr_model_selection=str(v['pure_sr2_model_selection']),
    )


def should_compute_official_results(args):
    return (
        args.compute
        and args.version is None
        and args.pysr_version is None
        and not args.petit
        and not args.megno
        and not args.ground_truth
        and not args.pure_sr
        and not args.create_input_cache
    )


def get_results_path(Ngrid, version=None, parallel_ix=None, parallel_total=None, pysr_version=None, pysr_model_selection=None, use_petit=False, use_megno=False, input_cache=False, ground_truth=False, special=False, minimal_plot=False, pure_sr=False):
    if use_petit:
        if USE_SUBFOLDERS:
            path = f'period_results/petit/petit_ngrid={Ngrid}'
        else:
            path = f'period_results/petit_ngrid={Ngrid}'
    elif use_megno:
        if USE_SUBFOLDERS:
            path = f'period_results/megno/megno_ngrid={Ngrid}'
        else:
            path = f'period_results/megno_ngrid={Ngrid}'
    elif input_cache:
        if USE_SUBFOLDERS:
            path = f'period_results/caches/cache_ngrid={Ngrid}'
        else:
            path = f'period_results/cache_ngrid={Ngrid}'
    elif ground_truth:
        T = GROUND_TRUTH_MAX_T
        T_str = f'{T:.0e}'.replace('e+0', 'e')
        if USE_SUBFOLDERS:
            path = f'period_results/ground_truth/ground_truth_ngrid={Ngrid}_T={T_str}'
        else:
            path = f'period_results/ground_truth_ngrid={Ngrid}_T={T_str}'
    elif pure_sr:
        if USE_SUBFOLDERS:
            path = f'period_results/pure_sr/v={pysr_version}_ngrid={Ngrid}_{pysr_model_selection}'
        else:
            path = f'period_results/v={pysr_version}_ngrid={Ngrid}_{pysr_model_selection}'
    else:
        if USE_SUBFOLDERS:
            path = f'period_results/v={version}/v={version}_ngrid={Ngrid}'
        else:
            path = f'period_results/v={version}_ngrid={Ngrid}'

        if pysr_version is not None:
            if pysr_model_selection == 'accuracy':
                model = pickle.load(open(f'../sr_results/{pysr_version}.pkl', 'rb'))
                if type(model.equations_) == list:
                    pysr_model_selection = max(model.equations_[0]['complexity'])
                else:
                    pysr_model_selection = max(model.equations_['complexity'])

            path += f'_pysr_f2_v={pysr_version}/{pysr_model_selection}'

    if special:
        path += '_' + special

    if parallel_ix is not None:
        path += f'/{parallel_ix}-{parallel_total}'

    if minimal_plot:
        path += '_minimal'

    path += '.pkl'
    return path


def collate_parallel_results(args):
    '''load the parallel results and save as one big list'''
    results = []
    if args.parallel_total is None:
        # try to detect the total
        path = get_results_path(args.Ngrid, args.version, use_petit=args.petit, input_cache=args.create_input_cache, use_megno=args.megno, ground_truth=args.ground_truth, pure_sr=args.pure_sr)

        files = os.listdir(get_results_path(args.Ngrid, args.version, use_petit=args.petit, input_cache=args.create_input_cache, use_megno=args.megno, ground_truth=args.ground_truth, pure_sr=args.pure_sr)[:-4])
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
            # print('Missing files:', missing)
    else:
        total = args.parallel_total

    missing = False
    for ix in range(total):
        # print(ix)
        path = get_results_path(args.Ngrid, args.version, ix, total, use_petit=args.petit, input_cache=args.create_input_cache, use_megno=args.megno, ground_truth=args.ground_truth, pure_sr=args.pure_sr)
        # print(f'path={path}')

        try:
            sub_results = load_pickle(path)

            if args.create_input_cache:
                sub_results = [t.cpu() if t is not None else None for t in sub_results]

            print('Found results for ix', ix)
        except FileNotFoundError:
            sub_results = None
            # print('Missing results for ix', ix)
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
    path = get_results_path(args.Ngrid, args.version, use_petit=args.petit, use_megno=args.megno, ground_truth=args.ground_truth, pure_sr=args.pure_sr)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print('Saved results to', path)


def plot_main_figure(args):
    v = load_json(args.version_json)
    scale = 0.7
    fig = plt.figure(figsize=(17.5 * scale, 15.5 * scale), constrained_layout=False)

    # Outer grid: two blocks vertically.
    #   Top block holds the original 2×4 layout (with narrow colour-bar col).
    #   Bottom block is a new 1×4 equally spaced row.
    gs_outer = GridSpec(
        2, 1,
        height_ratios=[2.4, 1],  # tweak as you like
        hspace=0.20,
        figure=fig
    )

    # Original layout inside the top block
    gs_top = gs_outer[0].subgridspec(
        2, 4,
        width_ratios=[0.92, 0.95, 0.15, 1.25],
        wspace=0,
        hspace=0.19
    )

    # New bottom row: four equal columns spanning full width
    gs_bottom = gs_outer[1].subgridspec(
        1, 4,
        width_ratios=[1, 1, 1, 1],
        wspace=0.15,
    )

    # ---------------- existing six axes ----------------
    axes_map = {
        0: fig.add_subplot(gs_top[0, 0]),  # GT
        1: fig.add_subplot(gs_top[0, 1]),  # Eqns
        2: fig.add_subplot(gs_top[0, 3]),  # distilled-RMSE
        3: fig.add_subplot(gs_top[1, 0]),  # NN
        4: fig.add_subplot(gs_top[1, 1]),  # Petit
        5: fig.add_subplot(gs_top[1, 3])   # ΔRMSE
    }
    axs = [axes_map[i] for i in range(6)]

    P12s, P23s = get_period_ratios(args.Ngrid)

    # ---------------- new bottom axes ------------------
    bottom_axes = [fig.add_subplot(gs_bottom[0, i]) for i in range(4)]

    # (A) original 4‑way panels
    nn_res = load_pickle(get_results_path(args.Ngrid, v['nn_version']))
    eq_res = load_pickle(get_results_path(
        args.Ngrid, v['nn_version'],
        pysr_version=v['pysr_version'],
        pysr_model_selection=v['pysr_model_selection']))
    petit_res = load_pickle(get_results_path(args.Ngrid, use_petit=True))
    gt_res = load_pickle(get_results_path(args.Ngrid, ground_truth=True))

    models = [gt_res, eq_res, nn_res, petit_res]
    names = ['ground_truth', 'eq', 'nn', 'petit']
    titles = ['Ground truth', 'Distilled equations', 'Neural network', 'Petit+ 2020']
    major_ticks = [0.55, 0.65, 0.75]
    minor_ticks = [0.55, 0.60, 0.65, 0.70, 0.75]
    panel_indices = [0, 1, 3, 4]

    im4way = None
    for idx_panel, (model_data, name, title) in zip(panel_indices, zip(models, names, titles)):
        ax = axs[idx_panel]
        show_x = idx_panel in (3, 4)
        show_y = idx_panel in (0, 3)

        ax.set_aspect('equal')
        ax.set_xticks(major_ticks, major=True)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks, major=True)
        ax.set_yticks(minor_ticks, minor=True)
        if not show_x:
            ax.set_xticklabels([])
        if not show_y:
            ax.set_yticklabels([])

        data = model_data
        if name == 'petit':
            data = [d['mean'] if d else np.nan for d in data]
        elif name == 'ground_truth':
            data = [np.log10(d['ground_truth']) if d else np.nan for d in data]
            data = [val if val >= 4 else np.nan for val in data]
        else:
            data = [d['mean'] if d else np.nan for d in data]

        X, Y, Z = get_centered_grid(P12s, P23s, np.asarray(data))

        cmap = COLOR_MAP.copy().reversed()
        cmap.set_bad(color='white')
        im4way = ax.pcolormesh(X, Y, Z, vmin=4, vmax=9, cmap=cmap, rasterized=True)

        if show_x:
            ax.set_xlabel(r"$P_1/P_2$")
        if show_y:
            ax.set_ylabel(r"$P_2/P_3$")
        ax.set_title(title)

    # Shared colour bar (top grid only)
    cax = fig.add_subplot(gs_top[:, 2])
    cb = fig.colorbar(im4way, cax=cax)
    cb.set_label(INSTABILITY_TIME_LABEL)
    bb = cax.get_position(fig)
    half = 0.5 * bb.height
    new_bottom = bb.y0 + 0.25 * bb.height
    width = 0.01
    cax.set_position([bb.x0, new_bottom, width, half])

    # RMSE panels
    _draw_diff_panel(axs[2], v, special='gt_diff', hide_xlabel=True, exclude_stable=args.exclude_stable)
    _draw_diff_panel(axs[5], v, special='rmse_diff', hide_xlabel=False, exclude_stable=args.exclude_stable)

    # -------- bottom row ----
    complexities = [3, 7, 14, 26]
    data = [
        load_pickle(get_results_path(
            args.Ngrid, v['nn_version'],
            pysr_version=v['pysr_version'],
            pysr_model_selection=comp))
        for comp in complexities
    ]
    data = [np.array([d['mean'] if d else np.nan for d in datum]) for datum in data]
    for i, ax in enumerate(bottom_axes):
        X, Y, Z = get_centered_grid(P12s, P23s, data[i])
        ax.set_aspect('equal')
        ax.set_xticks(major_ticks, major=True)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks, major=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        cmap = COLOR_MAP.copy().reversed()
        cmap.set_bad(color='white')
        ax.pcolormesh(X, Y, Z, vmin=4, vmax=9, cmap=cmap, rasterized=True)
        ax.set_title(f"Complexity {complexities[i]}")

    out_path = "period_results/period_ratio_main" + (".png" if args.png else ".pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=800, bbox_inches='tight')
    print("Saved figure to", out_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Helper: draw either the distilled-RMSE (“gt_diff”) or ΔRMSE (“rmse_diff”)
# panel inside a supplied Axes object.
# ----------------------------------------------------------------------
def _draw_diff_panel(ax, v, special, hide_xlabel=False, exclude_stable=False):
    """
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw into.
    Ngrid : int
        Grid resolution used by `get_period_ratios`.
    v : dict
        Parsed JSON holding version metadata (keys: 'nn_version',
        'pysr_version', 'pysr_model_selection', …).
    special : {'gt_diff', 'rmse_diff'}
        • 'gt_diff'    →  RMSE of distilled equations vs. ground truth
        • 'rmse_diff'  →  RMSE_NN − RMSE_Eqns
    hide_xlabel : bool, optional
        If True, x-tick labels are suppressed to align with upper row.
    """

    # ------------- shared prep ------------------------------------------------
    P12s, P23s = get_period_ratios(300)
    ticks = [0.55, 0.60, 0.65, 0.70, 0.75]
    major_ticks = [0.55, 0.65, 0.75]
    minor_ticks = [0.55, 0.60, 0.65, 0.70, 0.75]
    ax.set_aspect('equal')
    ax.set_xticks(major_ticks, major=True)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks, major=True)
    ax.set_yticks(minor_ticks, minor=True)

    nn_version = v['nn_version']
    pysr_v     = v['pysr_version']
    model_sel  = v['pysr_model_selection']

    # Base “equations” map (mean instability time)
    eq_data = load_pickle(
        get_results_path(
            300,
            nn_version,
            pysr_version=pysr_v,
            pysr_model_selection=model_sel
        )
    )
    eq_vals = np.array([d['mean'] if d is not None else np.nan
                        for d in eq_data])

    # Ground-truth map
    gt_data = load_pickle(
        get_results_path(300, nn_version, ground_truth=True)
    )
    gt_vals = np.array([d['ground_truth'] if d is not None else np.nan
                        for d in gt_data])
    gt_vals = np.log10(gt_vals)

    # ------------- panel-specific logic ---------------------------------------
    if special == 'gt_diff':
        # RMSE of distilled equations relative to GT
        eq_clip = np.clip(eq_vals, 4, 9)
        Z = np.sqrt((eq_clip - gt_vals) ** 2)

        cmap = RMSE_CMAP.copy()
        cmap.set_bad(color='white')
        norm = plt.Normalize(vmin=0, vmax=5)
        title = "Distilled equations RMSE"
        label = "RMSE (dex)"

    elif special == 'rmse_diff':
        # Difference of RMSEs: NN – Eqns
        # NN map
        nn_data = load_pickle(get_results_path(300, nn_version))
        nn_vals = np.array([d['mean'] if d is not None else np.nan
                            for d in nn_data])

        eq_clip = np.clip(eq_vals, 4, 9)
        nn_clip = np.clip(nn_vals, 4, 9)

        rmse_eq = np.sqrt((eq_clip - gt_vals) ** 2)
        rmse_nn = np.sqrt((nn_clip - gt_vals) ** 2)
        Z = rmse_nn - rmse_eq

        cmap = DIVERGING_CMAP.copy()
        cmap.set_bad(color='white')
        norm = plt.Normalize(vmin=-5, vmax=5)
        title = "Difference in RMSE (NN – Equations)"
        label = r"${\rm RMSE}_{\rm NN}\;-\;{\rm RMSE}_{\rm Eqns} \ ({\rm dex})$"

    else:
        raise ValueError("special must be 'gt_diff' or 'rmse_diff'")

    if exclude_stable:
        stable_mask = gt_vals >= 9
        Z[stable_mask] = np.nan

    # ------------- draw -------------------------------------------------------
    X, Y, ZZ = get_centered_grid(P12s, P23s, Z)
    im = ax.pcolormesh(X, Y, ZZ, cmap=cmap, norm=norm, rasterized=True)

    # axis labels
    if hide_xlabel:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r"$P_1/P_2$")
    if ax.is_first_col():
        ax.set_ylabel(r"$P_2/P_3$")
    else:
        ax.set_ylabel("")

    # colour-bar
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label)

    ax.set_title(title)
    return im


def plot_results(args, metric=None):
    if (not (args.petit or args.megno or args.ground_truth or args.equation_bounds)) and metric is None:
        for metric in ['mean', 'std']:
            plot_results(args, metric)
        return

    results = load_pickle(get_results_path(args.Ngrid, args.version, pysr_version=args.pysr_version, pysr_model_selection=args.pysr_model_selection, use_petit=args.petit, use_megno=args.megno, ground_truth=args.ground_truth, pure_sr=args.pure_sr))
    P12s, P23s = get_period_ratios(args.Ngrid)

    # scale=0.75
    # scale=0.7
    scale = 1
    # if args.minimal_plot:
        # scale = 0.775 * scale
    fig, ax = plt.subplots(figsize=(5*scale,4.5*scale))
    ax.set_aspect('equal', adjustable='box')

    major_ticks = [0.55, 0.65, 0.75]
    # major_ticks = [0.55, 0.60, 0.65, 0.70, 0.75]
    minor_ticks = [0.60, 0.70]
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # get the results for the specific metric
    if args.petit:
        results = [d['mean'] if d is not None else np.nan for d in results]
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
        base_results = load_pickle(get_results_path(args.Ngrid, args.version))
        results = [(pysr_d['mean'] - d['mean'])**2 if d is not None and pysr_d is not None else np.nan for pysr_d, d in zip(pysr_results, base_results)]
    elif metric == 'std':
        results = [d['std'] if d is not None else np.nan for d in results]
    elif metric == 'std2':
        results = [d['std'] / d['mean'] if d is not None else np.nan for d in results]

    results = np.array(results)

    if args.special == 'rmse_diff':
        ground_truth_results = load_pickle(get_results_path(args.Ngrid, args.version, ground_truth=True))
        ground_truth_results = [d['ground_truth'] if d is not None else np.nan for d in ground_truth_results]
        ground_truth_results = np.array(ground_truth_results)
        ground_truth_results = np.log10(ground_truth_results)

        results[results < 4] = 4
        results[results > 9] = 9
        eq_rmse = np.sqrt((results - ground_truth_results)**2)

        nn_results = load_pickle(get_results_path(args.Ngrid, args.version))
        nn_results = np.array([d['mean'] if d is not None else np.nan for d in nn_results])
        nn_results[nn_results < 4] = 4
        nn_results[nn_results > 9] = 9

        nn_rmse = np.sqrt((nn_results - ground_truth_results)**2)

        results = nn_rmse - eq_rmse

    elif args.special == 'gt_diff':
        ground_truth_results = load_pickle(get_results_path(args.Ngrid, args.version, ground_truth=True))
        ground_truth_results = [d['ground_truth'] if d is not None else np.nan for d in ground_truth_results]
        ground_truth_results = np.array(ground_truth_results)
        ground_truth_results = np.log10(ground_truth_results)

        results[results < 4] = 4
        results[results > 9] = 9
        gt_rmse = np.sqrt((results - ground_truth_results)**2)

        results = gt_rmse

    X,Y,Z = get_centered_grid(P12s, P23s, results)

    if args.special == 'rmse_diff':
        label = r"${\rm RMSE}_{\rm NN}\;-\;{\rm RMSE}_{\rm Eqns}$"
        cmap = DIVERGING_CMAP.copy()
        cmap.set_bad(color='white')
        norm = plt.Normalize(vmin=-5, vmax=5)
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, rasterized=True)
        ax.set_title("Difference in RMSE (NN – Equations)")
    elif args.special == 'gt_diff':
        label = "RMSE"
        cmap = RMSE_CMAP.copy()
        cmap.set_bad(color='white')
        norm = plt.Normalize(vmin=0, vmax=5)
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, rasterized=True)
        ax.set_title("Distilled equations RMSE")
    elif args.petit:
        cmap = COLOR_MAP.copy().reversed()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, rasterized=True)
        label = INSTABILITY_TIME_LABEL
    elif args.megno:
        Zfilt = Z
        Zfilt[Zfilt <= 2] = 2.01
        cmap = COLOR_MAP.copy()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, np.log10(Zfilt-2), vmin=-4, vmax=4, cmap=cmap, rasterized=True)
        label = MEGNO_LABEL
    elif args.ground_truth:
        Z = np.log10(Z)
        Z[Z <= 4] = 4
        cmap = COLOR_MAP.copy().reversed()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, rasterized=True)
        label = INSTABILITY_TIME_LABEL
    elif metric == 'std':
        cmap = COLOR_MAP.copy().reversed()
        cmap.set_bad(color='white')
        m = Z[~np.isnan(Z)].max()
        m = 3
        im = ax.pcolormesh(X, Y, Z, vmin=0, vmax=m, cmap=cmap, rasterized=True)
        label = "std(" + INSTABILITY_TIME_LABEL + ")"
    elif args.equation_bounds:
        cmap = COLOR_MAP.copy().reversed()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, Z, vmin=0, vmax=1, cmap=cmap, rasterized=True)
        label = 'Equation bounds'
    elif metric == 'mean' or metric == 'mean2':
        cmap = COLOR_MAP.copy().reversed()
        cmap.set_bad(color='white')

        zmax = 9
        zmin = 4

        im = ax.pcolormesh(X, Y, Z, vmin=zmin, vmax=zmax, cmap=cmap, rasterized=True)
        label = INSTABILITY_TIME_LABEL

    if args.highlight_resonant_points:
        if args.resonant_overlay_mode == 'old_fixed_outer':
            add_period_ratio_resonant_overlay(
                ax,
                args.Ngrid,
                p2_min=args.resonant_p2_min,
                p2_max=args.resonant_p2_max,
                inner_period=args.resonant_inner_period,
                outer_period=args.resonant_outer_period,
            )
        elif args.resonant_overlay_mode == 'inner32_phase_slice':
            add_period_ratio_inner32_slice_overlay(
                ax,
                args.Ngrid,
                y_min=args.resonant_y_min,
                y_max=args.resonant_y_max,
            )
        else:
            raise ValueError(f"Unknown resonant_overlay_mode: {args.resonant_overlay_mode}")

    if args.minimal_plot:
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"Complexity {args.pysr_model_selection}")
    else:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label)
        ax.set_xlabel(r"$P_1/P_2$")
        ax.set_ylabel(r"$P_2/P_3$")

        # if args.special == 'rmse_diff':
            # cb.ax.set_yticks([-8, -4, 0, 4, 8])
            # cb.ax.set_yticks(np.arange(-8, 9, 2), minor=True)


    plt.tight_layout()

    path = get_results_path(args.Ngrid, args.version, use_petit=args.petit, use_megno=args.megno, ground_truth=args.ground_truth, special=args.special, minimal_plot=args.minimal_plot, pure_sr=args.pure_sr, pysr_version=args.pysr_version, pysr_model_selection=args.pysr_model_selection)
    # get rid of .pkl
    path = path[:-4]

    if metric == 'std':
        path += '_std'
    if metric == 'std2':
        path += '_std2'
    if metric == 'mean2':
        path += '_mean2'
    if args.highlight_resonant_points:
        path += f'_{args.resonant_overlay_mode}_overlay'

    # if args.pysr_version is not None:
    #     path += f'_pysr_f2_v={args.pysr_version}/{args.pysr_model_selection}'

    path += '.png' if args.png else '.pdf'

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
        results = load_pickle(get_results_path(args.Ngrid, args.version))
    except FileNotFoundError:
        print('Results not found. Make sure you run --compute with the same --version, but without --pysr, before you compute with pysr f2')
        import sys
        sys.exit(0)

    if args.pysr_model_selection is None:
        reg = pickle.load(open(args.pysr_path, 'rb'))
        eqs = reg.equations_
        if type(eqs) == list:
            eqs = eqs[0]

        model_selections = list(eqs['complexity'])
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
            # assert_equal(results.shape[1], 2)
            # convert back to dictionary of 'mean': mean, 'std': std, for compatibility with the other results
            results2 = []
            for result in results:
                if np.isnan(result).any():
                    results2.append(None)
                else:
                    if results.shape[1] == 2:
                        results2.append({
                            'mean': result[0],
                            'std': result[1]
                        })
                    else:
                        results2.append({
                            'mean': result[0],
                            'std': 0
                        })

        results = results2

        # save the results
        path = get_results_path(args.Ngrid, args.version, pysr_version=args.pysr_version, pysr_model_selection=model_selection)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(results, f)

        print('Saved results to', path)

    return results


def get_pysr_model_selections(args):
    """
    if pysr_model_selection is provided as an arg, returns it
    otherwise, returns all available model selections.
    """
    # get the model selections by grepping for the files
    path = get_results_path(args.Ngrid,
                            args.version,
                            pysr_version=args.pysr_version,
                            pysr_model_selection='*')
    files = os.listdir(os.path.dirname(path))
    files = [f for f in files if f.endswith('.pkl')]

    # go from f'{model_selection}.pkl' to model_selection
    model_selections = [
        None if f[:-4] == 'None' else int(f.split('.')[0])
        for f in files
    ]
    model_selections = sorted(model_selections)

    # if model selection is provided, filter to those
    if args.pysr_model_selection is not None:
        if args.pysr_model_selection == 'accuracy':
            model_selections = [model_selections[-1]]
        else:
            model_selections = [m for m in model_selections if str(m) == args.pysr_model_selection]

    return model_selections


def plot_results_pysr_f2(args):
    model_selections = get_pysr_model_selections(args)

    original_model_selection = args.pysr_model_selection
    for model_selection in model_selections:
        args.pysr_model_selection = model_selection
        args.title = f'Equation complexity = {model_selection}'
        plot_results(args)

    args.pysr_model_selection = original_model_selection


def plot_4way_comparison(args):
    v = load_json(args.version_json)

    # Create the figure and axes
    # fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    scale = 0.85
    fig, axs = plt.subplots(
        2, 2,
        figsize=(12*scale, 10*scale),
        gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
    )
    axs = axs.flatten()

    show_xs = [False, False, True, True]
    show_ys = [True, False, True, False]

    for ax in axs:
        ax.set_aspect('equal', adjustable='box')

    ticks = [0.55, 0.60, 0.65, 0.70, 0.75]
    # major_ticks=[0.55, 0.75]
    # minor_ticks=[0.60, 0.65, 0.70]
    for ax, show_x, show_y in zip(axs, show_xs, show_ys):
        # ax.set_xticks(major_ticks, major=True)
        ax.set_xticks(ticks, major=True)
        # ax.set_xticks(minor_ticks, minor=True)

        if not show_x:
            ax.set_xticklabels([])

        # ax.set_yticks(major_ticks, major=True)
        ax.set_yticks(ticks, major=True)
        # ax.set_yticks(minor_ticks, minor=True)

        if not show_y:
            ax.set_yticklabels([])

    nn_results = load_pickle(get_results_path(args.Ngrid, v['nn_version']))
    eq_results = load_pickle(get_results_path(args.Ngrid, v['nn_version'], pysr_version=v['pysr_version'], pysr_model_selection=v['pysr_model_selection']))
    petit_results = load_pickle(get_results_path(args.Ngrid, use_petit=True))
    ground_truth_results = load_pickle(get_results_path(args.Ngrid, ground_truth=True))

    # model_results = [nn_results, eq_results, petit_results, megno_results]
    # names = ['nn', 'eq', 'petit', 'megno']
    # titles = ['Neural network', 'Distilled equations', 'Petit+ 2020', 'MEGNO'][i]
    model_results = [ground_truth_results, eq_results, nn_results, petit_results]
    names = ['ground_truth', 'eq', 'nn', 'petit']
    titles = ['Ground truth', 'Distilled equations', 'Neural network', 'Petit+ 2020']

    for i in range(4):
        results = model_results[i]
        name = names[i]
        show_x = show_xs[i]
        show_y = show_ys[i]

        P12s, P23s = get_period_ratios(args.Ngrid)

        # get the results for the specific metric
        if name == 'petit':
            results = [d['mean'] if d is not None else np.nan for d in results]
        elif name == 'megno':
            results = [d['megno'] if d is not None else np.nan for d in results]
        elif name == 'ground_truth':
            results = [np.log10(d['ground_truth']) if d is not None else np.nan for d in results]
            # we want to color times < 4 white
            results = [r if r >= 4 else np.nan for r in results]
        else:
            results = [d['mean'] if d is not None else np.nan for d in results]

        # results = [4. if np.isnan(r) else r for r in results]
        results = np.array(results)
        X,Y,Z = get_centered_grid(P12s, P23s, results)

        # NaN's get mapped to predicting instant instability
        # Z[np.isnan(Z)] = 4
        cmap = COLOR_MAP.copy().reversed()
        cmap.set_bad(color='white')
        im = axs[i].pcolormesh(X, Y, Z, vmin=4, vmax=9, cmap=cmap, rasterized=True)

        if show_x:
            axs[i].set_xlabel(r"$P_1/P_2$")
        else:
            axs[i].set_xlabel("")
        if show_y:
            axs[i].set_ylabel(r"$P_2/P_3$")
        else:
            axs[i].set_ylabel("")

        axs[i].set_title(titles[i])

        if args.resonance_guides:
            axs[i].axvline(2 / 3, color='black', lw=0.9, ls='--', alpha=0.7)
            axs[i].axhline(2 / 3, color='black', lw=0.9, ls='--', alpha=0.7)

    # Create a single colorbar
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
    cb.set_label(INSTABILITY_TIME_LABEL)

    path = 'period_results/4way'
    if args.resonance_guides:
        path += '_resonance_guides'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_path = path + ('.png' if args.png else '.pdf')
    plt.savefig(img_path, dpi=800, bbox_inches='tight')
    print('Saved figure to', img_path)
    plt.close(fig)


def plot_4way_std(args):
    v = load_json(args.version_json)
    nn_version = v['nn_version']
    pysr_version = v['pysr_version']
    pysr_model_selection = v.get('pysr_model_selection', 'accuracy')

    nn_results = load_pickle(get_results_path(args.Ngrid, nn_version))
    eq_results = load_pickle(get_results_path(
        args.Ngrid, nn_version,
        pysr_version=pysr_version,
        pysr_model_selection=pysr_model_selection,
    ))
    eq_std_results = load_pickle(get_results_path(args.Ngrid, v['eq_std_nn_version']))

    P12s, P23s = get_period_ratios(args.Ngrid)

    def extract(results, key):
        arr = [d[key] if d is not None else np.nan for d in results]
        return np.array(arr)

    panels = [
        ('Neural network predicted mean', extract(nn_results, 'mean'), 'mean'),
        ('Neural network predicted std', extract(nn_results, 'std'), 'std'),
        (f'Distilled equations', extract(eq_results, 'mean'), 'mean'),
        (f'NN predicted std of distilled equations', extract(eq_std_results, 'std'), 'std'),
    ]

    scale = 0.85
    fig, axs = plt.subplots(
        2, 2,
        figsize=(12 * scale, 10 * scale),
        gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
    )
    axs = axs.flatten()

    show_xs = [False, False, True, True]
    show_ys = [True, False, True, False]
    ticks = [0.55, 0.60, 0.65, 0.70, 0.75]

    mean_im = None
    std_im = None

    for i, (title, results, kind) in enumerate(panels):
        ax = axs[i]
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if not show_xs[i]:
            ax.set_xticklabels([])
        if not show_ys[i]:
            ax.set_yticklabels([])

        X, Y, Z = get_centered_grid(P12s, P23s, results)
        cmap = COLOR_MAP.copy().reversed()
        cmap.set_bad(color='white')

        if kind == 'mean':
            im = ax.pcolormesh(X, Y, Z, vmin=4, vmax=9, cmap=cmap, rasterized=True)
            mean_im = im
        else:
            im = ax.pcolormesh(X, Y, Z, vmin=0, vmax=3, cmap=cmap, rasterized=True)
            std_im = im

        if show_xs[i]:
            ax.set_xlabel(r"$P_1/P_2$")
        if show_ys[i]:
            ax.set_ylabel(r"$P_2/P_3$")
        ax.set_title(title)

    cb_mean = fig.colorbar(mean_im, ax=[axs[0], axs[2]], shrink=0.6, location='right')
    cb_mean.set_label(INSTABILITY_TIME_LABEL)
    cb_std = fig.colorbar(std_im, ax=[axs[1], axs[3]], shrink=0.6, location='right')
    cb_std.set_label("std(" + INSTABILITY_TIME_LABEL + ")")

    path = f'period_results/4way_std'
    img_path = path + ('.png' if args.png else '.pdf')
    plt.savefig(img_path, dpi=800, bbox_inches='tight')
    print('Saved figure to', img_path)
    plt.close(fig)


def plot_4way_rmse(args):
    """
    Four-panel plot showing per-cell RMSE vs. ground truth for:
      NN, distilled equations, pure SR, and Petit+2020.

    If args.exclude_stable, cells where ground truth == 9 (stable) are
    set to RMSE = 0 rather than NaN so they appear as "no error" in the
    colormap instead of blank.
    """
    v = load_json(args.version_json)
    Ngrid = args.Ngrid
    P12s, P23s = get_period_ratios(Ngrid)

    # --- ground truth ---
    gt_data = load_pickle(get_results_path(Ngrid, ground_truth=True))
    gt_vals = np.array([
        np.log10(d['ground_truth']) if d is not None else np.nan
        for d in gt_data
    ])
    stable_mask = gt_vals >= 9

    # --- per-model predictions ---
    def _mean(data):
        return np.array([d['mean'] if d is not None else np.nan for d in data])

    nn_preds   = _mean(load_pickle(get_results_path(Ngrid, v['nn_version'])))
    eq_preds   = _mean(load_pickle(get_results_path(Ngrid, v['nn_version'],
                       pysr_version=v['pysr_version'],
                       pysr_model_selection=v['pysr_model_selection'])))
    psr_preds  = _mean(load_pickle(get_results_path(Ngrid,
                       pysr_version=v['pure_sr_version'],
                       pysr_model_selection=v['pure_sr_model_selection'],
                       pure_sr=True)))
    pet_preds  = _mean(load_pickle(get_results_path(Ngrid, use_petit=True)))

    all_preds  = [nn_preds, eq_preds, psr_preds, pet_preds]
    titles     = ['Neural network', 'Distilled equations', 'Pure SR', 'Petit+ 2020']

    # --- per-cell RMSE (= abs error; one simulation per cell) ---
    all_Z = []
    for preds in all_preds:
        preds_clip = np.clip(preds, 4, 9)
        Z = np.abs(preds_clip - gt_vals)
        if args.exclude_stable:
            Z[stable_mask] = 0.0
        all_Z.append(Z)

    # --- figure ---
    scale = 0.85
    fig, axs = plt.subplots(
        2, 2,
        figsize=(12 * scale, 10 * scale),
        gridspec_kw={'wspace': 0.15, 'hspace': 0.15},
    )
    axs = axs.flatten()

    show_xs    = [False, False, True,  True]
    show_ys    = [True,  False, True,  False]
    major_ticks = [0.55, 0.65, 0.75]
    minor_ticks = [0.55, 0.60, 0.65, 0.70, 0.75]

    cmap = RMSE_CMAP.copy()
    cmap.set_bad(color='white')
    norm = plt.Normalize(vmin=0, vmax=5)

    im = None
    for i, (Z, title, show_x, show_y) in enumerate(
            zip(all_Z, titles, show_xs, show_ys)):
        ax = axs[i]
        ax.set_aspect('equal')
        ax.set_xticks(major_ticks, major=True)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks, major=True)
        ax.set_yticks(minor_ticks, minor=True)

        X, Y, ZZ = get_centered_grid(P12s, P23s, Z)
        im = ax.pcolormesh(X, Y, ZZ, cmap=cmap, norm=norm, rasterized=True)
        ax.set_title(title)

        if show_x:
            ax.set_xlabel(r"$P_1/P_2$")
        else:
            ax.set_xticklabels([])
        if show_y:
            ax.set_ylabel(r"$P_2/P_3$")
        else:
            ax.set_yticklabels([])

    cb = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
    cb.set_label("RMSE (dex)")

    out_path = 'period_results/four_way_rmse.pdf'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=800, bbox_inches='tight')
    print('Saved figure to', out_path)
    plt.close(fig)


def plot_pure_sr_comparison(args):
    v = load_json(args.version_json)

    # Create the figure and axes
    # fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig, axs = plt.subplots(
        2, 2,
        figsize=(12, 10),
        gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
    )
    axs = axs.flatten()

    show_xs = [False, False, True, True]
    show_ys = [True, False, True, False]

    for ax in axs:
        ax.set_aspect('equal', adjustable='box')

    ticks = [0.55, 0.60, 0.65, 0.70, 0.75]
    for ax, show_x, show_y in zip(axs, show_xs, show_ys):
        ax.set_xticks(ticks)
        if not show_x:
            ax.set_xticklabels([])
        ax.set_yticks(ticks)
        if not show_y:
            ax.set_yticklabels([])

    ground_truth_results = load_pickle(get_results_path(args.Ngrid, ground_truth=True))
    eq_results = load_pickle(get_results_path(args.Ngrid, version=v['nn_version'], pysr_version=v['pysr_version'], pysr_model_selection=v['pysr_model_selection']))
    pure_sr_results = load_pickle(get_results_path(args.Ngrid, pysr_version=v['pure_sr_version'], pysr_model_selection=v['pure_sr_model_selection'], pure_sr=True))
    pure_sr2_results = load_pickle(get_results_path(args.Ngrid, version=28114, pysr_version=v['pure_sr2_version'], pysr_model_selection=v['pure_sr2_model_selection']))

    model_results = [ground_truth_results, eq_results, pure_sr_results, pure_sr2_results]
    titles = ['Ground truth', 'Distilled equations', 'Pure SR', 'Pure SR (no intermediate features)']

    for i in range(4):
        results = model_results[i]
        show_x = show_xs[i]
        show_y = show_ys[i]
        title = titles[i]

        P12s, P23s = get_period_ratios(args.Ngrid)

        if title == 'Ground truth':
            results = [np.log10(d['ground_truth']) if d is not None else np.nan for d in results]
            results = [np.nan if r <= 4. else r for r in results]
        else:
            results = [d['mean'] if d is not None else np.nan for d in results]
        # results = [4. if np.isnan(r) else r for r in results]
        results = np.array(results)
        X,Y,Z = get_centered_grid(P12s, P23s, results)

        cmap = COLOR_MAP.copy().reversed()
        # cmap.set_bad(color='white')
        im = axs[i].pcolormesh(X, Y, Z, vmin=4, vmax=9, cmap=cmap, rasterized=True)

        if show_x:
            axs[i].set_xlabel(r"$P_1/P_2$")
        else:
            axs[i].set_xlabel("")
        if show_y:
            axs[i].set_ylabel(r"$P_2/P_3$")
        else:
            axs[i].set_ylabel("")

        axs[i].set_title(title)

    # Create a single colorbar
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
    cb.set_label(INSTABILITY_TIME_LABEL)

    path = 'period_results/period_ratio_pure_sr'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_path = path + ('.png' if args.png else '.pdf')
    plt.savefig(img_path, dpi=800, bbox_inches='tight')
    print('Saved figure to', img_path)
    plt.close(fig)


def plot_exprs(args):
    results = load_pickle(get_results_path(args.Ngrid, args.version))
    n_features = results[0]['f1'].shape[0] // 2

    P12s, P23s = get_period_ratios(args.Ngrid)

    # get value array for each feature
    all_values = []
    for i in range(n_features * 2):
        values = [d['f1'][i] if d is not None else np.nan for d in results]
        values = np.array(values)
        all_values.append(values)
        # values = values[~np.isnan(values)]

    var_names = [f'm{i}' for i in range(10)] + [f's{i}' for i in range(10)]
    var_name_to_values = dict(zip(var_names, all_values))
    v = var_name_to_values

    # exprs = {
    #     -v['m2'],
    #     3.6422653/(v['s4']**0.15489304),
    #     0.058552526**v['s1'],
    #     v['s4']**(-0.32957777),
    #     -np.sin(v['m2']),
    #     0.0841594**v['s1'],
    #     (v['s6']**0.35633504*(v['s2'] + v['s4']))**(-0.3054036),
    #     (v['m7'] - v['s8']),
    #     1.2004135**v['m1'],
    #     (v['m7'] - v['s8'])/1.2004135**v['m1']
    # }

    expr_dict = {
        # "-v['m2']": -v['m2'],
        # "3.6422653/(v['s4']**0.15489304)": 3.6422653/(v['s4']**0.15489304),
        # "0.058552526**v['s1']": 0.058552526**v['s1'],
        # "v['s4']**(-0.32957777)": v['s4']**(-0.32957777),
        # "-np.sin(v['m2'])": -np.sin(v['m2']),
        # "0.0841594**v['s1']": 0.0841594**v['s1'],
        # "(v['s6']**0.35633504*(v['s2'] + v['s4']))**(-0.3054036)": (v['s6']**0.35633504*(v['s2'] + v['s4']))**(-0.3054036),
        # "(v['m7'] - v['s8'])": (v['m7'] - v['s8']),
        # "1.2004135**v['m1']": 1.2004135**v['m1'],
        # "(v['m7'] - v['s8'])/1.2004135**v['m1']": (v['m7'] - v['s8'])/1.2004135**v['m1'],
        '(s6**0.35633504)': (v['s6']**0.35633504),
        # '(s6**0.35633504*(s2 + s4))**(-0.3054036) - sin(m2) + (m7 - s8)/1.2004135**m1)': (v['s6']**0.35633504*(v['s2'] + v['s4']))**(-0.3054036) - np.sin(v['m2']) + (v['m7'] - v['s8'])/1.2004135**v['m1'],
    }

    for contrast in [True, False]:
        for i, (s, values) in enumerate(expr_dict.items()):
            if contrast:
                values2 = values[~np.isnan(values)]
                fraction = 0.5
                lower, upper = np.percentile(values2, [100 * (fraction/2), 100 * (1 - fraction/2)])

            X,Y,Z = get_centered_grid(P12s, P23s, values)

            fig, ax = plt.subplots(figsize=(8,6))

            cmap = COLOR_MAP.copy().reversed()
            cmap.set_bad(color='white')
            if contrast:
                im = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=lower, vmax=upper)
            else:
                im = ax.pcolormesh(X, Y, Z, cmap=cmap)
            ax.set_title(s)
            ax.set_xlabel(r"$P_1/P_2$")
            ax.set_ylabel(r"$P_2/P_3$")

            cb = plt.colorbar(im, ax=ax)
            cb.set_label(s)
            ax.set_xlabel(r"$P_1/P_2$")
            ax.set_ylabel(r"$P_2/P_3$")
            plt.tight_layout()

            path = get_results_path(args.Ngrid, args.version)
            # get rid of .pkl
            path = path[:-4]
            if contrast:
                path += f'_plot_exprs/{i}_contrast'
            else:
                path += f'_plot_exprs/{i}'
            path += '.png' if args.png else '.pdf'

            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=800)
            print('Saved figure to', path)
            plt.close(fig)


def plot_f1_features2(args):
    """
    Display the first six μ/σ features on one page (3 × 2 grid),
    and the remaining four on a second page (2 × 2 grid).  All
    formatting, colour scaling, and file-naming conventions follow
    the original implementation.
    """

    v = load_json(args.version_json)
    Ngrid = 300
    nn_version = v['nn_version']
    pysr_version = v['pysr_version']

    # -------------------- load data / discover indices -------------------- #
    results = load_pickle(get_results_path(Ngrid, nn_version))
    n_features = results[0]['f1'].shape[0] // 2
    P12s, P23s = get_period_ratios(Ngrid)

    from interpret import get_variables_in_str, feature_string, get_feature_nn
    feature_nn = get_feature_nn(nn_version)

    pysr_results = pickle.load(
        open(f'../sr_results/{pysr_version}.pkl', 'rb')
    ).equations_
    if isinstance(results, list):
        pysr_results = pysr_results[0]      # only the mean equations

    # work out which f1-indices appear in an important symbolic expression
    all_vars = [
        v
        for sub in pysr_results['equation']
        for v in get_variables_in_str(sub)
    ]
    all_vars = list(dict.fromkeys(all_vars))          # dedupe, keep order
    mu_vars  = sorted(int(v[1]) for v in all_vars if v[0] == 'm')
    std_vars = sorted(int(v[1]) for v in all_vars if v[0] == 's')

    feat_indices = mu_vars + std_vars                 # 10 total
    is_std_flags = [False] * len(mu_vars) + [True] * len(std_vars)

    # ---------------------- helper to draw one figure --------------------- #
    def _draw_subset(sub_idx, sub_is_std, part_tag):
        n_plots = len(sub_idx)
        n_cols  = 2
        n_rows  = math.ceil(n_plots / n_cols)
        scale   = 0.85

        fig, axs = plt.subplots(
            n_rows, n_cols,
            figsize=(12 * scale, 10 * scale * n_rows / 2),
            gridspec_kw={'wspace': 0.15, 'hspace': 0.15}
        )
        axs = axs.flatten()

        # remove any unused axes (happens only for the second figure)
        for ax in axs[n_plots:]:
            ax.remove()
        axs = axs[:n_plots]

        major_ticks = [0.55, 0.65, 0.75]
        minor_ticks = [0.55, 0.60, 0.65, 0.70, 0.75]

        for ax_id, (ax, feat_i, is_std) in enumerate(zip(axs, sub_idx, sub_is_std)):
            # --------- populate each subplot (identical to old single-plot) ---- #
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks(major_ticks, minor=False)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(major_ticks, minor=False)
            ax.set_yticks(minor_ticks, minor=True)

            row, col = divmod(ax_id, n_cols)
            if row < n_rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"$P_1/P_2$")
            if col == 1:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(r"$P_2/P_3$")

            vals = np.array([
                d['f1'][feat_i + n_features if is_std else feat_i]
                if d is not None else np.nan
                for d in results
            ])
            valid = vals[~np.isnan(vals)]
            lower, upper = np.percentile(valid, [20, 80])   # middle 50 %
            X, Y, Z = get_centered_grid(P12s, P23s, vals)

            cmap = COLOR_MAP.copy().reversed()
            cmap.set_bad(color='white')

            im = ax.pcolormesh(
                X, Y, Z,
                vmin=lower, vmax=upper,
                cmap=cmap,
                rasterized=True
            )

            feat_str = feature_string(
                feature_nn, feat_i,
                include_ssx=True,
                latex=True,
                include_ssx_bias=not is_std,
                drop_mass=True
            )
            if is_std:
                title = r'$\sigma_{' + str(feat_i) + r'} = {\rm Std} \left(' + feat_str + r'\right)$'
            else:
                title = r'$\mu_{' + str(feat_i) + r'} = \mathbb{E} \left [' + feat_str + r' \right ]$'
            ax.set_title(title)

            fig.colorbar(im, ax=ax)

        # ---------------------------- save figure --------------------------- #
        out_dir = f"period_results/v={nn_version}_ngrid={Ngrid}_plot_features"
        os.makedirs(out_dir, exist_ok=True)
        ext = '.pdf'
        fname = f"{out_dir}/features_grid_{part_tag}{ext}"
        plt.savefig(fname, dpi=800, bbox_inches='tight')
        print("Saved combined figure to", fname)
        plt.close(fig)

    # ------------------------- render both figures ------------------------ #
    _draw_subset(feat_indices[:6],  is_std_flags[:6],  "part1")   # first  6 panels
    _draw_subset(feat_indices[6:], is_std_flags[6:], "part2")     # last   4 panels


def plot_f1_features(args):
    results = load_pickle(get_results_path(args.Ngrid, args.version))
    n_features = results[0]['f1'].shape[0] // 2

    P12s, P23s = get_period_ratios(args.Ngrid)

    from interpret import get_variables_in_str, feature_string, get_feature_nn
    feature_nn = get_feature_nn(args.version)

    pysr_results = pickle.load(open(f'../sr_results/{args.pysr_version}.pkl', 'rb'))
    pysr_results = pysr_results.equations_
    if type(results) == list:
        pysr_results = pysr_results[0]  # just the mean equations

    # only print variables used in an important equation in the pysr results
    all_vars = [get_variables_in_str(e) for e in pysr_results['equation']]
    # go from list of lists to just one big list
    all_vars = [item for sublist in all_vars for item in sublist]
    all_vars = list(dict.fromkeys(all_vars)) # remove duplicates, but keep order
    mu_vars = [int(s[1]) for s in all_vars if s[0] == 'm']
    std_vars = [int(s[1]) for s in all_vars if s[0] == 's']
    mu_vars = sorted(mu_vars)
    std_vars = sorted(std_vars)
    print(mu_vars, std_vars)

    mu_is_std = [False for _ in range(len(mu_vars))]
    std_is_std = [True for _ in range(len(std_vars))]

    for i, is_std in zip(mu_vars + std_vars, mu_is_std + std_is_std):

        values = [d['f1'][i + n_features if is_std else i] if d is not None else np.nan for d in results]
        values = np.array(values)
        values2 = values[~np.isnan(values)]
        fraction = 0.5
        lower, upper = np.percentile(values2, [100 * (fraction/2), 100 * (1 - fraction/2)])

        X,Y,Z = get_centered_grid(P12s, P23s, values)
        fig, ax = plt.subplots(figsize=(8,6))
        cmap = COLOR_MAP.copy().reversed()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=lower, vmax=upper, rasterized=True)

        feature_str = feature_string(feature_nn, i, include_ssx=True, latex=True, include_ssx_bias=not is_std)
        if is_std:
            title = r'$\sigma_{' + str(i) + r'} = {\rm Std}\left(' + feature_str + r'\right)$'
        else:
            title = r'$\mu_{' + str(i) + '} = { \\rm \mathbb{E}} \\left [' + feature_str + ' \\right ]$'
        ax.set_title(title)

        ax.set_xlabel(r"$P_1/P_2$")
        ax.set_ylabel(r"$P_2/P_3$")

        cb = plt.colorbar(im, ax=ax)
        cb.set_label(f'Feature {i} std' if is_std else f'Feature {i} mean')
        ax.set_xlabel(r"$P_1/P_2$")
        ax.set_ylabel(r"$P_2/P_3$")
        plt.tight_layout()

        path = get_results_path(args.Ngrid, args.version)
        # get rid of .pkl
        path = path[:-4]
        path += f'_plot_features/{i}'
        path += '_std' if is_std else '_mean'
        path += '.png' if args.png else '.pdf'

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
        results = load_pickle(get_results_path(args.Ngrid, args.version, pysr_version=args.pysr_version, pysr_model_selection=str(model_selection)))

        P12s, P23s = get_period_ratios(args.Ngrid)

        # get the results for the mean
        results = [d['mean'] if d is not None else np.nan for d in results]

        X,Y,Z = get_centered_grid(P12s, P23s, results)

        cmap = COLOR_MAP.copy().reversed()
        cmap.set_bad(color='white')
        zmax = 12
        zmin = 4
        im = axs[i].pcolormesh(X, Y, Z, vmin=zmin, vmax=zmax, cmap=cmap)
        axs[i].set_title(f'Complexity {model_selection}')
        axs[i].set_xlabel(r"$P_1/P_2$")
        axs[i].set_ylabel(r"$P_2/P_3$")

    # Create a single colorbar
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
    cb.set_label(INSTABILITY_TIME_LABEL)
    fig.set_constrained_layout(True)
    # fig.subplots_adjust(wspace=0.05, hspace=0.05)

    path = get_results_path(args.Ngrid, args.version, pysr_version=args.pysr_version, pysr_model_selection='comparison')[:-4]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_path = path + ('.png' if args.png else '.pdf')
    plt.savefig(img_path, dpi=800)
    print('Saved figure to', path + '.png')
    plt.close(fig)


def plot_selective_eq_6way(args):
    """2x3 plot:
      col 0: baseline 24880/11003 c=26 prediction (top) + ground truth (bottom)
      cols 1-2: 4 selective_eq panels at p=0.10, 0.25, 0.50, 0.75 (each at
                its best complexity from pickles/eval_selective_eq.pkl). Each
                panel shows baseline predictions, with non-selected points
                (where the SR score is high) masked white.

    Reads SR equation files from args.pysr_dir and best complexities from
    pickles/eval_selective_eq.pkl.
    """
    import pysr  # only needed for this plot

    v = load_json(args.version_json)
    Ngrid = args.Ngrid

    # --- load grid data ---
    nn_data = load_pickle(get_results_path(Ngrid, v['nn_version']))
    base_data = load_pickle(get_results_path(
        Ngrid, v['nn_version'],
        pysr_version=v['pysr_version'],
        pysr_model_selection=v['pysr_model_selection']))
    gt_data = load_pickle(get_results_path(Ngrid, ground_truth=True))

    valid = np.array([d is not None for d in nn_data])
    base_means = np.array([d['mean'] if d is not None else np.nan for d in base_data])
    gt_log = np.log10(np.array([d['ground_truth'] if d is not None else np.nan for d in gt_data]))
    gt_log = np.where(gt_log >= 4, gt_log, np.nan)

    f1 = np.full((len(nn_data), 20), np.nan, dtype=np.float64)
    for i, d in enumerate(nn_data):
        if d is not None:
            f1[i] = d['f1']

    # --- load selective_eq best-complexity table ---
    eval_path = '../pickles/eval_selective_eq.pkl'
    if not os.path.exists(eval_path):
        eval_path = 'pickles/eval_selective_eq.pkl'
    eval_info = load_pickle(eval_path)
    runs = sorted(eval_info.items(), key=lambda kv: kv[1]['p'])  # p ascending

    # --- compute selector masks ---
    sel_masks = []
    for pysr_version, info in runs:
        path = os.path.join(args.pysr_dir, f'{pysr_version}.pkl')
        reg = pysr.PySRRegressor.from_file(path)
        eqs = reg.equations_
        if isinstance(eqs, list):
            eqs = eqs[0]
        c_best = info['best_complexity']
        ix = int(np.argmin(np.abs(eqs['complexity'].values - c_best)))
        score_fn = eqs.iloc[ix]['lambda_format']

        scores = np.full(len(nn_data), np.inf)
        scores[valid] = np.asarray(score_fn(f1[valid]))

        # lowest-p fraction among VALID points
        p = info['p']
        valid_ix = np.where(valid)[0]
        k = max(1, int(np.floor(p * len(valid_ix))))
        order = np.argsort(scores[valid_ix])
        mask = np.zeros(len(nn_data), dtype=bool)
        mask[valid_ix[order[:k]]] = True
        sel_masks.append(mask)

    # --- plot ---
    P12s, P23s = get_period_ratios(Ngrid)
    scale = 0.95
    fig, axs = plt.subplots(
        2, 3,
        figsize=(13 * scale, 8 * scale),
        gridspec_kw={'wspace': 0.05, 'hspace': 0.12},
    )

    major_ticks = [0.55, 0.65, 0.75]
    minor_ticks = [0.55, 0.60, 0.65, 0.70, 0.75]

    def _style(ax, show_x, show_y, title):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(major_ticks, major=True)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks, major=True)
        ax.set_yticks(minor_ticks, minor=True)
        if not show_x:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$P_1/P_2$")
        if not show_y:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$P_2/P_3$")
        ax.set_title(title, fontsize=10)

    cmap = COLOR_MAP.copy().reversed()
    cmap.set_bad(color='white')

    # (0,0) baseline equation prediction
    X, Y, Z = get_centered_grid(P12s, P23s, base_means)
    im = axs[0, 0].pcolormesh(X, Y, Z, vmin=4, vmax=9, cmap=cmap, rasterized=True)
    _style(axs[0, 0], show_x=False, show_y=True,
           title=f"Baseline (pysr {v['pysr_version']}, c={v['pysr_model_selection']})")

    # (1,0) ground truth
    X, Y, Z = get_centered_grid(P12s, P23s, gt_log)
    axs[1, 0].pcolormesh(X, Y, Z, vmin=4, vmax=9, cmap=cmap, rasterized=True)
    _style(axs[1, 0], show_x=True, show_y=True, title='Ground truth')

    # selective_eq panels — order matches (p ascending) the row-major reading
    # of (0,1) (0,2) (1,1) (1,2)
    panel_pos = [(0, 1), (0, 2), (1, 1), (1, 2)]
    n_valid = int(valid.sum())
    for (r, c), (pysr_version, info), mask in zip(panel_pos, runs, sel_masks):
        masked = np.where(mask, base_means, np.nan)
        X, Y, Z = get_centered_grid(P12s, P23s, masked)
        axs[r, c].pcolormesh(X, Y, Z, vmin=4, vmax=9, cmap=cmap, rasterized=True)
        title = (f'selective_eq p={info["p"]:.2f}  c*={info["best_complexity"]}\n'
                 f'{int(mask.sum()):,}/{n_valid:,} pts highlighted')
        _style(axs[r, c], show_x=(r == 1), show_y=False, title=title)

    # shared colorbar on the right
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7, pad=0.02)
    cb.set_label(INSTABILITY_TIME_LABEL)

    out_path = 'period_results/selective_eq_6way' + ('.png' if args.png else '.pdf')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print('Saved figure to', out_path)
    plt.close(fig)


def plot_selective_eq_6way_rmse(args):
    """Same layout as plot_selective_eq_6way, but every panel except the
    ground-truth panel shows baseline-vs-GT RMSE (|clip(pred)-gt|) instead
    of the prediction value.
    """
    import pysr

    v = load_json(args.version_json)
    Ngrid = args.Ngrid

    nn_data = load_pickle(get_results_path(Ngrid, v['nn_version']))
    base_data = load_pickle(get_results_path(
        Ngrid, v['nn_version'],
        pysr_version=v['pysr_version'],
        pysr_model_selection=v['pysr_model_selection']))
    gt_data = load_pickle(get_results_path(Ngrid, ground_truth=True))

    valid = np.array([d is not None for d in nn_data])
    base_means = np.array([d['mean'] if d is not None else np.nan for d in base_data])
    gt_raw = np.log10(np.array([d['ground_truth'] if d is not None else np.nan
                                for d in gt_data]))
    gt_for_panel = np.where(gt_raw >= 4, gt_raw, np.nan)  # white on <4 like other plots

    # baseline RMSE field (per point); GT outside [4,9] is clipped for comparison
    base_clip = np.clip(base_means, 4, 9)
    gt_clip = np.clip(gt_raw, 4, 9)
    base_rmse = np.abs(base_clip - gt_clip)  # = sqrt((a-b)^2)
    base_rmse[np.isnan(base_means) | np.isnan(gt_raw)] = np.nan

    f1 = np.full((len(nn_data), 20), np.nan, dtype=np.float64)
    for i, d in enumerate(nn_data):
        if d is not None:
            f1[i] = d['f1']

    eval_path = '../pickles/eval_selective_eq.pkl'
    if not os.path.exists(eval_path):
        eval_path = 'pickles/eval_selective_eq.pkl'
    eval_info = load_pickle(eval_path)
    runs = sorted(eval_info.items(), key=lambda kv: kv[1]['p'])

    sel_masks = []
    for pysr_version, info in runs:
        path = os.path.join(args.pysr_dir, f'{pysr_version}.pkl')
        reg = pysr.PySRRegressor.from_file(path)
        eqs = reg.equations_
        if isinstance(eqs, list):
            eqs = eqs[0]
        c_best = info['best_complexity']
        ix = int(np.argmin(np.abs(eqs['complexity'].values - c_best)))
        score_fn = eqs.iloc[ix]['lambda_format']
        scores = np.full(len(nn_data), np.inf)
        scores[valid] = np.asarray(score_fn(f1[valid]))
        valid_ix = np.where(valid)[0]
        k = max(1, int(np.floor(info['p'] * len(valid_ix))))
        order = np.argsort(scores[valid_ix])
        mask = np.zeros(len(nn_data), dtype=bool)
        mask[valid_ix[order[:k]]] = True
        sel_masks.append(mask)

    # --- plot ---
    P12s, P23s = get_period_ratios(Ngrid)
    scale = 0.95
    fig, axs = plt.subplots(
        2, 3,
        figsize=(13 * scale, 8 * scale),
        gridspec_kw={'wspace': 0.05, 'hspace': 0.12},
    )

    major_ticks = [0.55, 0.65, 0.75]
    minor_ticks = [0.55, 0.60, 0.65, 0.70, 0.75]

    def _style(ax, show_x, show_y, title):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(major_ticks, major=True)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks, major=True)
        ax.set_yticks(minor_ticks, minor=True)
        if not show_x:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$P_1/P_2$")
        if not show_y:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$P_2/P_3$")
        ax.set_title(title, fontsize=10)

    rmse_cmap = RMSE_CMAP.copy()
    rmse_cmap.set_bad(color='#d9d9d9')  # light grey for not-selected / invalid sims
    rmse_norm = plt.Normalize(vmin=0, vmax=5)

    gt_cmap = COLOR_MAP.copy().reversed()
    gt_cmap.set_bad(color='white')

    # (0,0) baseline RMSE — full grid
    X, Y, Z = get_centered_grid(P12s, P23s, base_rmse)
    im_rmse = axs[0, 0].pcolormesh(X, Y, Z, norm=rmse_norm, cmap=rmse_cmap, rasterized=True)
    _style(axs[0, 0], show_x=False, show_y=True,
           title=f"Baseline RMSE (pysr {v['pysr_version']}, c={v['pysr_model_selection']})")

    # (1,0) ground truth — unchanged
    X, Y, Z = get_centered_grid(P12s, P23s, gt_for_panel)
    im_gt = axs[1, 0].pcolormesh(X, Y, Z, vmin=4, vmax=9, cmap=gt_cmap, rasterized=True)
    _style(axs[1, 0], show_x=True, show_y=True, title='Ground truth')

    # selective_eq panels — show baseline RMSE masked to selected points
    panel_pos = [(0, 1), (0, 2), (1, 1), (1, 2)]
    n_valid = int(valid.sum())
    for (r, c), (pysr_version, info), mask in zip(panel_pos, runs, sel_masks):
        masked = np.where(mask, base_rmse, np.nan)
        X, Y, Z = get_centered_grid(P12s, P23s, masked)
        axs[r, c].pcolormesh(X, Y, Z, norm=rmse_norm, cmap=rmse_cmap, rasterized=True)
        title = (f'selective_eq p={info["p"]:.2f}  c*={info["best_complexity"]}\n'
                 f'{int(mask.sum()):,}/{n_valid:,} pts highlighted')
        _style(axs[r, c], show_x=(r == 1), show_y=False, title=title)

    # two colorbars — RMSE (shared by 5 panels) + GT (only for the GT panel)
    cb_rmse = fig.colorbar(im_rmse, ax=[axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 1], axs[1, 2]],
                           shrink=0.7, pad=0.02)
    cb_rmse.set_label('RMSE (dex)')
    cb_gt = fig.colorbar(im_gt, ax=axs[1, 0], shrink=0.7, pad=0.02)
    cb_gt.set_label(INSTABILITY_TIME_LABEL)

    out_path = 'period_results/selective_eq_6way_rmse' + ('.png' if args.png else '.pdf')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print('Saved figure to', out_path)
    plt.close(fig)


def get_truths_and_preds(Ngrid, version=None, pysr_version=None, pysr_model_selection=None, petit=False, pure_sr=False, clip=True, exclude_stable=True):
    results_path = get_results_path(Ngrid, version, pysr_version=pysr_version, pysr_model_selection=pysr_model_selection, use_petit=petit, pure_sr=pure_sr)
    truth_path = get_results_path(Ngrid, ground_truth=True)
    if not os.path.exists(results_path):
        results_path = 'figures/' + results_path
        truth_path = 'figures/' + truth_path

    preds = load_pickle(results_path)
    truths = load_pickle(truth_path)

    truths = np.array([d['ground_truth'] if d is not None else np.nan for d in truths])
    truths = np.log10(truths)

    preds = np.array([d['mean'] if d is not None else np.nan for d in preds])
    assert_equal(truths.shape, preds.shape)

    # ignore entries where ground_truth is nan or input_cache is None
    # input_cache being None signals that simulating to 10^4 failed (aka early collision)
    input_cache = load_input_cache(Ngrid)
    bad_input_ixs = np.array([inp is None for inp in input_cache])
    nan_ixs = np.isnan(truths)
    bad_ixs = bad_input_ixs | nan_ixs
    truths = truths[~bad_ixs]
    preds = preds[~bad_ixs]
    assert not np.isnan(preds).any()

    if exclude_stable:
        # ignore stable systems (i.e., where ground_truth is 10^9)
        stable_ixs = truths == 9
        truths = truths[~stable_ixs]
        preds = preds[~stable_ixs]

    if clip:
        preds = np.clip(preds, 4, 9)
        truths = np.clip(truths, 4, 9)

    return truths, preds


def get_truths(Ngrid, clip=True):
    """Load ground truths and input cache once, returning filtered truths and the bad_ixs mask."""
    truth_path = get_results_path(Ngrid, ground_truth=True)
    if not os.path.exists(truth_path):
        truth_path = 'figures/' + truth_path

    truths_raw = load_pickle(truth_path)
    truths = np.array([d['ground_truth'] if d is not None else np.nan for d in truths_raw])
    truths = np.log10(truths)

    input_cache = load_input_cache(Ngrid)
    bad_input_ixs = np.array([inp is None for inp in input_cache])
    nan_ixs = np.isnan(truths)
    bad_ixs = bad_input_ixs | nan_ixs

    truths = truths[~bad_ixs]
    if clip:
        truths = np.clip(truths, 4, 9)
    return truths, bad_ixs


def load_full_preds(
    Ngrid,
    bad_ixs,
    version=None,
    pysr_version=None,
    pysr_model_selection=None,
    petit=False,
    pure_sr=False,
):
    """Load cached grid predictions as [N, 2] = (mean, std), using std=1 if missing."""
    results_path = get_results_path(
        Ngrid,
        version,
        pysr_version=pysr_version,
        pysr_model_selection=pysr_model_selection,
        use_petit=petit,
        pure_sr=pure_sr,
    )
    if not os.path.exists(results_path):
        results_path = 'figures/' + results_path

    preds_raw = load_pickle(results_path)
    full_preds = []
    for d in preds_raw:
        if d is None:
            full_preds.append([np.nan, np.nan])
            continue
        std = d.get('std', 1.0)
        # assert std >= 0
        full_preds.append([d['mean'], std])

    full_preds = np.array(full_preds)
    full_preds = full_preds[~bad_ixs]
    assert not np.isnan(full_preds).any()
    return full_preds


def load_preds(Ngrid, bad_ixs, version=None, pysr_version=None, pysr_model_selection=None, petit=False, pure_sr=False, clip=True):
    """Load predictions for a single model, filtering by pre-computed bad_ixs."""
    results_path = get_results_path(Ngrid, version, pysr_version=pysr_version, pysr_model_selection=pysr_model_selection, use_petit=petit, pure_sr=pure_sr)
    if not os.path.exists(results_path):
        results_path = 'figures/' + results_path

    preds_raw = load_pickle(results_path)
    preds = np.array([d['mean'] if d is not None else np.nan for d in preds_raw])
    preds = preds[~bad_ixs]
    assert not np.isnan(preds).any()

    if clip:
        preds = np.clip(preds, 4, 9)
    return preds


def safe_log_erf(x):
    base_mask = x < -1
    value_giving_zero = torch.zeros_like(x, device=x.device)
    x_under = torch.where(base_mask, x, value_giving_zero)
    x_over = torch.where(~base_mask, x, value_giving_zero)

    f_under = lambda x: (
         0.485660082730562*x + 0.643278438654541*torch.exp(x) +
         0.00200084619923262*x**3 - 0.643250926022749 - 0.955350621183745*x**2
    )
    f_over = lambda x: torch.log(1.0+torch.erf(x))

    return f_under(x_under) + f_over(x_over)


def lossfnc(testy, y, fixed_std=1.0):
    """Legacy eval loss used by evaluation.py, adapted to one truth per grid cell."""
    mu = testy[:, [0]]
    std = testy[:, [1]]
    # Match evaluation.py: reported LL uses a fixed uncertainty scale.
    std[...] = fixed_std

    var = std**2
    t_greater_9 = y >= 9

    regression_loss = -(y - mu)**2/(2*var)
    regression_loss += -torch.log(std)
    regression_loss += -safe_log_erf((mu - 4) / (torch.sqrt(2 * var)))

    classifier_loss = safe_log_erf((mu - 9) / torch.sqrt(2 * var))

    safe_regression_loss = torch.where(
        ~torch.isfinite(regression_loss),
        -torch.ones_like(regression_loss) * 100,
        regression_loss,
    )
    safe_classifier_loss = torch.where(
        ~torch.isfinite(classifier_loss),
        -torch.ones_like(classifier_loss) * 100,
        classifier_loss,
    )

    total_loss = (
        safe_regression_loss * (~t_greater_9) +
        safe_classifier_loss * t_greater_9
    )

    total_regression = -(safe_regression_loss * (~t_greater_9)).sum().item() / len(y)
    total_classifier = -(safe_classifier_loss * t_greater_9).sum().item() / len(y)
    total_loss = -(total_loss.sum().item()) / len(y)
    return total_loss, total_regression, total_classifier


FIXED_FPR_TARGETS = (0.05, 0.10, 0.20)


def _safe_rate(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def _roc_auc_or_nan(labels, scores):
    return roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else np.nan


def _fnr_at_fixed_fpr(labels, scores, target_fpr):
    if len(np.unique(labels)) < 2:
        return np.nan
    fprs, tprs, _ = roc_curve(labels, scores)
    unique_fprs = np.unique(fprs)
    best_tprs = np.array([np.max(tprs[fprs == fpr]) for fpr in unique_fprs])
    target_tpr = np.interp(target_fpr, unique_fprs, best_tprs)
    return 1.0 - target_tpr


def _binary_metrics(truths, preds):
    """Classification metrics with positive = unstable (prediction/truth < 9)."""
    pred_stable = preds >= 9
    pred_unstable = ~pred_stable
    true_stable = truths >= 9
    true_unstable = ~true_stable

    tp = np.sum(pred_unstable & true_unstable)
    tn = np.sum(pred_stable & true_stable)
    fp = np.sum(pred_unstable & true_stable)
    fn = np.sum(pred_stable & true_unstable)
    n_stable = np.sum(true_stable)
    n_unstable = np.sum(true_unstable)

    stable_only_acc = _safe_rate(tn, n_stable)
    unstable_recall = _safe_rate(tp, n_unstable)
    fpr = _safe_rate(fp, n_stable)
    fnr = _safe_rate(fn, n_unstable)
    precision_unstable = _safe_rate(tp, tp + fp)
    mcc_denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    unstable_score = -preds

    out = dict(
        acc=np.mean(pred_stable == true_stable),
        stable_only_acc=stable_only_acc,
        balanced_acc=0.5 * (stable_only_acc + unstable_recall),
        roc_auc=_roc_auc_or_nan(true_unstable, unstable_score),
        fpr=fpr,
        fnr=fnr,
        precision_unstable=precision_unstable,
        mcc=((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom else 0.0,
    )
    out['auc'] = out['roc_auc']
    for target_fpr in FIXED_FPR_TARGETS:
        key = f'fnr_at_fpr_{int(round(target_fpr * 100)):03d}'
        out[key] = _fnr_at_fixed_fpr(true_unstable, unstable_score, target_fpr)
    return out


def calculate_metrics(truths, preds, full_preds=None, truths_for_loss=None, fixed_std=1.0):
    """Calculate metrics given pre-loaded truths/preds, optionally including eval loss."""
    stable_ixs = truths >= 9
    unstable_truths, unstable_preds = truths[~stable_ixs], preds[~stable_ixs]
    rmse = np.average(np.square(truths - preds))**0.5
    unstable_rmse = np.average(np.square(unstable_truths - unstable_preds))**0.5
    out = dict(rmse=rmse, unstable_rmse=unstable_rmse, bias=np.mean(preds - truths))
    out.update(_binary_metrics(truths, preds))

    if full_preds is not None:
        if truths_for_loss is None:
            truths_for_loss = truths
        preds_t = torch.from_numpy(full_preds).float()
        truths_t = torch.from_numpy(truths_for_loss[:, None]).float()
        loss, reg_loss, cls_loss = lossfnc(preds_t, truths_t, fixed_std=fixed_std)
        out.update(loss=loss, reg_loss=reg_loss, cls_loss=cls_loss)

    return out


OFFICIAL_METRIC_COLUMNS = [
    ('rmse', 'RMSE'),
    ('unstable_rmse', 'UnstableRMSE'),
    ('acc', 'Acc'),
    ('balanced_acc', 'BalancedAcc'),
    ('roc_auc', 'ROC-AUC'),
    ('fpr', 'FPR'),
    ('fnr', 'FNR'),
    ('fnr_at_fpr_005', 'FNR@FPR=0.05'),
    ('fnr_at_fpr_010', 'FNR@FPR=0.10'),
    ('fnr_at_fpr_020', 'FNR@FPR=0.20'),
    ('precision_unstable', 'PrecisionUnstable'),
    ('mcc', 'MCC'),
    ('loss', 'Loss'),
]


def _format_metric(value):
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return 'nan'
    return f'{value:.3f}'


def _format_official_latex_row(latex_name, metrics, ll):
    """Format columns as RMSE/Acc/LL/ROC/FPR/FNR/bias."""
    return (
        f'{latex_name} & {metrics["unstable_rmse"]:.2f} & {metrics["acc"]:.2f} & '
        f'{ll} & {metrics["roc_auc"]:.2f} & {metrics["fpr"]:.2f} & '
        f'{metrics["fnr"]:.2f} & {metrics["bias"]:.2f} \\\\'
    )


def _print_metrics_table(table_rows):
    headers = ['method'] + [label for _, label in OFFICIAL_METRIC_COLUMNS]
    rows = [[name.strip()] + [_format_metric(metrics[key]) for key, _ in OFFICIAL_METRIC_COLUMNS]
            for name, metrics in table_rows]
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

    def format_row(row):
        return '\t'.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    print(format_row(headers))
    for row in rows:
        print(format_row(row))


def official_metrics(args):
    versions = load_json(args.version_json)

    truths, bad_ixs = get_truths(args.Ngrid)
    truths_for_loss, _ = get_truths(args.Ngrid, clip=False)

    models = [
        ('nn', 'Neural network', dict(version=versions['nn_version'])),
        ('distilled_eqs', 'Ours', dict(version=versions['nn_version'], pysr_version=versions['pysr_version'], pysr_model_selection=versions['pysr_model_selection'])),
        ('petit', 'Petit+ 2020', dict(petit=True)),
        ('pure_sr', 'Pure SR', dict(pure_sr=True, pysr_version=versions['pure_sr_version'], pysr_model_selection=versions['pure_sr_model_selection'])),
        ('pure_sr2', 'Pure SR 2', dict(version=28114, pysr_version=versions['pure_sr2_version'], pysr_model_selection=versions['pure_sr2_model_selection'])),
    ]

    results = {}
    table_rows = []
    for name, _, kwargs in models:
        preds = load_preds(args.Ngrid, bad_ixs, **kwargs)
        full_preds = load_full_preds(args.Ngrid, bad_ixs, **kwargs)
        fixed_std = 1.47 if name in {'distilled_eqs', 'pure_sr', 'pure_sr2'} else 1.0
        vals = calculate_metrics(
            truths,
            preds,
            full_preds=full_preds,
            truths_for_loss=truths_for_loss,
            fixed_std=fixed_std,
        )
        results[name] = vals
        table_rows.append((name, vals))

    _print_metrics_table(table_rows)
    print('LaTeX rows')
    print('% columns: RMSE & Acc & LL & ROC & FPR & FNR & bias')
    for name, latex_name, _ in models:
        metrics = results[name]
        loss = '--' if name == 'petit' else f'{-metrics["loss"]:.2f}'
        print(_format_official_latex_row(latex_name, metrics, loss))

    return results

def get_citation():
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.add(m=1.) # Star
    sim.add(m=1e-4, P=1, theta='uniform')
    sim.add(m=1e-4, P=1/0.75, theta='uniform')
    sim.add(m=1e-4, P=1/0.75/0.75, theta='uniform')
    sim.cite()


def load_and_calculate_metrics(args):
    truths, preds = get_truths_and_preds(args.Ngrid, args.version, args.pysr_version, args.pysr_model_selection, args.petit, args.pure_sr, clip=True, exclude_stable=args.exclude_stable)
    truths_for_loss, bad_ixs = get_truths(args.Ngrid, clip=False)
    full_preds = load_full_preds(
        args.Ngrid,
        bad_ixs,
        version=args.version,
        pysr_version=args.pysr_version,
        pysr_model_selection=args.pysr_model_selection,
        petit=args.petit,
        pure_sr=args.pure_sr,
    )
    if args.exclude_stable:
        stable_ixs = truths_for_loss >= 9
        truths_for_loss = truths_for_loss[~stable_ixs]
        full_preds = full_preds[~stable_ixs]
    result = calculate_metrics(truths, preds, full_preds=full_preds, truths_for_loss=truths_for_loss)
    print(
        f'RMSE={result["rmse"]:.3f} UnstableRMSE={result["unstable_rmse"]:.3f} '
        f'Acc={result["acc"]:.3f} ROC-AUC={result["roc_auc"]:.3f} '
        f'FPR={result["fpr"]:.3f} FNR={result["fnr"]:.3f} Loss={result["loss"]:.3f}'
    )
    print(result)


def get_args():
    print(get_script_execution_command())
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ngrid', '-n', type=int, default=300)
    parser.add_argument('--version', '-v', type=int, default=None)

    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--compute', action='store_true')
    parser.add_argument('--collate', action='store_true')
    parser.add_argument('--petit', action='store_true')
    parser.add_argument('--megno', action='store_true')
    parser.add_argument('--ground_truth', action='store_true')
    parser.add_argument('--pure_sr', action='store_true')
    parser.add_argument('--create_input_cache', action='store_true')
    parser.add_argument('--png', action='store_true')
    parser.add_argument('--seed', type=int, default=0, help='seed for loading NN model')

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
    parser.add_argument('--special', type=str, default=None, choices=['official_plots', '4way', '4way_std', 'main', 'pure_sr', '4way_pysr', 'f1_features', 'exprs', 'metrics', 'official_metrics', 'rmse_diff', 'gt_diff', 'four_way_rmse', 'selective_eq', 'selective_eq_rmse'])
    parser.add_argument('--minimal_plot', action='store_true')
    parser.add_argument('--version_json', type=str, default='../official_versions.json', help='Path to the JSON file containing model versions')
    parser.add_argument('--exclude_stable', action='store_true', help='exclude stable systems from RMSE calculation')
    parser.add_argument('--resonance_guides', action='store_true', help='draw dashed P1/P2=2/3 and P2/P3=2/3 guide lines on supported plots')
    parser.add_argument(
        '--highlight_resonant_points',
        action='store_true',
        help='overlay the resonant_figure grid as green points/locus in period-ratio coordinates',
    )
    parser.add_argument('--resonant_overlay_mode', choices=['old_fixed_outer', 'inner32_phase_slice'], default='old_fixed_outer')
    parser.add_argument('--resonant_p2_min', type=float, default=1.47)
    parser.add_argument('--resonant_p2_max', type=float, default=1.56)
    parser.add_argument('--resonant_y_min', type=float, default=0.55)
    parser.add_argument('--resonant_y_max', type=float, default=0.76)
    parser.add_argument('--resonant_inner_period', type=float, default=1.0)
    parser.add_argument('--resonant_outer_period', type=float, default=2.04)

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

    if args.max_t:
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

    if should_compute_official_results(args):
        compute_official_results(args)
        args.compute = False

    if args.compute:
        if args.pysr_path and not args.pure_sr:  # pure sr uses normal compute_results
            compute_pysr_f2_results(args)
        else:
            results = compute_results(args)

    if args.collate:
        collate_parallel_results(args)

    if args.plot:
        if args.pysr_path and not args.pure_sr:
            plot_results_pysr_f2(args)
        else:
            plot_results(args)

    if args.special:
        if args.special == 'official_plots':
            plot_main_figure(args)
            plot_pure_sr_comparison(args)
            plot_f1_features2(args)
            plot_4way_std(args)
        if args.special == '4way':
            plot_4way_comparison(args)
        if args.special == '4way_std':
            plot_4way_std(args)
        if args.special == 'main':
            plot_main_figure(args)
        elif args.special == 'pure_sr':
            plot_pure_sr_comparison(args)
        elif args.special == '4way_pysr':
            plot_4way_pysr_comparison(args)
        elif args.special == 'f1_features':
            plot_f1_features2(args)
        elif args.special == 'exprs':
            plot_exprs(args)
        elif args.special == 'metrics':
            load_and_calculate_metrics(args)
        elif args.special == 'official_metrics':
            metrics = official_metrics(args)
            save_pickle(metrics, '../pickles/period_ratio_official_metrics.pkl')
        elif args.special == 'four_way_rmse':
            plot_4way_rmse(args)
        elif args.special == 'selective_eq':
            plot_selective_eq_6way(args)
        elif args.special == 'selective_eq_rmse':
            plot_selective_eq_6way_rmse(args)

    end = time.time()
    formatted_time = time.strftime('%H:%M:%S', time.gmtime(end - start))
    print(f'Done (time taken: {formatted_time})')
    # print(get_citation())
