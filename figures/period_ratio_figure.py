# import pysr
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# import pysr  # just to avoid errors if its imported after pytorch

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
import argparse

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)


'''
Example commands:

# compute and plot for BNN predictions
python period_ratio_figure.py --Ngrid 4 --version 24880 --compute
python period_ratio_figure.py --Ngrid 4 --version 24880 --plot

# compute and plot for pysr f2 (need to have computed for BNN before running this)
python period_ratio_figure.py --Ngrid 4 --version 24880 --pysr_path 'sr_results/33060.pkl' --compute
python period_ratio_figure.py --Ngrid 4 --version 24880 --pysr_path 'sr_results/33060.pkl' --plot

# compute using 4 parallel jobs
python period_ratio_figure.py --Ngrid 400 --version 24880 --compute --parallel_ix 0 --parallel_total 4
python period_ratio_figure.py --Ngrid 400 --version 24880 --compute --parallel_ix 1 --parallel_total 4
python period_ratio_figure.py --Ngrid 400 --version 24880 --compute --parallel_ix 2 --parallel_total 4
python period_ratio_figure.py --Ngrid 400 --version 24880 --compute --parallel_ix 3 --parallel_total 4

# collate the parallel results and save
python period_ratio_figure.py --Ngrid 400 --version 24880 --collate --parallel_total 4

# copy needed files to local so we can plot
scopy bnn_chaos_model/figures/period_results/v=43139_ngrid=6_pysr_f2/ ~/code/bnn_chaos_model/figures/period_results/
scopy bnn_chaos_model/figures/period_results/v=43139_ngrid=6.pkl ~/code/bnn_chaos_model/figures/period_results/
'''

def get_args():
    print(utils2.get_script_execution_command())
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ngrid', '-n', type=int, required=True)
    parser.add_argument('--version', '-v', type=int, default=43139)

    parser.add_argument('--plot', '-p', action='store_true')
    parser.add_argument('--compute', action='store_true')
    parser.add_argument('--collate', action='store_true')

    parser.add_argument('--pysr_path', type=str, default=None) # PySR model to load and replace f2 with, e.g. 'sr_results/33060.pkl'
    parser.add_argument('--pysr_model_selection', type=str, default=None, help='"best", "accuracy", "score", or an integer of the pysr equation complexity. If not provided, will do all complexities. If plotting, has to be an integer complexity')

    # parallel processing args
    # ix should be in [0, total)
    parser.add_argument('--parallel_ix', '-i', type=int, default=None)
    parser.add_argument('--parallel_total', '-t', type=int, default=None)

    args = parser.parse_args()
    return args


def load_model(version):
    return spock.NonSwagFeatureRegressor(version=version)


def simulation(par):
    P12, P23 = par # unpack parameters
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.add(m=1.) # Star
    sim.add(m=1e-4, P=1, theta='uniform')
    sim.add(m=1e-4, P=1/P12, theta='uniform')
    sim.add(m=1e-4, P=1/P12/P23, theta='uniform')
    sim.move_to_com()

    sim.dt = 0.05
    sim.init_megno()
    sim.exit_max_distance = 20.
    try:
        sim.integrate(1e4)
        megno = sim.megno()
        return megno
    except rebound.Escape:
        return 10. # At least one particle got ejected, returning large MEGNO.


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


def get_model_prediction(sim, model):
    sim = sim.copy()
    sim.dt = 0.05
    sim.init_megno()
    sim.exit_max_distance = 20.
    try:
        out_dict = model.predict(sim)
        return {
            'mean': out_dict['mean'][0,0].detach().cpu().numpy(),
            'std': out_dict['std'][0,0].detach().cpu().numpy(),
            'f1': out_dict['summary_stats'][0].detach().cpu().numpy()
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


def compute_results_for_parameters(parameters, model):
    simulations = [get_simulation(par) for par in parameters]

    results = [get_model_prediction(sim, model) for sim in simulations]
    return results


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


def compute_results(args):
    model = load_model(args.version)
    P12s, P23s = get_period_ratios(args.Ngrid)
    parameters = get_parameters(P12s, P23s)
    if args.parallel_ix is not None:
        parameters = get_list_chunk(parameters, args.parallel_ix, args.parallel_total)

    results = compute_results_for_parameters(parameters, model)

    # save the results
    path = get_results_path(args.Ngrid, args.version, args.parallel_ix, args.parallel_total, args.pysr_model_selection)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(results, f)

    print('Saved results to', path)
    return results


def get_results_path(Ngrid, version, parallel_ix=None, parallel_total=None, pysr_model_selection=None):
    path = f'period_results/v={version}_ngrid={Ngrid}'

    if pysr_model_selection is not None:
        path += f'_pysr_f2/{pysr_model_selection}'

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
    for ix in range(args.parallel_total):
        path = get_results_path(args.Ngrid, args.version, ix, args.parallel_total)
        try:
            sub_results = load_results(path)
        except FileNotFoundError:
            sub_results = None
            print('Missing results for ix', ix)
        results.append(sub_results)

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
    path = get_results_path(args.Ngrid, args.version)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print('Saved results to', path)


def plot_results(args, metric=None):
    if metric is None:
        for metric in ['mean', 'std']:
            plot_results(args, metric)
        return

    results = load_results(get_results_path(args.Ngrid, args.version, args.pysr_model_selection))
    P12s, P23s = get_period_ratios(args.Ngrid)

    fig, ax = plt.subplots(figsize=(8,6))

    # get the results for the specific metric
    if args.plot == 'mean':
        results = [d['mean'] if d is not None else np.NaN for d in results]
    elif args.plot == 'std':
        results = [d['std'] if d is not None else np.NaN for d in results]

    results = np.array(results)
    X,Y,Z = get_centered_grid(P12s, P23s, results)

    if metric == 'std':
        cmap = plt.cm.inferno.copy().reversed()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, Z, vmin=0, vmax=6, cmap=cmap)
        label = "std(log(T_unstable))"
    elif metric == 'mean':
        cmap = plt.cm.inferno.copy().reversed()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, Z, vmin=4, vmax=12, cmap=cmap)
        label = "log(T_unstable)"

    cb = plt.colorbar(im, ax=ax)
    cb.set_label(label)
    ax.set_xlabel("P1/P2")
    ax.set_ylabel("P2/P3")

    path = get_results_path(args.Ngrid, args.version, args.pysr_model_selection)
    # get rid of .pkl
    path = path[:-4]
    path += '_plot'

    if metric == 'std':
        path += '_std'

    if args.pysr_model_selection is not None:
        path += f'_pysr_f2/{args.pysr_model_selection}'

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
    results = load_results(get_results_path(args.Ngrid, args.version))

    if args.pysr_model_selection is None:
        reg = pickle.load(open(args.pysr_path, 'rb'))
        model_selections = list(reg.equations_[0]['complexity'])
    else:
        model_selections = [args.pysr_model_selection]

    f1_results = [d['f1'] if d is not None else None for d in results]
    good_ixs = np.array([i for i in range(len(f1_results)) if f1_results[i] is not None])

    batch = np.array([d for d in f1_results if d is not None])
    batch = torch.tensor(batch).float().cuda()

    for model_selection in model_selections:
        regress_nn = modules.PySRNet(args.pysr_path, model_selection).cuda()
        pred = regress_nn(batch).detach().cpu().numpy()
        results = np.full((len(f1_results), pred.shape[1]), np.NaN)
        results[good_ixs] = pred

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
        path = get_results_path(args.Ngrid, args.version, pysr_model_selection=model_selection)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(results, f)

        print('Saved results to', path)

    return results


def plot_results_pysr_f2(args):
    # get the model selections by grepping for the files
    path = get_results_path(args.Ngrid, args.version, pysr_model_selection='*')
    files = os.listdir(os.path.dirname(path))

    # filter to those of form f'{i}.pkl'
    files = [file for file in files if file.endswith('.pkl')]

    # go from f'{model_selection}.pkl' to model_selection
    model_selections = sorted([int(f.split('.')[0]) for f in files])

    # if model selection is provided, filter to those
    if model_selection is not None:
        files = [file for file in files if file.startswith(model_selection)]

    # go from f'{model_selection}.pkl' to model_selection
    model_selections = [int(f.split('.')[0]) for f in files]
    model_selections = sorted(model_selections)

    original_model_selection = args.pysr_model_selection
    for model_selection in model_selections:
        args.pysr_model_selection = model_selection
        plot_results(args)
    args.pysr_model_selection = original_model_selection


if __name__ == '__main__':
    args = get_args()

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
        else:
            plot_results(args)

    print('Done')
