import pysr
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# import pysr  # just to avoid errors if its imported after pytorch

import sys
sys.path.append('../')

import rebound
import os
import numpy as np
import matplotlib.pyplot as plt

import spock
import utils2
from utils2 import assert_equal
import argparse

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)


def get_args():
    print(utils2.get_script_execution_command())
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ngrid', '-n', type=int)
    parser.add_argument('--metric', '-m', type=str, choices=['megno', 'f1', 'std', 'mean'], default='mean')
    parser.add_argument('--action', '-a', choices=['compute', 'collate', 'plot'], type=str, default='compute')
    parser.add_argument('--ix', type=int, default=None)
    parser.add_argument('--total', type=int, default=None)
    args = parser.parse_args()
    return args


def load_model():
    version = 43139 # val loss 1.603
    model = spock.NonSwagFeatureRegressor(version=version)
    return model


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


def get_megno_prediction(sim):
    sim = sim.copy()
    sim.dt = 0.05
    sim.init_megno()
    sim.exit_max_distance = 20.
    try:
        sim.integrate(1e4)
        megno = sim.calculate_megno()
        return megno
    except rebound.Escape:
        return 10. # At least one particle got ejected, returning large MEGNO.


def get_model_prediction(sim, model, metric):
    sim = sim.copy()
    sim.dt = 0.05
    sim.init_megno()
    sim.exit_max_distance = 20.
    try:
        out_dict = model.predict(sim)
        if metric == 'mean':
            out = out_dict['mean']
        elif metric == 'std':
            out = out_dict['std']
        elif metric == 'f1':
            out = out_dict['summary_stats']

        assert_equal(out.shape[0], 1)
        out = out[0].detach().cpu().numpy()
    except rebound.Escape:
        out = np.NaN

    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        print('returning NaN for now')

        out = np.NaN

    return out


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


def compute_results_for_parameters(parameters, metric):
    simulations = [get_simulation(par) for par in parameters]

    if metric == 'megno':
        f = get_megno_prediction
    else:
        model = load_model()
        f = lambda sim: get_model_prediction(sim, model, metric)

    results = [f(sim) for sim in simulations]
    if metric == 'f1':
        # convert NaN's to arrays of NaNs of the correct size
        entry = next((x for x in results if isinstance(x, np.ndarray)), None)
        length = len(entry)
        results = [np.full(length, np.NaN) if not isinstance(x, np.ndarray) else x for x in results]

    return results


def get_list_chunk(lst, ix, total):
    '''
    split list into total chunks and return the ix-th chunk
    example: get_list_chunk([0,1,2,3,4,5,6,7,8,9], 0, 3) -> [0,1,2]
             get_list_chunk([0,1,2,3,4,5,6,7,8,9], 1, 3) -> [2,3,4]
             get_list_chunk([0,1,2,3,4,5,6,7,8,9], 2, 3) -> [5,6,7,8,9]
    '''
    chunk_size = len(lst) // total
    start = ix * chunk_size
    end = start + chunk_size if ix < total - 1 else len(lst)
    return lst[start:end]


def compute_results(Ngrid, metric, parallel_ix=None, parallel_total=None):
    P12s, P23s = get_period_ratios(Ngrid)
    parameters = get_parameters(P12s, P23s)
    if parallel_ix is not None:
        parameters = get_list_chunk(parameters, parallel_ix, parallel_total)

    results = compute_results_for_parameters(parameters, metric)
    # save the results
    path = get_results_path(Ngrid, metric, parallel_ix, parallel_total)
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results = np.array(results)
    np.save(path, results)
    print('saved results to', path)
    return results


def get_results_path(Ngrid, metric, parallel_ix=None, parallel_total=None):
    model = 'megno' if metric == 'megno' else 'bnn'
    path = f'period_results/results_ngrid={Ngrid}_{model}'

    if metric == 'std':
        path += '_std'
    elif metric == 'f1':
        path += '_f1'

    if parallel_ix is not None:
        path += f'/{parallel_ix}-{parallel_total}'

    path += '.npy'
    return path


def load_results(path):
    return np.load(path)


def collate_parallel_results(Ngrid, metric, parallel_total):
    '''load the parallel results and save as one big list'''
    results = []
    for ix in range(parallel_total):
        path = get_results_path(Ngrid, metric, ix, parallel_total)
        try:
            sub_results = load_results(path)
        except FileNotFoundError:
            sub_results = None
            print('Missing results for ix', ix)
        results.append(sub_results)

    # replace None values with NaN arrays of the same length as the first non-None subresult
    length = None
    for sub_results in results:
        if sub_results is not None:
            length = len(sub_results)
            break

    if length is None:
        print('No results found')
        return

    results = [sub_result if sub_result is not None
               else np.array([np.NaN] * length)
               for sub_result in results]

    # concatenate into one big numpy array
    results = np.concatenate(results)
    np.save(get_results_path(Ngrid, metric), results)
    print('saved results to', get_results_path(Ngrid, metric))


def plot_results(results, Ngrid, metric):
    P12s, P23s = get_period_ratios(Ngrid)

    fig, ax = plt.subplots(figsize=(8,6))

    X,Y,Z = get_centered_grid(P12s, P23s, results)

    if metric == 'megno':
        Zfilt = Z
        Zfilt[Zfilt < 2] = 2.01

        cmap = plt.cm.seismic.copy()
        cmap.set_bad(color='yellow')
        im = ax.pcolormesh(X, Y, np.log10(Zfilt-2), vmin=-4, vmax=4, cmap=cmap)
        label = "log(MEGNO-2) (red = chaotic)"

    elif metric == 'std':
        cmap = plt.cm.inferno.copy().reversed()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, Z, vmin=0, vmax=6, cmap=cmap)
        label = "std(log(T_unstable))"
    elif metric == 'mean':
        cmap = plt.cm.inferno.copy().reversed()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, Z, vmin=4, vmax=12, cmap=cmap)
        label = "log(T_unstable)"
    else:
        raise NotImplementedError()

    cb = plt.colorbar(im, ax=ax)
    cb.set_label(label)
    ax.set_xlabel("P1/P2")
    ax.set_ylabel("P2/P3")

    s = 'megno' if metric == 'megno' else 'bnn'
    s += f'_ngrid={Ngrid}'
    if metric == 'std':
        s += '_std'
    elif metric == 'f1':
        s += '_f1'
    s = 'period_results/period_ratio_' + s + '.png'
    plt.savefig(s, dpi=800)
    print('saved figure to', s)


if __name__ == '__main__':
    args = get_args()
    Ngrid = args.Ngrid
    metric = args.metric
    parallel_ix = args.ix
    parallel_total = args.total

    if args.action == 'compute':
        results = compute_results(Ngrid, metric, parallel_ix, parallel_total)
    elif args.action == 'collate':
        collate_parallel_results(Ngrid, metric, parallel_total)
    elif args.action == 'plot':
        results = load_results(get_results_path(Ngrid, metric))
        plot_results(results, Ngrid, metric)

    print('Done')
