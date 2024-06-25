import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
sys.path.append('../')

import pytorch_lightning as pl

import rebound
import numpy as np
import matplotlib.pyplot as plt

import utils2
from spock import FeatureRegressor, NonSwagFeatureRegressor

from multiprocess import Pool
import argparse


def get_args():
    print(utils2.get_script_execution_command())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_megno', action='store_true')
    parser.add_argument('--std', action='store_true')
    parser.add_argument('--Ngrid', type=int, default=80)
    parser.add_argument('--ix', type=int, default=None)
    parser.add_argument('--total', type=int, default=None)
    args = parser.parse_args()
    return args.Ngrid, not args.use_megno, args.ix, args.total, args.std


def load_model():
    # version = 4157
    # model = FeatureRegressor(
    #     cuda=True,
    #     filebase='../' + utils2.ckpt_path(version, glob=True) +  '*output.pkl'
    #     # filebase='*' + 'v30' + '*output.pkl'
    #     #'long_zero_megno_with_angles_power_v14_*_output.pkl'
    # )
    version = 43139 # val loss 1.603
    model = NonSwagFeatureRegressor(version=43139)
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

def get_model_prediction(sim, model, return_std=False):
    sim = sim.copy()
    sim.dt = 0.05
    sim.init_megno()
    sim.exit_max_distance = 20.
    try:
        out = model.predict(sim, return_std=return_std)
    except:
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


def compute_results_for_parameters(parameters, use_model=False, return_std=False):
    if use_model:
        model = load_model()

    simulations = [get_simulation(par) for par in parameters]
    if use_model:
        f = lambda sim: get_model_prediction(sim, model, return_std)
    else:
        f = get_megno_prediction

    results = [f(sim) for sim in simulations]
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


def compute_results(Ngrid=80, use_model=False, parallel_ix=None, parallel_total=None, return_std=False):
    P12s, P23s = get_period_ratios(Ngrid)
    parameters = get_parameters(P12s, P23s)
    if parallel_ix is not None:
        parameters = get_list_chunk(parameters, parallel_ix, parallel_total)

    results = compute_results_for_parameters(parameters, use_model, return_std)
    # save the results
    path = get_results_path(Ngrid, use_model, parallel_ix, parallel_total, return_std)
    np.save(path, np.array(results))
    print('saved results to', path)
    return results


def get_results_path(Ngrid=80, use_model=False, parallel_ix=None, parallel_total=None, return_std=False):
    model = 'bnn' if use_model else 'megno'
    path = f'period_results/results_ngrid={Ngrid}_{model}'
    if return_std:
        path += '_std'
    if parallel_ix is not None:
        path += f'_{parallel_ix}-{parallel_total}'
    path += '.npy'
    return path


def load_results(path):
    return np.load(path)


def collate_parallel_results(Ngrid, use_model, parallel_total, return_std=False):
    '''load the parallel results and save as one big list'''
    results = []
    for ix in range(parallel_total):
        path = get_results_path(Ngrid, use_model, ix, parallel_total, return_std)
        try:
            sub_results = load_results(path)
        except FileNotFoundError:
            sub_results = None
            print('missing sub_results for ix', ix)
        results.append(sub_results)

    # replace None values with NaN arrays of the same length as the first non-None subresult
    length = None
    for sub_results in results:
        if sub_results is not None:
            length = len(sub_results)
            break

    results = [sub_result if sub_result is not None
               else np.array([np.NaN] * length)
               for sub_result in results]

    # concatenate into one big numpy array
    results = np.concatenate(results)
    np.save(get_results_path(Ngrid, use_model, return_std), results)
    print('saved results to', get_results_path(Ngrid, use_model, return_std))


def plot_results(results, Ngrid=80, use_model=False, return_std=False):
    P12s, P23s = get_period_ratios(Ngrid)

    fig, ax = plt.subplots(figsize=(8,6))

    X,Y,Z = get_centered_grid(P12s, P23s, results)

    if use_model:
        Zfilt = Z
        Zfilt[Zfilt == np.NaN] = 0
        im = ax.pcolormesh(X, Y, Zfilt, cmap='seismic')

    else:
        Zfilt = Z
        Zfilt[Zfilt <2] = 2.01
        im = ax.pcolormesh(X, Y, np.log10(Zfilt-2), vmin=-4, vmax=4, cmap='seismic')

    cb = plt.colorbar(im, ax=ax)
    if use_model:
        cb.set_label("log(T_unstable)")
    else:
        cb.set_label("log(MEGNO-2) (red = chaotic)")
    ax.set_xlabel("P1/P2")
    ax.set_ylabel("P2/P3")
    s = '_bnn' if use_model else '_megno'
    s += f'_ngrid={Ngrid}'
    if return_std:
        s += '_std'
    s = 'period_results/period_ratio_' + s + '.png'
    plt.savefig(s, dpi=200)
    print('saved figure to', s)


if __name__ == '__main__':
    Ngrid, use_model, parallel_ix, parallel_total, return_std = get_args()
    results = compute_results(Ngrid, use_model, parallel_ix, parallel_total, return_std)
    # collate_parallel_results(Ngrid, use_model, parallel_total, return_std)
    # results = load_results(get_results_path(Ngrid, use_model, return_std))
    # plot_results(results, Ngrid, use_model, return_std)
    print('done')
