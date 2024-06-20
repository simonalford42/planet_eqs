
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
sys.path.append('../')

import pytorch_lightning as pl

import rebound
import numpy as np
import matplotlib.pyplot as plt
import utils2
from spock import FeatureRegressor

from multiprocess import Pool

def get_args():
    print(utils2.get_script_execution_command())
    use_model = bool(int(sys.argv[1]))
    Ngrid = int(sys.argv[2])
    print('Using model:', use_model)
    print('Ngrid:', Ngrid)
    return Ngrid, use_model


def load_model():
    version = 4157
    model = FeatureRegressor(
        cuda=True,
        filebase='../' + utils2.ckpt_path(version, glob=True) +  '*output.pkl'
        # filebase='*' + 'v30' + '*output.pkl'
        #'long_zero_megno_with_angles_power_v14_*_output.pkl'
    )
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

def get_model_prediction(sim, model):
    sim = sim.copy()
    sim.dt = 0.05
    sim.init_megno()
    sim.exit_max_distance = 20.
    try:
        out = model.predict(sim)
    except rebound.Escape as error:
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


def compute_results(Ngrid=80, use_model=False):
    if use_model:
        model = load_model()

    with Pool() as pool:
        P12s, P23s = get_period_ratios(Ngrid)
        parameters = []
        for P12 in P12s:
            for P23 in P23s:
                parameters.append((P12,P23))

        simulations = [get_simulation(par) for par in parameters]
        if use_model:
            f = lambda sim: get_model_prediction(sim, model)
        else:
            f = get_megno_prediction

        # results = pool.map(f, simulations)

        # unparrallelized version for debugging
        results = []
        for sim in simulations:
            with utils2.Timing('simulation'):
                out = f(sim)
                print(out)
            results.append(out)

        # save the results
        np.save(get_results_path(Ngrid, use_model), np.array(results))
        return results


def get_results_path(Ngrid=80, use_model=False):
    return f'period_ratio_figure_results_ngrid={Ngrid}_model={use_model}.npy'


def load_results(path):
    return np.load(path)


def plot_results(results, Ngrid=80, use_model=False):
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
    s = 'period_ratio_figure' + s + '.png'
    plt.savefig(s, dpi=200)
    print('saved figure to', s)


if __name__ == '__main__':
    Ngrid, use_model = get_args()
    results = compute_results(Ngrid, use_model)
    # results = load_results(get_results_path(Ngrid, use_model))
    plot_results(results, Ngrid, use_model)
    print('done')



