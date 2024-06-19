
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


print(utils2.get_script_execution_command())
USE_MODEL = bool(sys.argv[1])
NGRID = int(sys.argv[2])
print('Using model:', USE_MODEL)
print('Ngrid:', NGRID)

version = 4157
model = FeatureRegressor(
    cuda=True,
    filebase='../' + utils2.ckpt_path(version, glob=True) +  '*output.pkl'
    # filebase='*' + 'v30' + '*output.pkl'
    #'long_zero_megno_with_angles_power_v14_*_output.pkl'
)


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

def get_model_prediction(sim):
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

from multiprocess import Pool

with Pool() as pool:
    # Ngrid = 80
    Ngrid = NGRID
    P12s = np.linspace(0.55, 0.76, Ngrid)
    P23s = np.linspace(0.55,0.76,Ngrid)
    parameters = []
    for P12 in P12s:
        for P23 in P23s:
            parameters.append((P12,P23))

    simulations = [get_simulation(par) for par in parameters]
    if USE_MODEL:
        f = get_model_prediction
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
    np.save(f'period_ratio_figure_results_ngrid={Ngrid}_model={USE_MODEL}.npy', np.array(results))


fig, ax = plt.subplots(figsize=(8,6))

X,Y,Z = get_centered_grid(P12s, P23s, results)
Zfilt = Z
Zfilt[Zfilt <2] = 2.01
im = ax.pcolormesh(X, Y, np.log10(Zfilt-2), vmin=-4, vmax=4, cmap='seismic')

cb = plt.colorbar(im, ax=ax)
cb.set_label("log(MEGNO-2) (red = chaotic)")
ax.set_xlabel("P1/P2")
ax.set_ylabel("P2/P3")
s = '_bnn' if USE_MODEL else '_megno'
s += f'_ngrid={Ngrid}'
plt.savefig("period_ratio_figure" + s + '.png', dpi=200)
print('done')
