#!/usr/bin/env python
# coding: utf-8
# %matplotlib inline

import sys
sys.path.append('../')

import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
import pickle
import spock_reg_model
import numpy as np
import rebound
import matplotlib.pyplot as plt
import torch
import sys
from multiprocessing import Pool
import pandas as pd
import time
from spock import NonSwagFeatureRegressor
import utils2
import argparse
import numpy as jnp
from time import time as ttime
from petit20_survival_time import Tsurv
from scipy.integrate import quad
from scipy.interpolate import interp1d
import einops
from oldsimsetup import init_sim_parameters
from collections import OrderedDict
import sys
sys.path.append('spock')
from tseries_feature_functions import get_extended_tseries
from pure_sr_evaluation import pure_sr_predict_fn
from modules import PureSRNet

from multiswag_5_planet_plot import make_plot, make_plot_separate, make_plot2

import warnings
warnings.filterwarnings("default", category=UserWarning)


try:
    plt.style.use('paper')
except:
    pass

spockoutfile = '../data/spockprobstesttrio.npz'
# version = int(sys.argv[1]) if len(sys.argv) > 1 else 24880
# Paper-ready is 5000:
# N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
# Paper-ready is 10000
# samples = int(sys.argv[3]) if len(sys.argv) > 3 else 100


def get_args():
    print(utils2.get_script_execution_command())
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=50)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--version', '-v', type=int, default=24880)

    parser.add_argument('--paper-ready', '-p', action='store_true')
    parser.add_argument('--turbo', action='store_true', help='skips sampling. No perf difference for equation models that are basically deterministic')

    parser.add_argument('--pysr_version', type=int, default=None)
    parser.add_argument('--pure_sr', action='store_true')
    parser.add_argument('--pysr_dir', type=str, default='../sr_results/')  # folder containing pysr results pkl
    parser.add_argument('--pysr_model_selection', type=str, default=None, help='"best", "accuracy", "score", or an integer of the pysr equation complexity. If not provided, will do all complexities. If plotting, has to be an integer complexity')

    parser.add_argument('--extrapolate', action='store_true', help='disables prior for >= 9')

    args = parser.parse_args()

    if args.paper_ready:
        args.N = 5000
        args.samples = 10000

    return args

args = get_args()
N = args.N
samples = args.samples
version = args.version

if args.pysr_version is not None:
    args.turbo = True

# try:
#     cleaned = pd.read_csv('cur_plot_dataset_1604437382.10866.csv')#'cur_plot_dataset_1604339344.2705607.csv')
#     # make_plot(cleaned, version)
#     make_plot(cleaned, args.version, t20=False)
#     exit(0)
# except FileNotFoundError:
#     ...

stride = 1
nsim_list = np.arange(0, 17500)
# Paper-ready is 5000:
# N = 50
# Paper-ready is 10000
# samples = 100
used_axes = np.linspace(0, 17500-1, N).astype(np.int32)#np.arange(17500//3, 17500, 1750//3)

nsim_list = nsim_list[used_axes]

# if args.pure_sr:
#     results = get_pure_sr_results(args.pysr_version)
#     model =
#     model = NonSwagFeatureRegressor(model=model)

model = spock_reg_model.load(args.version)
model = NonSwagFeatureRegressor(model=model)
bnn_model = None

if args.pysr_version:
    bnn_model = model

    if args.pure_sr:
        with open(f'../sr_results/{args.pysr_version}.pkl', 'rb') as f:
            reg = pickle.load(f)
        results = reg.equations_
        results.feature_names_in_ = reg.feature_names_in_
        pure_sr_fn = pure_sr_predict_fn(results, args.pysr_model_selection)
        pure_sr_model = PureSRNet(pure_sr_fn)
        bnn_model2 = spock_reg_model.load(24880)
        from modules import AddStdPredNN
        model = AddStdPredNN(pure_sr_model, bnn_model2)
        model = NonSwagFeatureRegressor(model=model)
    else:
        nn = model
        model = spock_reg_model.load_with_pysr_f2(args.version, args.pysr_version, args.pysr_model_selection, pysr_dir=args.pysr_dir)
        model = NonSwagFeatureRegressor(model=model)


# # read initial condition file
infile_delta_2_to_10 = '../data/initial_conditions_delta_2_to_10.npz'
infile_delta_10_to_13 = '../data/initial_conditions_delta_10_to_13.npz'

ic1 = np.load(infile_delta_2_to_10)
ic2 = np.load(infile_delta_10_to_13)

m_star = ic1['m_star'] # mass of star
m_planet = ic1['m_planet'] # mass of planets
rh = (m_planet/3.) ** (1./3.)

Nbody = ic1['Nbody'] # number of planets
year = 2.*np.pi # One year in units where G=1
tf = ic1['tf'] # end time in years

a_init = np.concatenate([ic1['a'], ic2['a']], axis=1) # array containing initial semimajor axis for each delta,planet
f_init = np.concatenate([ic1['f'], ic2['f']], axis=1) # array containing intial longitudinal position for each delta, planet, run
# -

# # create rebound simulation and predict stability for each system in nsim_list

# +
infile_delta_2_to_10 = '../data/initial_conditions_delta_2_to_10.npz'
infile_delta_10_to_13 = '../data/initial_conditions_delta_10_to_13.npz'

outfile_nbody_delta_2_to_10 = '../data/merged_output_files_delta_2_to_10.npz'
outfile_nbody_delta_10_to_13 = '../data/merged_output_files_delta_10_to_13.npz'

## load hill spacing

ic_delta_2_to_10 = np.load(infile_delta_2_to_10)
ic_delta_10_to_13 = np.load(infile_delta_10_to_13)

delta_2_to_10 = ic_delta_2_to_10['delta']
delta_10_to_13 = ic_delta_10_to_13['delta']

delta = np.hstack((delta_2_to_10, delta_10_to_13))
delta=delta[used_axes]

## load rebound simulation first close encounter times

nbody_delta_2_to_10 = np.load(outfile_nbody_delta_2_to_10)
nbody_delta_10_to_13 = np.load(outfile_nbody_delta_10_to_13)

t_exit_delta_2_to_10 = nbody_delta_2_to_10['t_exit']/(0.99)**(3./2)
t_exit_delta_10_to_13 = nbody_delta_10_to_13['t_exit']/(0.99)**(3./2)

t_exit = np.hstack((t_exit_delta_2_to_10, t_exit_delta_10_to_13))
t_exit = t_exit[used_axes]

df = pd.DataFrame(np.array([nsim_list, delta, t_exit]).T, columns=['nsim', 'delta', 't_exit'])
df.head()

sims = []
for nsim in nsim_list:
    # From Dan's fig 5
    sim = rebound.Simulation()
    sim.add(m=m_star)
    sim.G = 4*np.pi**2
    for i in range(Nbody): # add the planets
        sim.add(m=m_planet, a=a_init[i, nsim], f=f_init[i, nsim])


    init_sim_parameters(sim)
    sims.append(sim)


def data_setup_kernel(mass_array, cur_tseries):
    mass_array = np.tile(mass_array[None], (100, 1))[None]

    old_X = np.concatenate((cur_tseries, mass_array), axis=2)

    isnotfinite = lambda _x: ~np.isfinite(_x)

    old_X = np.concatenate((old_X, isnotfinite(old_X[:, :, [3]]).astype(float)), axis=2)
    old_X = np.concatenate((old_X, isnotfinite(old_X[:, :, [6]]).astype(float)), axis=2)
    old_X = np.concatenate((old_X, isnotfinite(old_X[:, :, [7]]).astype(float)), axis=2)

    old_X[..., :] = np.nan_to_num(old_X[..., :], posinf=0.0, neginf=0.0)

    X = []

    for j in range(old_X.shape[-1]):#: #, label in enumerate(old_axis_labels):
        if j in [11, 12, 13, 17, 18, 19, 23, 24, 25]: #if 'Omega' in label or 'pomega' in label or 'theta' in label:
            X.append(np.cos(old_X[:, :, [j]]))
            X.append(np.sin(old_X[:, :, [j]]))
        else:
            X.append(old_X[:, :, [j]])
    X = np.concatenate(X, axis=2)
    if X.shape[-1] != 41:
        raise NotImplementedError("Need to change indexes above for angles, replace ssX.")

    return X



def get_features_for_sim(sim_i, indices=None):
    sim = sims[sim_i]
    if sim.N_real < 4:
        raise AttributeError("SPOCK Error: SPOCK only works for systems with 3 or more planets")
    if indices:
        if len(indices) != 3:
            raise AttributeError("SPOCK Error: indices must be a list of 3 particle indices")
        trios = [indices] # always make it into a list of trios to test
    else:
        trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios

    kwargs = OrderedDict()
    kwargs['Norbits'] = int(1e4)
    kwargs['Nout'] = 100
    kwargs['trios'] = trios
    args = list(kwargs.values())
    # These are the .npy.
    tseries, stable = get_extended_tseries(sim, args)

    if not stable:
        return np.ones((3, 1, 100, 41))*4

    tseries = np.array(tseries)
    simt = sim.copy()
    alltime = []
    Xs = []
    for i, trio in enumerate(trios):
        sim = simt.copy()
        # These are the .npy.
        cur_tseries = tseries[None, i, :]
        mass_array = np.array([sim.particles[j].m/sim.particles[0].m for j in trio])
        X = data_setup_kernel(mass_array, cur_tseries)
        Xs.append(X)

    return Xs


pool = Pool(7)

X = np.array(pool.map(
    get_features_for_sim,
    range(len(sims))
))[:, :, 0, :, :]
#(sim, trio, time, feature)

# -

# allmeg = X[..., model.swag_ensemble[0].megno_location].ravel()
# allmeg = X[..., model.model.megno_location].ravel()


# Computationally heavy bit:
# Calculate samples:

a_init_p = a_init[:, used_axes]

# +
Xp = (model.ssX
      .transform(X.reshape(-1, X.shape[-1]))
      .reshape(X.shape)
)


Xpp = torch.tensor(Xp).float()

Xflat = Xpp.reshape(-1, X.shape[-2], X.shape[-1])

if model.cuda:
    Xflat = Xflat.cuda()


def get_predictions(model, samples, turbo=False):

    if turbo:
        # our model is basically deterministic, so to speed up, we can just sample once
        swag_samples = 1
    else:
        swag_samples = samples

    time = torch.cat([
        torch.cat([model.sample_full_swag(Xpart).detach().cpu() for Xpart in torch.chunk(Xflat, chunks=10)])[None]
        for _ in range(swag_samples)
    ], dim=0).reshape(swag_samples, X.shape[0], X.shape[1], 2).numpy()

    if turbo:
        # copy back out into (samples, X.shape[0], X.shape[1], 2)
        time = einops.repeat(time, '1 i j k -> samples i j k', samples=samples)

    def fast_truncnorm(
            loc, scale, left=jnp.inf, right=jnp.inf,
            d=10000, nsamp=50, seed=0):
        """Fast truncnorm sampling.

        Assumes scale and loc have the desired shape of output.
        length is number of elements.
        Select nsamp based on expecting at minimum one sample of a Gaussian
            to fit within your (left, right) range.
        Select d based on memory considerations - need to operate on
            a (d, nsamp) array.
        """
        oldscale = scale
        oldloc = loc

        scale = scale.reshape(-1)
        loc = loc.reshape(-1)
        samples = jnp.zeros_like(scale)
        start = 0
        try:
            rng = PRNGKey(seed)
        except:
            rng = 0

        for start in range(0, scale.shape[0], d):

            end = start + d
            if end > scale.shape[0]:
                end = scale.shape[0]

            cd = end-start
            try:
                rand_out = normal(
                    rng,
                    (nsamp, cd)
                )
            except:
                rand_out = np.random.normal(size=(nsamp, cd))
            rng += 1

            rand_out = (
                rand_out * scale[None, start:end]
                + loc[None, start:end]
            )

            #rand_out is (nsamp, cd)
            if right == jnp.inf:
                mask = (rand_out > left)
            elif left == jnp.inf:
                mask = (rand_out < right)
            else:
                mask = (rand_out > left) & (rand_out < right)

            first_good_val = rand_out[
                mask.argmax(0), jnp.arange(cd)
            ]

            try:
                samples = jax.ops.index_update(
                    samples, np.s_[start:end], first_good_val
                )
            except:
                samples[start:end] = first_good_val

        return samples.reshape(*oldscale.shape)

    samps_time = np.array(fast_truncnorm(
            time[..., 0], time[..., 1],
            left=4, d=10000, nsamp=40,
            seed=int((ttime()*1e6) % 1e10)
        ))

    #Resample with prior:
    if not args.extrapolate:
        stable_past_9 = samps_time >= 9

        _prior = lambda logT: (
            3.27086190404742*np.exp(-0.424033970670719 * logT) -
            10.8793430454878*np.exp(-0.200351029031774 * logT**2)
        )
        normalization = quad(_prior, a=9, b=np.inf)[0]

        prior = lambda logT: _prior(logT)/normalization

        # Let's generate random samples of that prior:
        n_samples = stable_past_9.sum()
        bins = n_samples*4
        top = 100.
        bin_edges = np.linspace(9, top, num=bins)
        cum_values = [0] + list(np.cumsum(prior(bin_edges)*(bin_edges[1] - bin_edges[0]))) + [1]
        bin_edges = [9.] +list(bin_edges)+[top]
        inv_cdf = interp1d(cum_values, bin_edges)
        r = np.random.rand(n_samples)
        samples = inv_cdf(r)

        samps_time[stable_past_9] = samples

    #min of samples of sampled mu
    outs = np.min(samps_time, 2).T

    log_t_exit = np.log10(t_exit)

    return delta, a_init_p, m_planet, outs, log_t_exit


delta, a_init_p, m_planet, outs, log_t_exit = get_predictions(model, samples, turbo=args.turbo)
if bnn_model:
    _, _, _, bnn_outs, _ = get_predictions(bnn_model, samples, turbo=args.turbo)


cleaned = dict(
    median=[],
    average=[],
    l=[],
    u=[],
    ll=[],
    uu=[],
    true=[],
    delta=[],
    p12=[],
    p23=[],
    m1=[],
    m2=[],
    m3=[],
    bnn_average=[],
    bnn_median=[],
    bnn_l=[],
    bnn_u=[],
    bnn_ll=[],
    bnn_uu=[],
)

if not bnn_model:
    bnn_outs = outs

for i in range(len(outs)):
    cleaned['true'].append(log_t_exit[i])
    cleaned['delta'].append(delta[i])
    a1 = a_init_p[0, i]
    a2 = a_init_p[1, i]
    a3 = a_init_p[2, i]
    p12 = (a1/a2)**(3./2)
    p23 = (a2/a3)**(3./2)
    m1 = m_planet
    m2 = m_planet
    m3 = m_planet
    cleaned['p12'].append(p12)
    cleaned['p23'].append(p23)
    cleaned['m1'].append(m1)
    cleaned['m2'].append(m2)
    cleaned['m3'].append(m3)

    if log_t_exit[i] <= 4.0:
        cleaned['average'].append(log_t_exit[i])
        cleaned['median'].append(log_t_exit[i])
        cleaned['l'].append(log_t_exit[i])
        cleaned['u'].append(log_t_exit[i])
        cleaned['ll'].append(log_t_exit[i])
        cleaned['uu'].append(log_t_exit[i])
    elif outs[i] is not None:
        cleaned['average'].append(np.average(outs[i]))
        cleaned['median'].append(np.median(outs[i]))
        cleaned['l'].append(np.percentile(outs[i], 50+68/2))
        cleaned['u'].append(np.percentile(outs[i], 50-68/2))
        cleaned['ll'].append(np.percentile(outs[i], 50+95/2))
        cleaned['uu'].append(np.percentile(outs[i], 50-95/2))
    else:
        cleaned['average'].append(4.)
        cleaned['median'].append(4.)
        cleaned['l'].append(4.)
        cleaned['u'].append(4.)
        cleaned['ll'].append(4.)
        cleaned['uu'].append(4.)

    if bnn_model:
        if log_t_exit[i] <= 4.0:
            cleaned['bnn_average'].append(log_t_exit[i])
            cleaned['bnn_median'].append(log_t_exit[i])
            cleaned['bnn_l'].append(log_t_exit[i])
            cleaned['bnn_u'].append(log_t_exit[i])
            cleaned['bnn_ll'].append(log_t_exit[i])
            cleaned['bnn_uu'].append(log_t_exit[i])
        elif bnn_outs[i] is not None:
            cleaned['bnn_average'].append(np.average(bnn_outs[i]))
            cleaned['bnn_median'].append(np.median(bnn_outs[i]))
            cleaned['bnn_l'].append(np.percentile(bnn_outs[i], 50+68/2))
            cleaned['bnn_u'].append(np.percentile(bnn_outs[i], 50-68/2))
            cleaned['bnn_ll'].append(np.percentile(bnn_outs[i], 50+95/2))
            cleaned['bnn_uu'].append(np.percentile(bnn_outs[i], 50-95/2))
        else:
            cleaned['bnn_average'].append(4.)
            cleaned['bnn_median'].append(4.)
            cleaned['bnn_l'].append(4.)
            cleaned['bnn_u'].append(4.)
            cleaned['bnn_ll'].append(4.)
            cleaned['bnn_uu'].append(4.)

cleaned = pd.DataFrame(cleaned)

for key in 'average median l u ll uu'.split(' '):
    cleaned.loc[cleaned['true']<=4.0, key] = cleaned.loc[cleaned['true']<=4.0, 'true']


# +
cleaned['petitf'] = np.log10(pd.Series([Tsurv(
        *list(cleaned[['p12', 'p23']].iloc[i]),
        [m_planet, m_planet, m_planet],
        res=False,
        fudge=1,
        m0=1
    )
    for i in range(len(cleaned))]))

cleaned['pperiodetitf'] = np.log10(pd.Series([Tsurv(
        *list(cleaned[['p12', 'p23']].iloc[i]),
        [m_planet, m_planet, m_planet],
        res=False,
        fudge=2,
        m0=1
    )
     for i in range(len(cleaned))]))

mse = ((cleaned['true'] - cleaned['median'])**2).mean()
print('rmse: ', mse**0.5)
bnn_mse = ((cleaned['true'] - cleaned['bnn_median'])**2).mean()
print('bnn rmse: ', bnn_mse**0.5)

# trying to compare rmse to petit
# ix1 = cleaned[cleaned['true'] != cleaned['bnn_median']].index
# ix2 = cleaned[cleaned['petitf'] <= 10].index
# ix3 = ix1.intersection(ix2)
# c2 = cleaned.loc[ix3]
# rmse_eq = np.sqrt(np.mean((c2['true'] - c2['median'])**2))
# 1.1543915552024004
# rmse_bnn = np.sqrt(np.mean((c2['true'] - c2['bnn_median'])**2))
# 0.3554099783721814
# rmse_petit = np.sqrt(np.mean((c2['true'] - c2['petitf'])**2))
# 1.0603360947824365


# +
# petit = Tsurv(p12, p23, [m1, m2, m3])
# petit = np.nan_to_num(np.log10(Tsurv), posinf=1e9, neginf=0.0)
# cleaned['petit'].append(petit)
# petit = Tsurv(p12, p23, [m1, m2, m3], fudge=2)
# petit = np.nan_to_num(np.log10(Tsurv), posinf=1e9, neginf=0.0)
# cleaned['petitf'].append(petit)
# -

path = f'five_planet_figures/five_planet2_v{args.version}_pysr{args.pysr_version}'
# path = f'five_planet_v{args.version}_pysr{args.pysr_version}'
if args.pysr_model_selection != 'accuracy':
    path += f'_ms={args.pysr_model_selection}'

path += f'_N={N}'
if args.turbo:
    path += f'_turbo'
else:
    path += f'_samps={samples}'

if args.extrapolate:
    path += '_extrapolate'

filename = f'cur_plot_datasets/{path}_{time.time()}.csv'
cleaned.to_csv(filename)
print('saved data to', filename)

path += '.png'

# make_plot_separate(cleaned, path=path)
make_plot2(cleaned, path=path)

print('made plot')
