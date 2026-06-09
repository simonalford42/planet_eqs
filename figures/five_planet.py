#!/usr/bin/env python
# coding: utf-8
# %matplotlib inline

import sys
import os
import json
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(REPO_ROOT)

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
from types import SimpleNamespace
import sys
sys.path.append(os.path.join(SCRIPT_DIR, 'spock'))
from tseries_feature_functions import get_extended_tseries
from pure_sr_evaluation import pure_sr_predict_fn
from modules import PureSRNet
import evaluation


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
    parser.add_argument('--N', type=int, default=5000)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--version', '-v', type=int, default=24880)

    parser.add_argument('--turbo', action='store_true', help='skips sampling. No perf difference for equation models that are basically deterministic')

    parser.add_argument('--pysr_version', type=int, default=11003)
    parser.add_argument('--pure_sr', action='store_true')
    parser.add_argument('--pysr_dir', type=str, default='../sr_results/')  # folder containing pysr results pkl
    parser.add_argument('--pysr_model_selection', type=str, default=26, help='"best", "accuracy", "score", or an integer of the pysr equation complexity. If not provided, will do all complexities. If plotting, has to be an integer complexity')
    parser.add_argument('--eq_std_version', type=int, default=None, help='Version of NN trained to predict std for the pysr equation (e.g. 15687). When set with --pysr_version, the equation model uses this NN for std instead of its own.')
    parser.add_argument('--empirical_val_std', action='store_true', default=True,
                        help='Use the selected equation method validation residual RMSE as five-planet sampling std.')
    parser.add_argument('--no_empirical_val_std', action='store_false', dest='empirical_val_std',
                        help='Use the model-provided std instead of validation residual RMSE.')

    parser.add_argument('--extrapolate', action='store_true', help='disables prior for >= 9')

    parser.add_argument('--rebuild_cache', action='store_true',
                        help='ignore the cached features and recompute (the cache only depends on --N).')
    parser.add_argument('--official', action='store_true',
                        help='Generate all official five-planet CSVs, then run five_planet_plot.py.')
    parser.add_argument('--version_json', type=str, default='../official_versions.json',
                        help='Version JSON used by --official and the final official plot step.')

    args = parser.parse_args()

    return args


def resolve_path(path, base_dir):
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


def load_version_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def run_official(args):
    version_json = resolve_path(args.version_json, SCRIPT_DIR)
    v = load_version_json(version_json)
    script_path = os.path.abspath(__file__)

    runs = [
        [
            sys.executable, script_path,
            '--version', str(v['nn_version']),
            '--pysr_version', str(v['pysr_version']),
            '--pysr_model_selection', str(v['pysr_model_selection']),
        ],
        [
            sys.executable, script_path,
            '--pure_sr',
            '--pysr_version', str(v['pure_sr_version']),
            '--pysr_model_selection', str(v['pure_sr_model_selection']),
        ],
        [
            sys.executable, script_path,
            '--version', str(v['pure_sr2_nn_version']),
            '--pysr_version', str(v['pure_sr2_version']),
            '--pysr_model_selection', str(v['pure_sr2_model_selection']),
        ],
    ]

    runs.append([
        sys.executable, script_path,
        '--version', str(v['nn_version']),
        '--pysr_version', str(v['pysr_version']),
        '--pysr_model_selection', str(v['pysr_model_selection']),
        '--eq_std_version', str(v['eq_std_nn_version']),
        '--no_empirical_val_std',
    ])

    common = ['--N', str(args.N), '--turbo', '--extrapolate']
    for i, cmd in enumerate(runs):
        full_cmd = cmd + common
        if args.rebuild_cache and i == 0:
            full_cmd.append('--rebuild_cache')
        print('Running:', ' '.join(full_cmd))
        subprocess.run(full_cmd, check=True, cwd=SCRIPT_DIR)

    plot_cmd = [
        sys.executable, 'five_planet_plot.py',
        '--version_json', version_json,
        '--N', str(args.N),
    ]
    print('Running:', ' '.join(plot_cmd))
    subprocess.run(plot_cmd, check=True, cwd=SCRIPT_DIR)


def get_empirical_validation_std(args):
    """Validation residual RMSE for the selected equation-like method.

    This follows evaluation.py's plotting metric convention: average the two
    truth columns, use the mean prediction, clip predictions to [4, 9], and
    restrict to systems with averaged truth < 9. We use RMSE as the predictive
    sigma because the five-planet sampler does not subtract validation bias.
    """
    if args.pure_sr:
        eval_args = SimpleNamespace(
            eval_type='pure_sr',
            pysr_version=args.pysr_version,
            pysr_model_selection=str(args.pysr_model_selection),
            dataset='val',
        )
        label = f'pure_sr_{args.pysr_version}_{args.pysr_model_selection}'
    else:
        eval_args = SimpleNamespace(
            eval_type='pysr',
            version=args.version,
            pysr_version=args.pysr_version,
            pysr_model_selection=str(args.pysr_model_selection),
            dataset='val',
        )
        label = f'pysr_{args.version}_{args.pysr_version}_{args.pysr_model_selection}'

    old_cwd = os.getcwd()
    try:
        os.chdir(REPO_ROOT)
        truths, preds = evaluation.get_truths_and_preds(eval_args, clip=True)
    finally:
        os.chdir(old_cwd)
    unstable = truths < 9
    residuals = preds[unstable] - truths[unstable]
    rmse = float(np.sqrt(np.mean(residuals**2)))
    residual_std = float(np.std(residuals))
    bias = float(np.mean(residuals))
    print(
        f'Using empirical validation residual RMSE for {label}: '
        f'rmse={rmse:.4f}'
    )
    return rmse

args = get_args()
if args.official:
    run_official(args)
    sys.exit(0)

N = args.N
samples = args.samples
version = args.version

if args.pysr_version is not None:
    args.turbo = True

# if args.turbo:
#     args.samples = 1

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

        if args.eq_std_version is not None:
            # Use a separately-trained NN's std for the equation, keeping the
            # equation's mean. Both models share the same f1, so we can swap
            # the std at the predict_instability level.
            from spock_reg_model import soft_clamp
            eq_std_full = spock_reg_model.load(args.eq_std_version)
            eq_std_full.eval()
            if model.cuda:
                eq_std_full = eq_std_full.cuda()
            eq_std_regress = eq_std_full.regress_nn
            eq_std_lowest = float(eq_std_full.lowest)

            base_predict_instability = model.model.predict_instability

            def _predict_instability_with_eq_std(summary_stats):
                mu, _ = base_predict_instability(summary_stats)
                std_raw = eq_std_regress(summary_stats)[:, [1]]
                # 15687's regress_nn was trained against soft_clamp, NOT hard
                # clamp — its raw outputs are inverse-tanh of the target std.
                std = soft_clamp(std_raw, eq_std_lowest, 6.0)
                return mu, std

            model.model.predict_instability = _predict_instability_with_eq_std


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

# sims is only needed when we actually run get_features_for_sim (i.e. on a
# cache miss). It's populated below inside the else branch of the cache check.
sims = None


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
        return np.ones((3, 1, 100, 41))*4, False

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

    return Xs, True


cache_dir = 'feature_cache'
os.makedirs(cache_dir, exist_ok=True)
cache_path = os.path.join(cache_dir, f'features_N={N}.npz')

if os.path.exists(cache_path) and not args.rebuild_cache:
    print(f'Loading cached features from {cache_path} (use --rebuild_cache to recompute)')
    _cache = np.load(cache_path)
    X = _cache['X']
    stable_flags = _cache['stable_flags']
else:
    print(f'Computing features for N={N} sims (will cache to {cache_path})')
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

    pool = Pool(7)
    _results = pool.map(get_features_for_sim, range(len(sims)))
    X = np.array([r[0] for r in _results])[:, :, 0, :, :]
    stable_flags = np.array([r[1] for r in _results], dtype=bool)
    np.savez(cache_path, X=X, stable_flags=stable_flags)
    print(f'Cached to {cache_path}')
#(sim, trio, time, feature)
# print(f'stable=False for {(~stable_flags).sum()} / {len(stable_flags)} simulations '
    #   f'(their predictions will be replaced with the true value)')

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


def get_predictions(model, samples, turbo=False, fixed_eq_std=None):

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

    # Equation-like models are deterministic in mean. For five-planet sampling,
    # give them a method-specific empirical validation residual scatter.
    if fixed_eq_std is not None:
        time[..., 1] = fixed_eq_std

    if samples == 1:
        samps_time = time[0, :, :, 0]
    else:
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
        bin_edges = [9.] + list(bin_edges)+[top]
        inv_cdf = interp1d(cum_values, bin_edges)
        r = np.random.rand(n_samples)
        samples = inv_cdf(r)

        samps_time[stable_past_9] = samples

    #min of samples of sampled mu
    outs = np.min(samps_time, 2).T

    log_t_exit = np.log10(t_exit)

    # raw model output for each trio (before truncated-normal sampling and
    # min-over-trios). shape (n_systems, n_trios, 2): col 0 = mu, col 1 = std.
    raw_per_trio = time[0].copy()

    return delta, a_init_p, m_planet, outs, log_t_exit, raw_per_trio


eq_fixed_std = (
    get_empirical_validation_std(args)
    if args.pysr_version and args.empirical_val_std
    else None
)
delta, a_init_p, m_planet, outs, log_t_exit, raw_per_trio = get_predictions(model, samples, turbo=args.turbo, fixed_eq_std=eq_fixed_std)
if bnn_model:
    _, _, _, bnn_outs, _, bnn_raw_per_trio = get_predictions(bnn_model, samples, turbo=args.turbo)
else:
    bnn_raw_per_trio = raw_per_trio


cleaned = dict(
    median=[],
    average=[],
    average_std=[],
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
    bnn_average_std=[],
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

    if log_t_exit[i] <= 4.0 or not stable_flags[i]:
        cleaned['average'].append(log_t_exit[i])
        cleaned['median'].append(log_t_exit[i])
        cleaned['average_std'].append(0.)
        cleaned['l'].append(log_t_exit[i])
        cleaned['u'].append(log_t_exit[i])
        cleaned['ll'].append(log_t_exit[i])
        cleaned['uu'].append(log_t_exit[i])
    elif outs[i] is not None:
        cleaned['average'].append(np.average(outs[i]))
        cleaned['median'].append(np.median(outs[i]))
        cleaned['average_std'].append(np.std(outs[i]))
        cleaned['l'].append(np.percentile(outs[i], 50+68/2))
        cleaned['u'].append(np.percentile(outs[i], 50-68/2))
        cleaned['ll'].append(np.percentile(outs[i], 50+95/2))
        cleaned['uu'].append(np.percentile(outs[i], 50-95/2))
    else:
        cleaned['average'].append(4.)
        cleaned['median'].append(4.)
        cleaned['average_std'].append(0.)
        cleaned['l'].append(4.)
        cleaned['u'].append(4.)
        cleaned['ll'].append(4.)
        cleaned['uu'].append(4.)

    if log_t_exit[i] <= 4.0 or not stable_flags[i]:
        cleaned['bnn_average'].append(log_t_exit[i])
        cleaned['bnn_median'].append(log_t_exit[i])
        cleaned['bnn_average_std'].append(0.)
        cleaned['bnn_l'].append(log_t_exit[i])
        cleaned['bnn_u'].append(log_t_exit[i])
        cleaned['bnn_ll'].append(log_t_exit[i])
        cleaned['bnn_uu'].append(log_t_exit[i])
    elif bnn_outs[i] is not None:
        cleaned['bnn_average'].append(np.average(bnn_outs[i]))
        cleaned['bnn_median'].append(np.median(bnn_outs[i]))
        cleaned['bnn_average_std'].append(np.std(bnn_outs[i]))
        cleaned['bnn_l'].append(np.percentile(bnn_outs[i], 50+68/2))
        cleaned['bnn_u'].append(np.percentile(bnn_outs[i], 50-68/2))
        cleaned['bnn_ll'].append(np.percentile(bnn_outs[i], 50+95/2))
        cleaned['bnn_uu'].append(np.percentile(bnn_outs[i], 50-95/2))
    else:
        cleaned['bnn_average'].append(4.)
        cleaned['bnn_median'].append(4.)
        cleaned['bnn_average_std'].append(0.)
        cleaned['bnn_l'].append(4.)
        cleaned['bnn_u'].append(4.)
        cleaned['bnn_ll'].append(4.)
        cleaned['bnn_uu'].append(4.)

cleaned = pd.DataFrame(cleaned)
cleaned['empirical_val_rmse_as_std'] = np.nan if eq_fixed_std is None else eq_fixed_std
cleaned['uses_empirical_val_rmse_as_std'] = eq_fixed_std is not None

# Per-trio raw model output (before truncnorm sampling and min-over-trios).
# raw_per_trio is shape (n_systems, n_trios, 2): [..., 0] = mu, [..., 1] = std.
# "eq_" columns reflect the (possibly patched) equation model; "bnn_" columns
# reflect the original NN model when --pysr_version is set, else they mirror eq.
n_trios = raw_per_trio.shape[1]
for j in range(n_trios):
    cleaned[f'eq_mu_t{j}'] = raw_per_trio[:, j, 0]
    cleaned[f'eq_std_t{j}'] = raw_per_trio[:, j, 1]
    cleaned[f'bnn_mu_t{j}'] = bnn_raw_per_trio[:, j, 0]
    cleaned[f'bnn_std_t{j}'] = bnn_raw_per_trio[:, j, 1]

# σ-free "predicted min instability": min over trios of the per-trio μ. Also
# expose the std of the trio that wins the min (i.e. the most-constrained one).
eq_min_trio_idx = np.argmin(raw_per_trio[:, :, 0], axis=1)
bnn_min_trio_idx = np.argmin(bnn_raw_per_trio[:, :, 0], axis=1)
row_ix = np.arange(raw_per_trio.shape[0])
cleaned['eq_mean_min'] = raw_per_trio[row_ix, eq_min_trio_idx, 0]
cleaned['eq_std_min_trio'] = raw_per_trio[row_ix, eq_min_trio_idx, 1]
cleaned['bnn_mean_min'] = bnn_raw_per_trio[row_ix, bnn_min_trio_idx, 0]
cleaned['bnn_std_min_trio'] = bnn_raw_per_trio[row_ix, bnn_min_trio_idx, 1]

cleaned['stable'] = stable_flags

# stable=False simulations were fed placeholder features (all 4s), so every
# downstream prediction for them is meaningless. Replace per-trio μ with the
# true t_exit and zero out σ; the aggregate columns are already handled by
# the loop above via `not stable_flags[i]`.
unstable_mask = ~stable_flags
for j in range(n_trios):
    cleaned.loc[unstable_mask, f'eq_mu_t{j}'] = log_t_exit[unstable_mask]
    cleaned.loc[unstable_mask, f'eq_std_t{j}'] = 0.0
    cleaned.loc[unstable_mask, f'bnn_mu_t{j}'] = log_t_exit[unstable_mask]
    cleaned.loc[unstable_mask, f'bnn_std_t{j}'] = 0.0
cleaned.loc[unstable_mask, 'eq_mean_min'] = log_t_exit[unstable_mask]
cleaned.loc[unstable_mask, 'eq_std_min_trio'] = 0.0
cleaned.loc[unstable_mask, 'bnn_mean_min'] = log_t_exit[unstable_mask]
cleaned.loc[unstable_mask, 'bnn_std_min_trio'] = 0.0

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
bnn_mse = ((cleaned['true'] - cleaned['bnn_median'])**2).mean()

path = f'five_planet_figures/v{args.version}_pysr{args.pysr_version}'
if args.pysr_model_selection != 'accuracy':
    path += f'_ms={args.pysr_model_selection}'
if args.eq_std_version is not None:
    path += f'_eqstd{args.eq_std_version}'

path += f'_N={N}'
if args.turbo:
    path += f'_turbo'
else:
    path += f'_samps={samples}'

if args.extrapolate:
    path += '_extrapolate'

filename = path + '.csv'
cleaned.to_csv(filename)
print('saved data to', filename)

# filename = path + '.pdf'
# from five_planet_plot import make_main_plot
# make_main_plot(cleaned, path=filename)
