import pysr
import warnings
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
import argparse

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)


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

# copy needed files to local so we can plot
scopy bnn_chaos_model/figures/period_results/v=43139_ngrid=6_pysr_f2_v=33060/ ~/code/bnn_chaos_model/figures/period_results/
scopy bnn_chaos_model/figures/period_results/v=43139_ngrid=6.pkl ~/code/bnn_chaos_model/figures/period_results/
'''

def get_args():
    print(utils2.get_script_execution_command())
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ngrid', '-n', type=int, default=1600)
    parser.add_argument('--version', '-v', type=int, default=43139)

    parser.add_argument('--plot', '-p', action='store_true')
    parser.add_argument('--compute', action='store_true')
    parser.add_argument('--collate', action='store_true')

    parser.add_argument('--pysr_version', type=str, default=None) # sr_results/33060.pkl
    parser.add_argument('--pysr_dir', type=str, default='../sr_results/')  # folder containing pysr results pkl

    parser.add_argument('--pysr_model_selection', type=str, default=None, help='"best", "accuracy", "score", or an integer of the pysr equation complexity. If not provided, will do all complexities. If plotting, has to be an integer complexity')

    # parallel processing args
    # ix should be in [0, total)
    parser.add_argument('--parallel_ix', '-i', type=int, default=None)
    parser.add_argument('--parallel_total', '-t', type=int, default=None)

    args = parser.parse_args()

    if args.pysr_version is not None:
        args.pysr_path = os.path.join(args.pysr_dir, f'{args.pysr_version}.pkl')
    else:
        args.pysr_path = None

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
    path = get_results_path(args.Ngrid, args.version, args.parallel_ix, args.parallel_total, args.pysr_version, args.pysr_model_selection)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(results, f)

    print('Saved results to', path)
    return results


def get_results_path(Ngrid, version, parallel_ix=None, parallel_total=None, pysr_version=None, pysr_model_selection=None):
    path = f'period_results/v={version}_ngrid={Ngrid}'

    if pysr_model_selection == 'best':
        # find the best model selection, which is the largest number that has a file associated with it
        files = os.listdir(path + f'_pysr_f2_v={pysr_version}/')
        files = [f for f in files if f.endswith('.pkl')]
        nums = [int(file.split('.')[0]) for file in files]
        pysr_model_selection = max(nums)

    if pysr_model_selection is not None:
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
        files = os.listdir(get_results_path(args.Ngrid, args.version)[:-4])
        # filter to those of form f'{ix}-{total}.pkl'
        files = [file for file in files if file.endswith('.pkl')]
        # get the total. use the largest possible total
        assert len(files) > 0
        total = int(files[0].split('-')[1].split('.')[0])
        print('Detected parallel total as', total)
    else:
        total = args.parallel_total

    for ix in range(total):
        path = get_results_path(args.Ngrid, args.version, ix, total)
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
        for metric in ['mean', 'std', 'std2', 'mean2']:
            plot_results(args, metric)
        return

    results = load_results(get_results_path(args.Ngrid, args.version, pysr_version=args.pysr_version, pysr_model_selection=args.pysr_model_selection))
    P12s, P23s = get_period_ratios(args.Ngrid)

    fig, ax = plt.subplots(figsize=(8,6))

    # get the results for the specific metric
    if metric == 'mean':
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

    if metric == 'std' or metric == 'std2':
        cmap = plt.cm.inferno.copy().reversed()
        cmap.set_bad(color='white')
        m = Z[~np.isnan(Z)].max()
        im = ax.pcolormesh(X, Y, Z, vmin=0, vmax=m, cmap=cmap)
        label = "std(log(T_unstable))"
    elif metric == 'mean' or metric == 'mean2':
        cmap = plt.cm.inferno.copy().reversed()
        cmap.set_bad(color='white')
        # zmax = Z[~np.isnan(Z)].max()
        # zmin = Z[~np.isnan(Z)].min()
        zmax = 12
        zmin = 4

        im = ax.pcolormesh(X, Y, Z, vmin=zmin, vmax=zmax, cmap=cmap)
        label = "log(T_unstable)"

    cb = plt.colorbar(im, ax=ax)
    cb.set_label(label)
    ax.set_xlabel("P1/P2")
    ax.set_ylabel("P2/P3")

    if hasattr(args, 'title'):
        ax.set_title(args.title)

    path = get_results_path(args.Ngrid, args.version)
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
    results = load_results(get_results_path(args.Ngrid, args.version))

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
        print(model_selection)
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
    model_selections = [1, 3, 5, 7, 9, 11, 14, 18, 20, 27, 29]

    # if model selection is provided, filter to those
    if args.pysr_model_selection is not None:
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


def plot_summary_stats(args):
    feature_nn = torch.load(f'../models/{args.version}_feature_nn.pt')

    results = load_results(get_results_path(args.Ngrid, args.version))
    # this is the summary stats, so divide by 2 to get feature_nn size
    num_features = next(d['f1'] for d in results if d is not None).shape[0] // 2
    P12s, P23s = get_period_ratios(args.Ngrid)

    def get_title(ix):
        # first half are means, second half are stds
        ix2 = ix % num_features
        feature_str = f1_feature_str(feature_nn, ix2)
        if ix < num_features:
            return f'mean of feature {ix2} = {feature_str}'
        else:
            return f'stdev of feature {ix2} = {feature_str}'


    for ix in range(num_features * 2):

        fig, ax = plt.subplots(figsize=(8,6))

        # get the results for the specific metric
        features = [d['f1'][ix] if d is not None else np.nan for d in results]

        X,Y,Z = get_centered_grid(P12s, P23s, features)

        cmap = plt.cm.inferno.copy().reversed()
        cmap.set_bad(color='white')
        im = ax.pcolormesh(X, Y, Z, cmap=cmap)
        label = f"f1 feature {ix}"

        cb = plt.colorbar(im, ax=ax)
        cb.set_label(label)
        ax.set_xlabel("P1/P2")
        ax.set_ylabel("P2/P3")

        ax.set_title(get_title(ix))

        path = get_results_path(args.Ngrid, args.version)
        # get rid of .pkl
        path = path[:-4]
        path += '_plot'

        path += f'_f1/ix={ix}'

        path += '.png'

        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=800)
        print('Saved figure to', path)
        plt.close(fig)



def f1_feature_str(feature_nn: modules.MaskedLinear, feature_ix):
    labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

    # not all of these labels are actually used. for training, these inputs are zeroed out, but still passed in as zeroes.
    # ideally, the linear layer ignores them, which does happen if i do l1 regularization to it
    skipped = ['nan_mmr_near', 'nan_mmr_far', 'nan_megno', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno']

    # let's make the linear transformation a bit easier to read
    def format_num(x):
        if abs(x) > 0.1:
            return f'{x:.2f}'
        if abs(x) > 0.01:
            return f'{x:.3f}'
        elif abs(x) > 0.001:
            return f'{x:.4f}'
        else:
            return f'{x:.2e}'

    def linear_transformation(i):
        return feature_nn.masked_weight[feature_ix].detach().cpu().numpy()

    # now we can write it as a combination of the input features
    # we'll sort the features by their absolute value to make it a bit easier to read
    def feature_equation(i):
        transformation = linear_transformation(i)
        sorted_ixs = np.argsort(np.abs(transformation))[::-1]
        features = [format_num(transformation[i]) + ' * ' + labels[i] for i in sorted_ixs if transformation[i] != 0]
        return features

    return ' + '.join(feature_equation(feature_ix))


def pysr_error_analysis(args):
    results = load_results(get_results_path(args.Ngrid, args.version))
    pysr_results = load_results(get_results_path(args.Ngrid, args.version, pysr_version=args.pysr_version, pysr_model_selection='best'))
    means = [d['mean'] if d is not None else np.nan for d in results]
    pysr_means = [d['mean'] if d is not None else np.nan for d in pysr_results]
    means = np.array(means)
    pysr_means = np.array(pysr_means)
    assert_equal(means.shape, pysr_means.shape)
    # get rid of spots where either is nan
    good_ixs = np.logical_and(~np.isnan(means), ~np.isnan(pysr_means))
    good_ixs
    means = means[good_ixs]
    pysr_means = pysr_means[good_ixs]
    # sort by ground truth means
    ix = np.argsort(means)
    means = means[ix]
    pysr_means = pysr_means[ix]
    # plot the mse between the two
    fig, ax = plt.subplots()
    # make the dots pretty small
    ax.plot(means, (means - pysr_means)**2 / means**2, '.', markersize=1)
    # set the y height to be 30
    ax.set_ylim(0, 0.4)
    ax.set_xlabel('mean')
    ax.set_ylabel('mse / mean^2')
    plt.savefig('period_results/pysr_error.png')



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
            # pysr_error_analysis(args)
        else:
            plot_results(args)
            # plot_summary_stats(args)

    print('Done')
