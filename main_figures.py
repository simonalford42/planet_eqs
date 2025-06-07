#!/usr/bin/env python
# coding: utf-8
import argparse
from copy import deepcopy as copy
import einops
import glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import quad
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score
import spock_reg_model
from tqdm.notebook import tqdm
import wandb
import petit
import pickle
from pure_sr_evaluation import pure_sr_predict_fn, get_pure_sr_results, lambdify_pure_sr_expression

# to fix the fonts?
plt.rcParams.update(plt.rcParamsDefault)


def fast_truncnorm(
        loc, scale, left=np.inf, right=np.inf,
        d=10000, nsamp=50, seed=0):
    """Fast truncnorm sampling.

    Assumes scale and loc have the desired shape of output.
    length is number of elements.
    Select nsamp based on expecting at last one sample
        to fit within your (left, right) range.
    Select d based on memory considerations - need to operate on
        a (d, nsamp) array.
    """
    oldscale = scale
    oldloc = loc

    scale = scale.reshape(-1)
    loc = loc.reshape(-1)
    samples = np.zeros_like(scale)
    start = 0

    for start in range(0, scale.shape[0], d):

        end = start + d
        if end > scale.shape[0]:
            end = scale.shape[0]

        cd = end-start
        rand_out = np.random.randn(
            nsamp, cd
        )

        rand_out = (
            rand_out * scale[None, start:end]
            + loc[None, start:end]
        )

        if right == np.inf:
            mask = (rand_out > left)
        elif left == np.inf:
            mask = (rand_out < right)
        else:
            mask = (rand_out > left) & (rand_out < right)

        first_good_val = rand_out[
            mask.argmax(0), np.arange(cd)
        ]

        samples[start:end] = first_good_val

    return samples.reshape(*oldscale.shape)


def default_dataloader(random=False, train=False):
    model = spock_reg_model.load(24880)
    model.make_dataloaders(train=train)
    if random:
        from copy import deepcopy as copy
        assert model.ssX is not None
        tmp_ssX = copy(model.ssX)
        model.make_dataloaders(
            ssX=model.ssX,
            train=False,
            plot_random=True) #train=False means we show the whole dataset (assuming we don't train on it!)

        assert np.all(tmp_ssX.mean_ == model.ssX.mean_)

    return model._val_dataloader

def get_colors():
    ########################################
    # Stuff with loading colors

    colorstr = """*** Primary color:

       shade 0 = #A0457E = rgb(160, 69,126) = rgba(160, 69,126,1) = rgb0(0.627,0.271,0.494)
       shade 1 = #CD9CBB = rgb(205,156,187) = rgba(205,156,187,1) = rgb0(0.804,0.612,0.733)
       shade 2 = #BC74A1 = rgb(188,116,161) = rgba(188,116,161,1) = rgb0(0.737,0.455,0.631)
       shade 3 = #892665 = rgb(137, 38,101) = rgba(137, 38,101,1) = rgb0(0.537,0.149,0.396)
       shade 4 = #74104F = rgb(116, 16, 79) = rgba(116, 16, 79,1) = rgb0(0.455,0.063,0.31)

    *** Secondary color (1):

       shade 0 = #CDA459 = rgb(205,164, 89) = rgba(205,164, 89,1) = rgb0(0.804,0.643,0.349)
       shade 1 = #FFE9C2 = rgb(255,233,194) = rgba(255,233,194,1) = rgb0(1,0.914,0.761)
       shade 2 = #F1D195 = rgb(241,209,149) = rgba(241,209,149,1) = rgb0(0.945,0.82,0.584)
       shade 3 = #B08431 = rgb(176,132, 49) = rgba(176,132, 49,1) = rgb0(0.69,0.518,0.192)
       shade 4 = #956814 = rgb(149,104, 20) = rgba(149,104, 20,1) = rgb0(0.584,0.408,0.078)

    *** Secondary color (2):

       shade 0 = #425B89 = rgb( 66, 91,137) = rgba( 66, 91,137,1) = rgb0(0.259,0.357,0.537)
       shade 1 = #8C9AB3 = rgb(140,154,179) = rgba(140,154,179,1) = rgb0(0.549,0.604,0.702)
       shade 2 = #697DA0 = rgb(105,125,160) = rgba(105,125,160,1) = rgb0(0.412,0.49,0.627)
       shade 3 = #294475 = rgb( 41, 68,117) = rgba( 41, 68,117,1) = rgb0(0.161,0.267,0.459)
       shade 4 = #163163 = rgb( 22, 49, 99) = rgba( 22, 49, 99,1) = rgb0(0.086,0.192,0.388)

    *** Complement color:

       shade 0 = #A0C153 = rgb(160,193, 83) = rgba(160,193, 83,1) = rgb0(0.627,0.757,0.325)
       shade 1 = #E0F2B7 = rgb(224,242,183) = rgba(224,242,183,1) = rgb0(0.878,0.949,0.718)
       shade 2 = #C9E38C = rgb(201,227,140) = rgba(201,227,140,1) = rgb0(0.788,0.89,0.549)
       shade 3 = #82A62E = rgb(130,166, 46) = rgba(130,166, 46,1) = rgb0(0.51,0.651,0.18)
       shade 4 = #688C13 = rgb(104,140, 19) = rgba(104,140, 19,1) = rgb0(0.408,0.549,0.075)"""

    colors = []
    shade = 0
    for l in colorstr.replace(' ', '').split('\n'):
        elem = l.split('=')
        if len(elem) != 5: continue
        if shade == 0:
            new_color = []

        new_color.append(eval(elem[2]))

        shade += 1
        if shade == 5:
            colors.append(np.array(new_color))
            shade = 0
    colors = np.array(colors)/255.0
    return colors

def comparison_figure(model, train_all=False, plot_random=False, use_petit=False, train_set=False, use_pure_sr=False, pure_sr_version=None, pure_sr_i=None):
    model.eval()
    model.cuda()

    path = 'plots/' + model.path()
    if use_petit:
        path = 'plots/petit'

    if plot_random:
        path += '_random'

    plt.switch_backend('agg')

    colors = get_colors()

    #################################################3
    ### Load the dataloader

    if use_petit:
        val_dataloader = petit.petit_dataloader(validation=True)

        def sample_model(X_sample):
            return petit.tsurv(X_sample)

    elif use_pure_sr:
        model.make_dataloaders(train=train_set)
        val_dataloader = model._val_dataloader

        reg = pickle.load(open(f'sr_results/{pure_sr_version}.pkl', 'rb'))
        i = pure_sr_i

        expr = reg.equations_.iloc[i].equation
        var_names = reg.feature_names_in_
        expr_fn = lambdify_pure_sr_expression(expr, var_names)

        def sample_model(X_sample):
            return expr_fn(X_sample)

    else:
        model.make_dataloaders(train=train_set)
        if plot_random:
            assert model.ssX is not None
            tmp_ssX = copy(model.ssX)
            if train_all:
                model.make_dataloaders(
                    ssX=model.ssX,
                    train=True,
                    plot_random=True)
            else:
                model.make_dataloaders(
                    ssX=model.ssX,
                    train=False,
                    plot_random=True) #train=False means we show the whole dataset (assuming we don't train on it!)

            assert np.all(tmp_ssX.mean_ == model.ssX.mean_)

        val_dataloader = model._val_dataloader

        def sample_model(X_sample):
            return model(X_sample, noisy_val=False)

    # collect the samples

    truths = []
    preds = []
    raw_preds = []

    N = 1  # since we're not using swag, we're deterministic. so just do one sample
    # keeping the rest of the code the same so the shapes dont have to be messed with

    nc = 0
    for X_sample, y_sample in tqdm(val_dataloader):
        X_sample = X_sample.cuda()
        y_sample = y_sample.cuda()
        nc += len(y_sample)
        truths.append(y_sample.cpu().detach().numpy())

        raw_preds.append(
            np.array([sample_model(X_sample).cpu().detach().numpy() for _ in range(N)])
        )

    truths = np.concatenate(truths)

    _preds = np.concatenate(raw_preds, axis=1)

    if use_petit or use_pure_sr:
        sample_preds = _preds
    else:
        std = _preds[..., 1]
        mean = _preds[..., 0]

        sample_preds = np.array(
                fast_truncnorm(np.array(mean), np.array(std),
                       left=4, d=874000, nsamp=40));

    if use_petit or use_pure_sr:
        # sample_preds is a [1, 8720] tensor of means
        # sample with std 1 to create [2000, 8740] tensor of samples
        # samples = np.random.randn(2000, 8740)
        # sample_preds = einops.repeat(sample_preds[0], 'b -> R b', R=2000)
        # sample_preds = sample_preds + samples
        sample_preds = einops.repeat(sample_preds[0], 'b -> R b', R=2000)
        # need to make _preds [1,8740, 2]
        # _preds is currently [1, 8740] of means
        # just use std of one.
        mean = _preds
        std = np.ones_like(mean)
        _preds = einops.rearrange([mean, std], 'two one eight -> one eight two')

    else:
        stable_past_9 = sample_preds >= 9

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

        sample_preds[stable_past_9] = samples

    preds = np.median(_preds[..., 0], 0)
    stds = np.median(_preds[..., 1], 0)

    show_transparency = True

    main_shade = 3

    main_color = colors[2, main_shade]
    off_color = colors[2, main_shade]

    plt.style.use('default')
    sns.set_style('white')
    plt.rc('font', family='serif')

    confidence = 'low'
    py = preds

    py = np.clip(py, 4, 9)
    px = np.average(truths, 1)

    mask = np.all(truths < 9.99, 1)

    ppx = px[mask]
    ppy = py[mask]
    p_std = stds[mask]

    title = 'Our model'
    fig = plt.figure(figsize=(4, 4),
                     dpi=300,
                     constrained_layout=True)

    alpha = 1.0

    main_color = main_color.tolist()
    g = sns.jointplot(ppx, ppy,
                    alpha=alpha,# ax=ax,
                      color=main_color,
#                     hue=(ppy/p_std)**2,
                    s=0.0,
                    xlim=(3, 13),
                    ylim=(3, 13),
                    marginal_kws=dict(bins=15),
                   )

    ax = g.ax_joint
    snr = (ppy/p_std)**2
    relative_snr = snr / max(snr)
    point_color = relative_snr

    rmse = np.average(np.square(ppx[ppx < 8.99] - ppy[ppx < 8.99]))**0.5
    snr_rmse = np.average(np.square(ppx[ppx < 8.99] - ppy[ppx < 8.99]), weights=snr[ppx<8.99])**0.5
    print(f'{confidence} confidence gets RMSE of {rmse:.6f}')
    print(f'Weighted by SNR, this is: {snr_rmse:.6f}')

    #Transparency:
    if show_transparency:
        if plot_random:
            transparency_adjuster = 1.0 #0.5 * 0.2
        else:
            transparency_adjuster = 1.0
        point_color = np.concatenate(
            (einops.repeat(colors[2, 3], 'c -> row c', row=len(ppy)),
             point_color[:, None]*transparency_adjuster), axis=1)
    #color mode:
    else:
        point_color = np.einsum('r,i->ir', main_color, point_color) +\
            np.einsum('r,i->ir', off_color, 1-point_color)

    ax.scatter(ppx, ppy, marker='o', c=point_color, s=10, edgecolors='none')
    ax.plot([4-3, 9+3], [4-3, 9+3], color='k')
    ax.plot([4-3, 9+3], [4+0.61-3, 9+0.61+3], color='k', ls='--')
    ax.plot([4-3, 9+3], [4-0.61-3, 9-0.61+3], color='k', ls='--')
    ax.set_xlim(3+0.9, 10-0.9)
    ax.set_ylim(3+0.9, 10-0.9)

    ax.set_xlabel('Truth')
    ax.set_ylabel('Predicted')
    plt.suptitle(title, y=1.0)
    plt.tight_layout()

    plt.savefig(path + 'comparison.png', dpi=300)
    print(f'Saving to {path}comparison.png')



def nn_prediction_fn(model):
    model.eval()
    model.cuda()
    def predict_fn(x):
        pred = model(x.cuda(), noisy_val=False).cpu().detach().numpy()
        # just the mean
        return pred[..., 0]

    return predict_fn


def const_predict_fn(const):
    return lambda *_: const

def calculate_k_results():
    d = {
        2:  {
            'version': 24880,
            'pysr_version': 11003,
            'val_loss': 1.62794,
        },
        3:  {
            'version': 74649,
            'pysr_version': 83278,
            'val_loss': 1.67736,
        },
        4:  {
            'version': 11566,
            'pysr_version': 51254,
            'val_loss': 1.63825,
        },
        5:  {
            'version': 72646,
            'pysr_version': 55894,
            'val_loss': 1.66181,
        }
    }

    overall_results = {}
    for k in d:
        version = d[k]['version']
        pysr_version = d[k]['pysr_version']
        reg = pickle.load(open(f'sr_results/{pysr_version}.pkl', 'rb'))
        results = reg.equations_[0]
        complexities = results['complexity']
        k_results = {}
        for c in complexities:
            args = get_args()
            args.version = version
            args.pysr_version = pysr_version
            args.pysr_model_selection = c
            args.just_rmse = True
            rmse = main(args)
            k_results[c] = rmse
            print(f'k={k}, c={c}, rmse={rmse}')
        overall_results[k] = k_results

    pickle.dump(overall_results, open('pickles/k_results_test.pkl', 'wb'))


def calculate_f2_lin_results():
    d = {20: 2702,
         10: 13529,
         5: 7307,
         2: 22160}
    for k, version in d.items():
        args = get_args()
        args.version = version
        args.just_rmse = True
        rmse = main(args)
        print(f'k={k}, rmse={rmse}')
        d[k] = {'version': version, 'rmse': rmse}

    pickle.dump(d, open('pickles/f2_lin_results_test.pkl', 'wb'))


def calculate_f1_id_results():
    version = 12370
    pysr_version = 22943

    f1_id_results = {}
    reg = pickle.load(open(f'sr_results/{pysr_version}.pkl', 'rb'))
    complexities = reg.equations_[0]['complexity']
    for c in complexities:
        args = get_args()
        args.version = version
        args.pysr_version = pysr_version
        args.pysr_model_selection = c
        args.just_rmse = True
        rmse = main(args)
        f1_id_results[c] = rmse
        print(f'c={c}, rmse={rmse}')

    pickle.dump(f1_id_results, open('pickles/f1_id_results_test.pkl', 'wb'))


def calculate_nn_and_petit_results():
    version = 24880
    args = get_args()
    args.version = version
    args.just_rmse = True
    rmse = main(args)
    print(f'nn, rmse={rmse}')
    args.petit = True
    petit_rmse = main(args)
    d = {'version': version, 'rmse': rmse, 'petit_rmse': petit_rmse}
    pickle.dump(d, open('pickles/nn_and_petit_results_test.pkl', 'wb'))


def calculate_results_all_complexities(version=24880, pysr_version=11003, test=True):
    results = {}
    reg = pickle.load(open(f'sr_results/{pysr_version}.pkl', 'rb'))
    complexities = reg.equations_[0]['complexity']
    for c in complexities[::-1][0:3]:
    # for c in complexities[::-1]:3]:
        args = get_args()
        args.version = version
        args.pysr_version = pysr_version
        args.pysr_model_selection = c
        args.just_rmse = True
        if not test:
            args.train_set = True
        rmse = main(args)
        results[c] = rmse
        print(f'c={c}, rmse={rmse}')

    pickle.dump(results, open(f"pickles/{version}_{pysr_version}_{'test' if test else 'train'}.pkl", 'wb'))


def calculate_all_results():
    calculate_nn_and_petit_results()
    # calculate_f2_lin_results()
    # calculate_f1_id_results()
    # calculate_k_results()


def main(args):
    if args.pysr_version and not args.pure_sr:
        model = spock_reg_model.load_with_pysr_f2(version=args.version, pysr_version=args.pysr_version, pysr_model_selection=args.pysr_model_selection)
    else:
        model = spock_reg_model.load(version=args.version)

    calc_scores_nonswag(model, train_all=False, plot_random=args.random, use_petit=args.petit, use_pure_sr=args.pure_sr, pure_sr_version=args.pysr_version, pure_sr_i=args.pure_sr_complexity)

    # return calculate_rmse(predict_fn, dataloader)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=int, default=24880)
    parser.add_argument('--pysr_version', type=int, default=None)
    parser.add_argument('--petit', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--pure_sr', action='store_true')
    parser.add_argument('--pure_sr_complexity', type=int, default=None)
    parser.add_argument('--pysr_model_selection', type=str, default='accuracy', help='"best", "accuracy", "score", or an integer of the pysr equation complexity.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # calculate_all_results()
    # calculate_f1_id_results()

    # k = 3 all day
    # calculate_results_all_complexities(version=74649, pysr_version=11900, test=False)
    # k = 3 8 hours
    # calculate_results_all_complexities(version=74649, pysr_version=11900, test=False)

    # k = 4 all day
    # calculate_results_all_complexities(version=11566, pysr_version=51254, test=False)
    # k = 4 8 hours
    # calculate_results_all_complexities(version=11566, pysr_version=83278, test=False)

    # calculate_results_all_complexities(version=24880, pysr_version=58106, test=False)
    args = get_args()
    rmse = main(args)
    print('RMSE:', rmse)

