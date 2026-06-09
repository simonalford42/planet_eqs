"""
Calibration of predicted uncertainty for the NN (v=24880) and for the
distilled equations (mean = v=11003, std = v=15687) on the random test set.

For each method, computes |truth - mu| / sigma and compares its distribution
to |N(0, 1)| (half-normal). A well-calibrated sigma yields the two
histograms overlapping.

Run:
    cd figures
    python calibration_figure.py
"""
import os
import json
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


PICKLE_DIR = '../pickles'

MODEL_COLOR = '#74104F'
GAUSS_COLOR = '#CDA459'


def load_cache(name):
    with open(os.path.join(PICKLE_DIR, name), 'rb') as f:
        return pickle.load(f)


def abs_z(truths_full, mu, sigma, truth_lo, truth_hi):
    """|truth - mu| / sigma, broadcasting mu/sigma over both truth columns,
    keeping only entries whose truth is in (truth_lo, truth_hi)."""
    mu_tiled = np.tile(mu[:, None], (1, truths_full.shape[1]))
    sigma_tiled = np.tile(sigma[:, None], (1, truths_full.shape[1]))
    mask = (truths_full > truth_lo) & (truths_full < truth_hi)
    z = (truths_full - mu_tiled) / sigma_tiled
    return np.abs(z[mask])


def plot_panel(ax, z, title, bins, xrange):
    ax.hist(
        z,
        bins=bins, range=xrange, density=True,
        color=MODEL_COLOR, alpha=1.0,
        label='Model error distribution',
    )
    rng = np.random.default_rng(0)
    ax.hist(
        np.abs(rng.standard_normal(len(z))),
        bins=bins, range=xrange, density=True,
        color=GAUSS_COLOR, alpha=0.5,
        label='Gaussian distribution',
    )
    ax.set_xlim(*xrange)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('Error over sigma', fontsize=12)
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version_json', default='../official_versions.json')
    parser.add_argument('--truth_lo', type=float, default=5.0,
                        help='lower bound on truth to include')
    parser.add_argument('--truth_hi', type=float, default=9.0,
                        help='upper bound on truth to include')
    parser.add_argument('--bins', type=int, default=30)
    parser.add_argument('--xmax', type=float, default=2.5)
    parser.add_argument('--png', action='store_true')
    parser.add_argument('--out', default='calibration')
    args = parser.parse_args()

    v = load_json(args.version_json)
    nn_version = v['nn_version']             # 24880
    pysr_version = v['pysr_version']         # 11003
    pysr_sel = v['pysr_model_selection']     # 26
    eq_std_version = 15687

    nn_cache = load_cache(f'cache_truths_preds_nn_{nn_version}_random.pkl')
    pysr_cache = load_cache(
        f'cache_truths_preds_pysr_{nn_version}_{pysr_version}_{pysr_sel}_random.pkl'
    )
    eq_std_cache = load_cache(f'cache_truths_preds_nn_{eq_std_version}_random.pkl')

    truths = nn_cache['truths_full']  # [N, 2]
    assert np.allclose(truths, pysr_cache['truths_full'])
    assert np.allclose(truths, eq_std_cache['truths_full'])

    nn_mu = nn_cache['preds_full'][:, 0]
    nn_sigma = nn_cache['preds_full'][:, 1]
    eq_mu = pysr_cache['preds_full'][:, 0]
    eq_sigma = eq_std_cache['preds_full'][:, 1]

    nn_z = abs_z(truths, nn_mu, nn_sigma, args.truth_lo, args.truth_hi)
    eq_z = abs_z(truths, eq_mu, eq_sigma, args.truth_lo, args.truth_hi)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    plot_panel(axs[0], nn_z, f'Neural network (v={nn_version})',
               args.bins, (0, args.xmax))
    plot_panel(axs[1], eq_z,
               f'Equations (mean v={pysr_version}, std v={eq_std_version})',
               args.bins, (0, args.xmax))
    axs[0].set_ylabel('Density', fontsize=12)
    axs[1].legend(loc='upper right', fontsize=9)
    fig.suptitle(
        f'Calibration on random test set '
        f'(truth in ({args.truth_lo:g}, {args.truth_hi:g}), '
        f'N_NN={len(nn_z)}, N_eq={len(eq_z)})',
        y=1.02,
    )
    plt.tight_layout()

    ext = '.png' if args.png else '.pdf'
    out_path = args.out + ext
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved calibration figure to {out_path}')


if __name__ == '__main__':
    main()
