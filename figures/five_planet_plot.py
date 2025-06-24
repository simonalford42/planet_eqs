import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import argparse
from utils2 import load_json
# plt.style.use('seaborn-darkgrid')
plt.style.use('seaborn-ticks')

def make_main_plot(cleaned, path=None):

    plt.rc('font', family='serif')

    scale = 1
    lw = 1.2
    fig, ax = plt.subplots(figsize=(7*scale, 2.5*scale), dpi=300)

    tmp = cleaned
    tmp2 = tmp.query('true > 4 & delta > 5')

    tmp.plot('delta', 'true', ax=ax, label='True', c='k', linewidth=lw)
    tmp2.plot('delta', 'median', ax=ax, label='Distilled equations', c='tab:blue', linewidth=lw)
    tmp.plot('delta', 'pperiodetitf', ax=ax, label='Petit+20', c='tab:red', linewidth=lw)

    ax.set_xlabel(r'Interplanetary separation $\Delta$')
    ax.set_ylabel(r'Instability Time')
    leg = ax.legend(loc='upper left', frameon=True, fontsize=8, framealpha=1)
    for line in leg.get_lines():
        line.set_linewidth(3)

    xlims = (2, 13)
    ax.set_xlim(*xlims)
    ax.set_xticks(np.arange(*xlims), minor=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='x', which='minor', direction='in')


    ylims = (0, 12)
    ax.set_ylim(*ylims)
    ax.set_yticks([0, 5, 10])
    ax.set_yticks(np.arange(*ylims), minor=True)
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')

    if path is None:
        t = time.strftime('%Y%m%d_%H%M%S')
        path = f'five_planet_figures/five_planet2_{t}.png'

    fig.tight_layout()
    fig.savefig(path)
    print('Saved to', path)


def make_separate_comparison_plot(cleaned, path):
    plt.rc('font', family='serif')

    scale = 1
    lw=1.2
    fig, axarr = plt.subplots(5, 1, figsize=(7*scale, 10.5 * scale), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.2)
    tmp = cleaned
    # tmp2 = tmp.query('true > 4 & delta > 5')
    tmp2 = tmp
    # decrease buffer around plot in image
    # plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05)


    for i, label in enumerate(['Petit+20', 'Neural Network', 'Distilled Equations', 'Pure SR', 'Pure SR (no intermediate features)']):
        ax = axarr[i]

        tmp.plot('delta', 'true', ax=ax, label='True', c='k', linewidth=lw)
        if label == 'Neural Network':
            tmp2.plot('delta', 'bnn_median', ax=ax, label='Neural network', c='tab:orange', linewidth=lw)
        if label == 'Distilled Equations':
            tmp2.plot('delta', 'median', ax=ax, label='Distilled equations', c='tab:blue', linewidth=lw)
        elif label == 'Petit+20':
            tmp.plot('delta', 'pperiodetitf', ax=ax, label='Petit+20', c='tab:red', linewidth=lw)
        elif label == 'Pure SR':
            tmp2.plot('delta', 'pure_sr', ax=ax, label='Pure SR', c='tab:green', linewidth=lw)
        elif label == 'Pure SR (no intermediate features)':
            tmp2.plot('delta', 'pure_sr2', ax=ax, label='Pure SR (no intermediate features)', c='tab:purple', linewidth=lw)

        ax.set_xlabel(r'Interplanetary separation $\Delta$')
        ax.set_ylabel(r'Instability Time')
        leg = ax.legend(loc='upper left', frameon=True, fontsize=8, framealpha=1)
        for line in leg.get_lines():
            line.set_linewidth(3)

        ax.set_xlim(2, 13)
        ax.set_xticks(np.arange(2, 13), minor=True)
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='x', which='minor', direction='in')

        ax.set_ylim(0, 12)
        ax.set_yticks(np.arange(0, 12), minor=True)
        ax.set_yticks([0, 5, 10])
        ax.tick_params(axis='y', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')

    fig.tight_layout()
    fig.savefig(path)
    print('Saved to', path)


def official_plots(args):
    v = load_json(args.version_json)
    ms = v['pysr_model_selection_five_planet'] if 'pysr_model_selection_five_planet' in v else v['pysr_model_selection']
    main_path = f"five_planet_figures/v{v['nn_version']}_pysr{v['pysr_version']}_ms={ms}_N={args.N}_turbo_extrapolate.csv"
    cleaned = pd.read_csv(main_path)
    make_main_plot(cleaned, path='five_planet_figures/five_planet_main.pdf')

    pure_sr_path = f"five_planet_figures/v24880_pysr{v['pure_sr_version']}_ms={v['pure_sr_model_selection']}_N={args.N}_turbo_extrapolate.csv"
    pure_sr = pd.read_csv(pure_sr_path)
    cleaned['pure_sr'] = pure_sr['median']

    pure_sr2_path = f"five_planet_figures/v28114_pysr{v['pure_sr2_version']}_ms={v['pure_sr2_model_selection']}_N={args.N}_turbo_extrapolate.csv"
    pure_sr2 = pd.read_csv(pure_sr2_path)
    cleaned['pure_sr2'] = pure_sr2['median']
    make_separate_comparison_plot(cleaned, path='five_planet_figures/five_planet_all.pdf')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version_json', type=str, default='../official_versions.json')
    parser.add_argument('--N', type=int, default=5000)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    official_plots(args)
