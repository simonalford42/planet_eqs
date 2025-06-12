import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
plt.style.use('seaborn-darkgrid')
# plt.style.use('seaborn-ticks')

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


def make_separate_comparison_plot(cleaned, path=None):
    plt.rc('font', family='serif')

    scale = 8
    lw=1.2
    fig, axarr = plt.subplots(4, 1, figsize=(scale, scale), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.2)
    tmp = cleaned
    # tmp2 = tmp.query('true > 4 & delta > 5')
    tmp2 = tmp
    # decrease buffer around plot in image
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05)


    for i, label in enumerate(['Petit+20', 'Neural Network', 'Distilled Equations', 'Pure SR']):
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

    if path == None:
        t = time.strftime('%Y%m%d_%H%M%S')
        path = f'five_planet_figures/five_planet_{t}.png'
    fig.savefig(path)
    print('Saved to', path)


if __name__ == '__main__':
    main_path = 'cur_plot_datasets/five_planet_figures/five_planet2_v24880_pysr11003_ms=29_N=5000_turbo_extrapolate_1749663354.199567.csv'
    cleaned = pd.read_csv(main_path)
    make_main_plot(cleaned, path='five_planet_figures/five_planet_main.pdf')

    pure_sr_path = 'cur_plot_datasets/five_planet_figures/five_planet2_v24880_pysr83941_ms=None_N=5000_turbo_extrapolate_1748898353.284526.csv'
    pure_sr = pd.read_csv(pure_sr_path)
    cleaned['pure_sr'] = pure_sr['median']
    make_separate_comparison_plot(cleaned, path='five_planet_figures/five_planet_all.pdf')


