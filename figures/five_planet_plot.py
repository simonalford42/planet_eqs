import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import time
import argparse
import json
plt.style.use('seaborn-darkgrid')
# plt.style.use('seaborn-ticks')
from sklearn.metrics import roc_auc_score


def _lighten(color, amount):
    """Blend `color` toward white by `amount` in [0, 1] (0=unchanged, 1=white)."""
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (1.0 - c) * amount)


def _missing_std_error(path, command):
    return ValueError(
        f"{path} is missing uncertainty columns. Recompute it with:\n"
        f"  {command}"
    )


def _attach_prediction_columns(cleaned, source, prefix, path, recompute_command=None, require_std=False):
    cleaned[prefix] = source['average']

    optional_columns = {
        'average_std': f'{prefix}_std',
        'l': f'{prefix}_l',
        'u': f'{prefix}_u',
        'll': f'{prefix}_ll',
        'uu': f'{prefix}_uu',
    }
    missing = [c for c in optional_columns if c not in source.columns]
    if require_std and missing:
        raise _missing_std_error(path, recompute_command)

    for src, dst in optional_columns.items():
        if src in source.columns:
            cleaned[dst] = source[src]


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def make_main_plot(cleaned, path=None):

    plt.rc('font', family='sans-serif')

    scale = 1
    lw = 0.9
    fig, ax = plt.subplots(figsize=(7*scale, 2.5*scale), dpi=300)

    tmp = cleaned
    tmp2 = tmp.query('true > 4 & delta > 5')

    eq_col = 'average'

    tmp.plot('delta', 'true', ax=ax, label='True', c='k', linewidth=lw)
    tmp2.plot('delta', eq_col, ax=ax, label='Distilled equations', c='tab:blue', linewidth=lw)
    ax.fill_between(tmp2['delta'], tmp2['l'], tmp2['u'], color='tab:blue', alpha=0.2, linewidth=0)
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
    plt.rc('font', family='sans-serif')

    scale = 1
    lw=0.9
    fig, axarr = plt.subplots(5, 1, figsize=(7*scale, 10.5 * scale), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.2)
    tmp = cleaned
    tmp2 = tmp.query('true > 4 & delta > 5')
    # tmp2 = tmp
    # decrease buffer around plot in image
    # plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05)


    eq_col = 'average'
    bnn_col = 'bnn_average'

    for i, label in enumerate(['Petit+20', 'Neural Network', 'Distilled Equations', 'Pure SR', 'Pure SR (no intermediate features)']):
        ax = axarr[i]

        tmp.plot('delta', 'true', ax=ax, label='True', c='k', linewidth=lw)
        if label == 'Neural Network':
            tmp2.plot('delta', bnn_col, ax=ax, label='Neural network', c='tab:orange', linewidth=lw)
            ax.fill_between(
                tmp2['delta'], tmp2['bnn_l'], tmp2['bnn_u'], color='tab:orange', alpha=0.2, linewidth=0)
        if label == 'Distilled Equations':
            tmp2.plot('delta', eq_col, ax=ax, label='Distilled equations', c='tab:blue', linewidth=lw)
            ax.fill_between(
                tmp2['delta'], tmp2['l'], tmp2['u'], color='tab:blue', alpha=0.2, linewidth=0)
        elif label == 'Petit+20':
            tmp.plot('delta', 'pperiodetitf', ax=ax, label='Petit+20', c='tab:red', linewidth=lw)
        elif label == 'Pure SR':
            tmp2.plot('delta', 'pure_sr', ax=ax, label='Pure SR', c='tab:green', linewidth=lw)
            if {'pure_sr_l', 'pure_sr_u'}.issubset(tmp2.columns):
                ax.fill_between(
                    tmp2['delta'], tmp2['pure_sr_l'], tmp2['pure_sr_u'],
                    color='tab:green', alpha=0.2, linewidth=0)
        elif label == 'Pure SR (no intermediate features)':
            tmp2.plot('delta', 'pure_sr2', ax=ax, label='Pure SR (no intermediate features)', c='tab:purple', linewidth=lw)
            if {'pure_sr2_l', 'pure_sr2_u'}.issubset(tmp2.columns):
                ax.fill_between(
                    tmp2['delta'], tmp2['pure_sr2_l'], tmp2['pure_sr2_u'],
                    color='tab:purple', alpha=0.2, linewidth=0)

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


def make_nn_eq_std_plot(cleaned, path, show_std_axis=False):
    """Two-panel plot: NN (mean ± std band) on top, Equations (mean ± std band) below.
    The std band uses a lighter shade of the line color via fill_between, with a
    thin red outline so its limits stay visible against the black 'True' line.
    If show_std_axis is True, also overlays the MC σ on a secondary y-axis
    scaled so σ=1.5 lines up with logT=5 on the LHS."""
    plt.rc('font', family='sans-serif')

    scale = 1
    lw = 0.7
    panels = [
        ('Neural network', 'bnn_average', 'bnn_u', 'bnn_l', 'bnn_average_std', 'tab:orange'),
        ('Distilled equations', 'average', 'u', 'l', 'average_std', 'tab:blue'),
    ]

    fig_height = 2.25 * len(panels)
    fig, axarr = plt.subplots(len(panels), 1, figsize=(7 * scale, fig_height * scale), dpi=300, sharex=True)
    axarr = np.atleast_1d(axarr)
    plt.subplots_adjust(hspace=0.15)

    tmp = cleaned.sort_values('delta').reset_index(drop=True)
    tmp2 = tmp.query('true > 4 & delta > 5')

    # Linear mapping for the RHS std axis: std=1.5 should sit at the same y as
    # logT=5 on the LHS. LHS spans 0..12, so std_max = 1.5 * 12/5 = 3.6.
    std_max = 1.5 * 12.0 / 5.0  # = 3.6

    for ax, (label, mean_col, lo_col, hi_col, std_col, color) in zip(axarr, panels):
        # If the CSV predates the average_std MC column, fall back to half the
        # ±1σ band width (l is 84th pct, u is 16th — a normal-equivalent σ).
        if show_std_axis and std_col not in tmp2.columns:
            tmp2 = tmp2.copy()
            tmp2[std_col] = (tmp2[lo_col] - tmp2[hi_col]) / 2.0

        tmp.plot('delta', 'true', ax=ax, label='True', c='k', linewidth=lw)
        # Tint the underlying black more strongly: use a lightened version of
        # the panel color so the fill itself stays pale, then crank alpha so
        # the fill dominates whatever it's over.
        band_color = _lighten(color, amount=0.55)
        ax.fill_between(
            tmp2['delta'], tmp2[lo_col], tmp2[hi_col],
            color=band_color, alpha=0.6, linewidth=0,
        )
        # Predicted-mean line goes on top of the band.
        tmp2.plot('delta', mean_col, ax=ax, label=label, c=color, linewidth=lw, zorder=4)
        # Thin red outline so the band edges stay readable when they overlap
        # the black 'True' line.
        # outline_kw = dict(color='red', linewidth=0.2, alpha=0.75, zorder=2.5)
        # ax.plot(tmp2['delta'], tmp2[lo_col], **outline_kw)
        # ax.plot(tmp2['delta'], tmp2[hi_col], **outline_kw)

        ax.set_ylabel(r'Instability Time')

        ax.set_xlim(2, 13)
        ax.set_xticks(np.arange(2, 13), minor=True)
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='x', which='minor', direction='in')

        ax.set_ylim(0, 12)
        ax.set_yticks([0, 5, 10])
        ax.set_yticks(np.arange(0, 12), minor=True)
        ax.tick_params(axis='y', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')

        if show_std_axis:
            # Secondary y-axis: Monte-Carlo std of the min-over-trios sampling
            # distribution (np.std(outs[i]) computed in five_planet.py).
            ax2 = ax.twinx()
            ax2.plot(
                tmp2['delta'], tmp2[std_col],
                color=color, linewidth=lw * 0.6, linestyle='--', alpha=0.85,
                label=f'{label} std',
            )
            ax2.set_ylim(0, 2* std_max)
            ax2.set_yticks([0, 1, 2])
            ax2.set_ylabel(r'Predicted $\sigma$')
            # Drive the background gridlines from the std axis (1, 2, 3)
            # instead of the LHS logT ticks, so the white lines span the full
            # width of the plot at std=1,2,3.
            ax.yaxis.grid(False, which='both')
            ax2.yaxis.grid(False)
            # Draw the std=1,2,3 gridlines directly on ax at the LHS-equivalent
            # y positions, so they sit on top of the seaborn-darkgrid grey
            # patch but below the data lines (zorder < 2).
            for s in (1, 2, 3):
                y_lhs = s * 12.0 / (2 * std_max)  # ax2 ylim is (0, 2*std_max)
                ax.axhline(y_lhs, color='white', linewidth=1, zorder=1)

            # Combined legend so the std line shows up alongside everything else.
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            leg = ax.legend(h1 + h2, l1 + l2, loc='upper left',
                            frameon=True, fontsize=8, framealpha=1)
        else:
            leg = ax.legend(loc='upper left', frameon=True, fontsize=8, framealpha=1)

        for line in leg.get_lines():
            line.set_linewidth(3)

    axarr[-1].set_xlabel(r'Interplanetary separation $\Delta$')

    fig.tight_layout()
    fig.savefig(path)
    print('Saved to', path)


def calculate_metrics(args):
    v = load_json(args.version_json)
    ms = v['pysr_model_selection']
    main_path = f"five_planet_figures/v{v['nn_version']}_pysr{v['pysr_version']}_ms={ms}_N={args.N}_turbo_extrapolate.csv"
    cleaned = pd.read_csv(main_path)
    # make_main_plot(cleaned, path='five_planet_figures/five_planet_main_eq29.pdf')

    pure_sr_path = f"five_planet_figures/v24880_pysr{v['pure_sr_version']}_ms={v['pure_sr_model_selection']}_N={args.N}_turbo_extrapolate.csv"
    pure_sr = pd.read_csv(pure_sr_path)
    _attach_prediction_columns(cleaned, pure_sr, 'pure_sr', pure_sr_path)

    pure_sr2_path = f"five_planet_figures/v28114_pysr{v['pure_sr2_version']}_ms={v['pure_sr2_model_selection']}_N={args.N}_turbo_extrapolate.csv"
    pure_sr2 = pd.read_csv(pure_sr2_path)
    _attach_prediction_columns(cleaned, pure_sr2, 'pure_sr2', pure_sr2_path)

    # calculate rmse for predictions where ground truth is between 4 and 9
    # also calculate accuracy for predicting stable (ground truth > 9) as well as accuracy predicting very stable (ground truth > 10)
    # also calculate AUC and bis
    cleaned[cleaned > 10] = 10
    cleaned[cleaned < 4] = 4

    for col in ['bnn_average', 'average', 'pperiodetitf', 'pure_sr', 'pure_sr2']:
        rmse = np.sqrt(np.mean((cleaned['true'] - cleaned[col])**2))
        exclude_stable_rmse = np.sqrt(np.mean((cleaned.query('true < 10')['true'] - cleaned.query('true < 10')[col])**2))
        # where delta < 8
        rmse_below8 = np.sqrt(np.mean((cleaned.query('true < 10 & delta < 8')['true'] - cleaned.query('true < 10 & delta < 8')[col])**2))
        rmse_above8 = np.sqrt(np.mean((cleaned.query('true < 10 & delta >= 8')['true'] - cleaned.query('true < 10 & delta >= 8')[col])**2))
        # acc = np.mean((cleaned['true'] >= 9) == (cleaned[col] >= 9))
        acc10 = np.mean((cleaned['true'] >= 10) == (cleaned[col] >= 10))
        # auc = roc_auc_score(cleaned['true'] >= 9, cleaned[col])
        # bias = np.me'true'])
        t = "\t"
        print(f'{col} {t}RMSE: {exclude_stable_rmse:.3f}, RMSE (< 8): {rmse_below8:.3f}, RMSE (>= 8): {rmse_above8:.3f}, Accuracy (> 10): {acc10:.3f}')



def official_plots(args):
    v = load_json(args.version_json)
    ms = v['pysr_model_selection']
    main_path = f"five_planet_figures/v{v['nn_version']}_pysr{v['pysr_version']}_ms={ms}_N={args.N}_turbo_extrapolate.csv"
    cleaned = pd.read_csv(main_path)
    make_main_plot(cleaned, path='five_planet_figures/five_planet_main.pdf')

    pure_sr_path = f"five_planet_figures/v24880_pysr{v['pure_sr_version']}_ms={v['pure_sr_model_selection']}_N={args.N}_turbo_extrapolate.csv"
    pure_sr = pd.read_csv(pure_sr_path)
    pure_sr_command = (
        f"python five_planet.py --pure_sr --pysr_version {v['pure_sr_version']} "
        f"--pysr_model_selection {v['pure_sr_model_selection']} --N {args.N} "
        "--turbo --extrapolate"
    )
    _attach_prediction_columns(
        cleaned, pure_sr, 'pure_sr', pure_sr_path,
        recompute_command=pure_sr_command, require_std=True,
    )

    pure_sr2_path = f"five_planet_figures/v28114_pysr{v['pure_sr2_version']}_ms={v['pure_sr2_model_selection']}_N={args.N}_turbo_extrapolate.csv"
    pure_sr2 = pd.read_csv(pure_sr2_path)
    pure_sr2_command = (
        f"python five_planet.py --version {v['pure_sr2_nn_version']} "
        f"--pysr_version {v['pure_sr2_version']} "
        f"--pysr_model_selection {v['pure_sr2_model_selection']} --N {args.N} "
        "--turbo --extrapolate"
    )
    _attach_prediction_columns(
        cleaned, pure_sr2, 'pure_sr2', pure_sr2_path,
        recompute_command=pure_sr2_command, require_std=True,
    )
    make_separate_comparison_plot(cleaned, path='five_planet_figures/five_planet_all.pdf')

    eq_std_version = v['eq_std_nn_version']
    ms = v['pysr_model_selection']
    csv_path = (
        f"five_planet_figures/v{v['nn_version']}_pysr{v['pysr_version']}"
        f"_ms={ms}_eqstd{eq_std_version}_N={args.N}_turbo_extrapolate.csv"
    )
    cleaned = pd.read_csv(csv_path)
    _attach_prediction_columns(
        cleaned, pure_sr, 'pure_sr', pure_sr_path,
        recompute_command=pure_sr_command, require_std=True,
    )
    _attach_prediction_columns(
        cleaned, pure_sr2, 'pure_sr2', pure_sr2_path,
        recompute_command=pure_sr2_command, require_std=True,
    )

    out_path = "five_planet_figures/five_planet_std.pdf"
    make_nn_eq_std_plot(cleaned, path=out_path, show_std_axis=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version_json', type=str, default='../official_versions.json')
    parser.add_argument('--N', type=int, default=5000)
    parser.add_argument('--metrics', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.metrics:
        calculate_metrics(args)
    else:
        official_plots(args)
