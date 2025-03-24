import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
import time

try:
    plt.style.use('paper')
except:
    pass

basedir_bayes = '.'
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
    rgb = lambda x, y, z: np.array([x, y, z]).astype(np.float32)

    new_color.append(eval(elem[2]))

    shade += 1
    if shade == 5:
        colors.append(np.array(new_color))
        shade = 0
colors = np.array(colors)/255.0

def make_plot2(cleaned, path=None):

    plt.rc('font', family='serif')

    scale = 1
    lw = 1.2
    fig, ax = plt.subplots(figsize=(7*scale, 2*scale), dpi=400)
    # plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)

    tmp = cleaned
    tmp2 = tmp.query('true > 4 & delta > 5')

    # Ground truth
    tmp.plot('delta', 'true', ax=ax, label='True', c='k', linewidth=lw)

    # Distilled equations
    tmp.plot('delta', 'median', ax=ax, label='Distilled equations', c=colors[3, 3], linewidth=lw)
    # no error bars
    # ax.fill_between(
    #     tmp2['delta'], tmp2['l'], tmp2['u'], color=colors[3, [3]], alpha=0.2, linewidth=lw
    # )

    # Petit+20
    tmp.plot('delta', 'petitf', ax=ax, label='Petit+20', c=colors[0, 3], linewidth=lw)

    # Training range
    ax.annotate('Training range', (12, 4.5), fontsize=9)
    ax.plot([0, 14], [9, 9], '--k', linewidth=0.9)
    ax.plot([0, 14], [4, 4], '--k', linewidth=0.9)

    ax.set_xlim(1, 14)
    ax.set_ylim(0, 12)
    ax.set_xlabel(r'Interplanetary separation $\Delta$')
    ax.set_ylabel(r'Instability Time')

    leg = ax.legend(loc='upper left', frameon=True, fontsize=8, framealpha=1)
    for line in leg.get_lines():
        line.set_linewidth(3)

    major_ticks = [0, 5, 10]
    ax.set_yticks(major_ticks)
    ax.set_yticks(np.arange(1, 15), minor=True)
    ax.set_xticks(np.arange(1, 14), minor=True)
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='x', which='minor', direction='in')
    # ax.annotate('(a)', (-0.14, 1.05), xycoords='axes fraction', fontsize=10)

    if path is None:
        t = time.strftime('%Y%m%d_%H%M%S')
        path = f'five_planet_figures/five_planet2_{t}.png'

    fig.tight_layout()
    fig.savefig(path)
    print('Saved to', path)


def make_plot_separate(cleaned, path=None):
    plt.rc('font', family='serif')

    scale = 1
    lw=1.2
    fig, axarr = plt.subplots(3, 1, figsize=(9.5*scale, 8*scale), dpi=400, sharex=True)
    plt.subplots_adjust(hspace=0.2)
    tmp = cleaned
    tmp2 = tmp.query('true > 4 & delta > 5')
    # decrease buffer around plot in image
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)


    for i, label in enumerate(['BNN', 'Ours', 'Petit+20']):
        ax = axarr[i]

        tmp.plot('delta', 'true', ax=ax, label='True', c='k', linewidth=lw)
        if label == 'BNN':
            tmp.plot('delta', 'bnn_median', ax=ax, label='Neural network', c=colors[2, 3], linewidth=lw)
            ax.fill_between(
                tmp2['delta'], tmp2['bnn_l'], tmp2['bnn_u'], color=colors[2, [3]], alpha=0.2, linewidth=lw)
        if label == 'Ours':
            tmp.plot('delta', 'median', ax=ax, label='Distilled equations', c=colors[3, 3], linewidth=lw)
            ax.fill_between(
                tmp2['delta'], tmp2['l'], tmp2['u'], color=colors[3, [3]], alpha=0.2, linewidth=lw)
        elif label == 'Petit+20':
            tmp.plot('delta', 'petitf', ax=ax, label='Petit+20', c=colors[0, 3], linewidth=lw)

        if label != 'Petit+20':
            # ax.annotate('Training range', (12, 4.5))
            ax.annotate('Training range', (12, 4.5), fontsize=9)
            ax.plot([0, 14], [9, 9], '--k', linewidth=0.9)
            ax.plot([0, 14], [4, 4], '--k', linewidth=0.9)

        ax.set_xlim(1, 14)
        ax.set_ylim(0, 12)
        # ax.set_xlabel(r'$\Delta$')
        ax.set_xlabel(r'Interplanetary separation $\Delta$')
        ax.set_ylabel(r'Instability Time')
        leg = ax.legend(loc='upper left', frameon=True, fontsize=8, framealpha=1)
        for line in leg.get_lines():
            line.set_linewidth(3)

        major_ticks = [0, 5, 10]
        ax.set_yticks(major_ticks)
        ax.set_yticks(np.arange(1, 15), minor=True)
        ax.set_xticks(np.arange(1, 14), minor=True)
        ax.tick_params(axis='y', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.annotate(f'({chr(97 + i)})', (-0.14, 1.05), xycoords='axes fraction', fontsize=10)

    if path == None:
        # datetime in readable format
        t = time.strftime('%Y%m%d_%H%M%S')
        path = f'five_planet_figures/five_planet_{t}.png'
    fig.savefig(path)
    print('Saved to', path)


def make_plot(cleaned, version, pysr_version=None, t20=True, pysr_model_selection=None):
# +
# %matplotlib inline
    # plt.style.use('science')
    fig, axarr = plt.subplots(1, 1, figsize=(16/2,4/2), dpi=400, sharex=True)
    plt.subplots_adjust(hspace=0, wspace=0)
    ax = plt.gca()
    kwargs = dict(alpha=0.5,
                  markersize=5/4)
    tmp = cleaned#.query('true > 4')
    tmp.loc[:, 'xgb'] = tmp.loc[:, 'true'] * np.array(tmp['true'] <= 4.0) + tmp.loc[:, 'xgb'] * np.array(tmp['true'] > 4.0)
    tmp2 = tmp.query('true > 4 & delta > 5')
    ax.fill_between(
        tmp2['delta'], tmp2['l'], tmp2['u'], color=colors[2, [3]], alpha=0.2)
    # ax.fill_between(
        # tmp2['delta'], tmp2['ll'], tmp2['uu'], color=colors[2, [3]], alpha=0.1)

    tmp.plot(
        'delta', 'true', ax=ax,
        label='True',
        c='k'
    )
    if t20:
        tmp.plot(
            'delta', 'xgb', ax=ax,
            label='Modified T20',
            c=colors[1, 3]
        )
    # tmp.plot(
        # 'delta', 'petit', ax=ax,
        # label='Petit+20, no tuning',
# #     style='o',
# #     ms=5./4*0.3,
        # c=colors[3, 3]
    # )
    tmp.plot(
        'delta', 'petitf', ax=ax,
        label='Petit+20',
#     style='o',
#     ms=5./4*0.3,
        c=colors[0, 3]
    )
    tmp.plot(
        'delta', 'median', ax=ax,
        label='Ours',
#     style='o',ms=5./4*0.3, color=colors[2, 3]
        c=colors[2, 3]
    )
# xlim = ax.get_xlim()
    ax.plot([0, 14], [9, 9], '--k')
    ax.plot([0, 14], [4, 4], '--k')
    ax.annotate('Training range', (12, 4.5))
# ax.set_xlim(*xlim)
# # plt.plot()
    ax.set_xlim(1, 14)
    ax.set_ylim(0, 12)
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: r'$10^{%d}$'%(x,)))
    ax.set_xlabel(r'$\Delta$')
    ax.set_ylabel(r'Instability Time')
    plt.legend(loc='upper left',
               frameon=True, fontsize=8)

    s = '' if pysr_version is None else f'pysr={pysr_version}_'
    if pysr_version is not None and pysr_model_selection is not None:
        s += f'ms={pysr_model_selection}_'
    path = basedir_bayes + '/' + f'comparison_v{version}_' + s + '5planet.png'
    fig.savefig(path)
    print('Saved to', path)


# -



    plt.style.use('default')
    sns.set_style('white')
    plt.rc('font', family='serif')

    for key in 'median petitf'.split(' '):
        px = tmp['true']
        py = tmp[key]


        from scipy.stats import gaussian_kde

        mask = (px > 4)
        px = np.clip(px, 4, 9)
        ppx = px[mask]
        py = np.clip(py, 4, 9)
        ppy = py[mask]
        # bw = 0.2

        fig = plt.figure(figsize=(4, 4),
                         dpi=300,
                         constrained_layout=True)

        color = mcolors.to_rgba(colors[2, 3])
        g = sns.jointplot(x=ppx, y=ppy,
                        alpha=1.0,# ax=ax,
                        color=color,
                        s=5,
                        xlim=(3, 10),
                        ylim=(3, 10),
                        marginal_kws=dict(bins=15),
                       )
        ax = g.ax_joint

        ax.plot([4, 9], [4, 9], color='k')

        ## Errorbars:
        # if key == 'median':
            # upper = tmp['u']
            # lower = tmp['l']
            # upper = np.clip(upper[mask], 4, 9)
            # lower = np.clip(lower[mask], 4, 9)
            # upper = upper - ppy
            # lower = ppy - lower
            # plt.scatter

            # ax.errorbar(
                    # ppx,
                    # ppy,
                    # yerr=[lower, upper],
                    # fmt='o',
                    # ecolor=list(colors[2, 3]) + [0.2],
                    # ms=5,
                    # color=colors[2, 3]
                # )
        # plt.colorbar(im)

        #Try doing something like this: https://seaborn.pydata.org/examples/kde_ridgeplot.html
        #Stack all contours on the same axis. Have horizontal lines to help compare residuals.

        title = 'Ours' if key == 'median' else 'Petit+20'
        ax.set_xlabel('Truth')
        ax.set_ylabel('Predicted')
        plt.suptitle(title, y=1.0)
        plt.tight_layout()
        s = '' if pysr_version is None else f'pysr={pysr_version}_'
        path = basedir_bayes + f'/comparison_v{version}_{s}5planet_{key}.png'
        plt.savefig(path, dpi=300)
        print('Saved to', path)


if __name__ == '__main__':
    path = 'five_planet_figures/five_planet2_v24880_pysr11003_ms=30_N=5000_samps=10000_turbo'
    time = '1739990877.3044198'

    # path = 'five_planet_figures/five_planet2_v24880_pysr11003_ms=26_N=5000_samps=10000_turbo'
    # time = '1740510166.2667198'

    csv_path = f'cur_plot_datasets/{path}_{time}.csv'
    cleaned = pd.read_csv(csv_path)
    make_plot2(cleaned, path=path + '.png')
