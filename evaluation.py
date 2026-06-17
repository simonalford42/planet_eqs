import argparse
import os
import torch
import numpy as np
from spock_reg_model import load, load_with_pysr_f2
from tqdm.notebook import tqdm
import petit
from pure_sr_evaluation import pure_sr_predict_fn, get_pure_sr_results
from interpret import get_pysr_results
from utils import assert_equal, load_pickle, save_pickle
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from types import SimpleNamespace
from utils import load_json, load_pickle
from matplotlib import pyplot as plt
from plot_2d import plot_2d, draw_joint_panel, MAIN_COLOR, clipped
# to fix the fonts?
plt.rcParams.update(plt.rcParamsDefault)


###############################################################################
# HOW `get_data` SPLITS & SERVES THE DATA
#
# ┌───────────────────────────┬──────────────────────────────────────────────┐
# │ argument combination      │ what it loads / how it splits               │
# ├───────────────────────────┼──────────────────────────────────────────────┤
# │ train=True, plot=False    │                                           │
# │ (default)                 │ • load *resonant_dataset.pkl*               │
# │                           │ • two nested train_test_split calls         │
# │                           │      1) 90 % / 10 % → hold-out → *finalX*   │
# │                           │      2) 90 % of 90 % → 81 % train           │
# │                           │                        9 % validation       │
# │                           │ • returns:                                  │
# │                           │     dataloader  → 81 % “train”              │
# │                           │     test_loader →  9 % “val”                │
# ├───────────────────────────┼──────────────────────────────────────────────┤
# │ train=True, plot=True     │ same as above **but** validation loader is  │
# │                           │ swapped with the 10 % *finalX* hold-out →   │
# │                           │ returned as “test”.                         │
# ├───────────────────────────┼──────────────────────────────────────────────┤
# │ train=False, plot_random=True│ • load *random_dataset.pkl*              │
# │                              │ • no splitting; whole file becomes       │
# │                              │   first returned loader (“random test”). │
# │                              │ • second loader is empty.                │
# └───────────────────────────┴──────────────────────────────────────────────┘
#
# Anything with train_all=True loads the combined resonant+random set and
# treats the “random” flag vector `r` to mask out subsets when plotting.
###############################################################################

def get_dataloader(
    split: str = "val",
    petit=False,
):
    """
    Parameters
    ----------
    split : {'train', 'val', 'test', 'random'}
        Which loader to obtain (see table above).
    petit : bool
        If True, loads data for use by Petit

    Returns
    -------
    torch.utils.data.DataLoader
    """
    model = load(24880)

    if petit:
        kwargs = {
            # no op scalar
            'ssX' :  StandardScaler(with_mean=False, with_std=False),
            # the no op scalar will not do anything even if we train it,
            # but we need to train it so we can use it
            'train_ssX' : True,
        }
    else:
        kwargs = {}

    if split == "train":
        model.make_dataloaders(**kwargs)
        return model.train_dataloader()
    if split == "val":
        model.make_dataloaders(**kwargs)
        return model.val_dataloader()
    if split == "test":
        model.make_dataloaders(train=True, plot=True, **kwargs)
        return model._val_dataloader
    if split == "random":
        if not petit:
            # need to load ssX for the resonant dataset
            model.make_dataloaders(train=True)
            from copy import deepcopy as copy
            assert model.ssX is not None
            tmp_ssX = copy(model.ssX)
            kwargs = {'ssX': tmp_ssX}

        model.make_dataloaders(train=False, plot_random=True, **kwargs)
        return model._dataloader

    raise ValueError("split must be one of 'train', 'val', 'test', or 'random'")


def _torch_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def nn_prediction_fn(model, full=False):
    model.eval()
    device = _torch_device()
    model.to(device)
    def predict_fn(x):
        pred = model(x.to(device), noisy_val=False, deterministic=True).cpu().detach().numpy()
        if full:
            return pred  # [B, 2] with mean and std
        return pred[..., 0]

    return predict_fn


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
        rgb = lambda x, y, z: np.array([x, y, z]).astype(np.float32)

        new_color.append(eval(elem[2]))

        shade += 1
        if shade == 5:
            colors.append(np.array(new_color))
            shade = 0
    colors = np.array(colors)/255.0
    return colors


def comparison_figure(args):
    assert args.dataset in ['test', 'random']
    s = 'resonant' if args.dataset == 'test' else 'random'
    path = 'plots/' + get_model_name(args) + f'_comparison_{s}.png'

    truths, preds = get_truths_and_preds(args, clip=True, within_range=True)
    colors = get_colors()

    main_shade = 3

    main_color = colors[2, main_shade]
    off_color = colors[2, main_shade]

    plt.switch_backend('agg')

    plt.style.use('default')
    sns.set_style('white')
    plt.rc('font', family='serif')

    ppx = truths
    ppy = preds

    title = 'Our model'

    main_color = main_color.tolist()
    g = sns.jointplot(
        ppx, ppy,
        alpha=1,
        color=main_color,
        s=0.0,
        xlim=(3, 13),
        ylim=(3, 13),
        marginal_kws=dict(bins=15),
    )

    ax = g.ax_joint

    point_color = np.ones_like(ppy)
    point_color = (np.einsum('r,i->ir', main_color, point_color)
                   + np.einsum('r,i->ir', off_color, 1-point_color))

    ax.scatter(ppx, ppy, marker='o', c=point_color, alpha=0.1, s=10, edgecolors='none')
    ax.plot([4-3, 9+3], [4-3, 9+3], color='k')
    ax.set_xlim(3+0.9, 10-0.9)
    ax.set_ylim(3+0.9, 10-0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Truth')
    ax.set_ylabel('Predicted')
    plt.suptitle(title, y=1.0)
    plt.tight_layout()

    plt.savefig(path, dpi=300)
    print(f'Saved comparison to {path}')


def get_model_name(args):
    if args.eval_type == 'petit':
        return 'petit'
    elif args.eval_type == 'pure_sr':
        return f'pure_sr_{args.pysr_version}_{args.pysr_model_selection}'
    elif args.eval_type == 'nn':
        return f'nn_{args.version}'
    elif args.eval_type == 'pysr':
        return f'pysr_{args.version}_{args.pysr_version}_{args.pysr_model_selection}'


def _get_cache_path(args):
    model_name = get_model_name(args)
    return f'pickles/cache_truths_preds_{model_name}_{args.dataset}.pkl'


def _is_nn_eval_type(eval_type):
    return eval_type in ('nn', 'pysr')


def _nn_cache_args(version, dataset):
    return SimpleNamespace(eval_type='nn', version=version, dataset=dataset)


def _load_or_create_nn_cache_with_summaries(version, dataset):
    """Load the NN cache, upgrading it with summary_stats when needed."""
    cache_path = _get_cache_path(_nn_cache_args(version, dataset))
    if os.path.exists(cache_path):
        cached = load_pickle(cache_path)
        if 'truths_full' in cached and 'preds_full' in cached and 'summary_stats' in cached:
            return cached

    model = load(version=version)
    model.eval()
    device = _torch_device()
    model.to(device)

    dataloader = get_dataloader(dataset)
    truths = []
    preds = []
    summaries = []

    for x, y in tqdm(dataloader):
        with torch.no_grad():
            out = model(
                x.to(device),
                noisy_val=False,
                deterministic=True,
                return_intermediates=True,
            )
        truths.append(y.numpy())
        preds.append(out['prediction'].detach().cpu().numpy())
        summaries.append(out['summary_stats'].detach().cpu().numpy())

    truths = np.concatenate(truths)
    preds = np.concatenate(preds)
    summary_stats = np.concatenate(summaries)

    nan_mask = np.isnan(preds[:, 0])
    if np.sum(nan_mask) > 0:
        print('num nan:', np.sum(nan_mask), '/', len(preds))
    preds[nan_mask, 0] = 4

    cached = {
        'truths_full': truths,
        'preds_full': preds,
        'summary_stats': summary_stats,
    }
    save_pickle(cached, cache_path)
    print(f'Cached NN truths/preds/summaries to {cache_path}')
    return cached


def _eval_pysr_from_nn_cache(args, nn_cache, batch_size=4096):
    model = load_with_pysr_f2(
        version=args.version,
        pysr_version=args.pysr_version,
        pysr_model_selection=args.pysr_model_selection,
    )
    model.eval()
    device = _torch_device()
    model.to(device)

    summary_stats = torch.as_tensor(nn_cache['summary_stats'], dtype=torch.float32)
    preds = []

    with torch.no_grad():
        for i in range(0, len(summary_stats), batch_size):
            summary_batch = summary_stats[i:i + batch_size].to(device)
            mu, std = model.predict_instability(summary_batch)
            pred = torch.cat((mu, std), dim=1)
            preds.append(pred.cpu().numpy())

    return np.concatenate(preds)


def _replace_nan_predictions(preds, is_nn):
    if is_nn:
        nan_mask = np.isnan(preds[:, 0])
    else:
        nan_mask = np.isnan(preds)
    if np.sum(nan_mask) > 0:
        print('num nan:', np.sum(nan_mask), '/', len(preds))
    if is_nn:
        preds[nan_mask, 0] = 4
    else:
        preds[nan_mask] = 4


def _load_and_cache(args):
    """Load predictions and cache full-fidelity data.

    For NN models: preds is [N, 2] (mean, std), truths is [N, 2].
    For non-NN models: preds is [N], truths is [N, 2].
    """
    cache_path = _get_cache_path(args)
    if os.path.exists(cache_path):
        cached = load_pickle(cache_path)
        # support old caches that only have reduced data
        if 'truths_full' in cached:
            return cached['truths_full'], cached['preds_full']
        else:
            return cached['truths'], cached['preds']

    if args.eval_type == 'pysr' and args.version == 24880:
        nn_cache = _load_or_create_nn_cache_with_summaries(args.version, args.dataset)
        truths = nn_cache['truths_full']
        preds = _eval_pysr_from_nn_cache(args, nn_cache)
        assert_equal(truths.shape[0], preds.shape[0])
        _replace_nan_predictions(preds, is_nn=True)
        save_pickle({'truths_full': truths, 'preds_full': preds}, cache_path)
        print(f'Cached truths/preds to {cache_path}')
        return truths, preds

    is_petit = args.eval_type == 'petit'
    dataloader = get_dataloader(args.dataset, petit=is_petit)

    is_nn = _is_nn_eval_type(args.eval_type)
    if is_nn:
        predict_fn = get_prediction_fn(args, full=True)
    else:
        predict_fn = get_prediction_fn(args)

    truths = []
    preds = []
    for x, y in tqdm(dataloader):
        truths.append(y.numpy())
        preds.append(predict_fn(x))

    truths = np.concatenate(truths)  # [N, 2]
    preds = np.concatenate(preds)    # [N, 2] for NN, [N] for non-NN
    assert_equal(truths.shape[0], preds.shape[0])

    _replace_nan_predictions(preds, is_nn)

    save_pickle({'truths_full': truths, 'preds_full': preds}, cache_path)
    print(f'Cached truths/preds to {cache_path}')
    return truths, preds


def get_truths_and_preds(args, clip=True):
    truths_full, preds_full = _load_and_cache(args)

    # reduce to 1D: average truths, take mean prediction
    truths = np.average(truths_full, axis=1) if truths_full.ndim == 2 else truths_full
    preds = preds_full[:, 0] if preds_full.ndim == 2 else preds_full

    if clip:
        preds = np.clip(preds, 4, 9)

    return truths, preds


def get_full_truths_and_preds(args):
    """Return full-fidelity cached data for ll computation.

    Returns (truths, preds) where:
      - truths: [N, 2] numpy array (un-averaged)
      - preds: [N, 2] numpy array (mean + std)
    For non-NN models (pure_sr, petit), preds is 1D so we stack with std=1.
    """
    truths_full, preds_full = _load_and_cache(args)
    if preds_full.ndim == 1:
        preds_full = np.stack([preds_full, np.ones_like(preds_full)], axis=1)
    # elif args.eval_type in 'pysr':
    #     # replace stds with 1.0
    #     preds_full[:, 1] = 1.0

    return truths_full, preds_full

def safe_log_erf(x):
    base_mask = x < -1
    value_giving_zero = torch.zeros_like(x, device=x.device)
    x_under = torch.where(base_mask, x, value_giving_zero)
    x_over = torch.where(~base_mask, x, value_giving_zero)

    f_under = lambda x: (
         0.485660082730562*x + 0.643278438654541*torch.exp(x) +
         0.00200084619923262*x**3 - 0.643250926022749 - 0.955350621183745*x**2
    )
    f_over = lambda x: torch.log(1.0+torch.erf(x))

    return f_under(x_under) + f_over(x_over)


def lossfnc(testy, y, fixed_std=1.0):
    # testy: [B, 2] model output (mean, std)
    # y: [B, 2] batch of ground truth means. each input system has two
    #  simulations, one with a small initial perturbation, so the two means
    #  are samples from the distribution of instability times for that
    #  initial system
    # so we just sum over the loss for both of them.
    mu = testy[:, [0]]
    std = testy[:, [1]]
    # fix std for reported LL calculations
    std[...] = fixed_std

    var = std**2
    t_greater_9 = y >= 9

    regression_loss = -(y - mu)**2/(2*var)
    regression_loss += -torch.log(std)

    regression_loss += -safe_log_erf(
                (mu - 4)/(torch.sqrt(2*var))
            )

    classifier_loss = safe_log_erf(
                (mu - 9)/(torch.sqrt(2*var))
        )
    # note: gpt-5.3 discovered that the loss term was missing this normalization,
    # but it was not used in the original training.
    classifier_loss2 = (
        safe_log_erf((mu - 9) / torch.sqrt(2*var)) -
        safe_log_erf((mu - 4) / torch.sqrt(2*var))
    )

    safe_regression_loss = torch.where(
            ~torch.isfinite(regression_loss),
            -torch.ones_like(regression_loss)*100,
            regression_loss)
    safe_classifier_loss = torch.where(
            ~torch.isfinite(classifier_loss),
            -torch.ones_like(classifier_loss)*100,
            classifier_loss)
    safe_classifier_loss2 = torch.where(
            ~torch.isfinite(classifier_loss2),
            -torch.ones_like(classifier_loss2)*100,
            classifier_loss2)

    total_loss2 = (
        safe_regression_loss * (~t_greater_9) +
        safe_classifier_loss2 * ( t_greater_9)
    )
    total_loss2 = -(total_loss2.sum() / len(y)).item()

    total_regression = safe_regression_loss * (~t_greater_9)
    total_classifier = safe_classifier_loss * ( t_greater_9)
    total_regression = -(total_regression.sum() / len(y)).item()
    total_classifier = -(total_classifier.sum() / len(y)).item()
    avg_std = std.mean().item()

    total_loss = total_regression + total_classifier

    return total_loss, total_regression, total_classifier, avg_std, total_loss2


def calculate_ll(args, fixed_std=1.0):
    """Calculate loss using cached full predictions. Returns dict."""
    # if args.eval_type == 'petit':
        # return dict(loss=np.nan, reg_loss=np.nan, cls_loss=np.nan)
    truths_full, preds_full = get_full_truths_and_preds(args)
    preds_t = torch.from_numpy(preds_full)
    truths_t = torch.from_numpy(truths_full)

    total_loss, reg, cls, _, total_loss2 = lossfnc(preds_t, truths_t, fixed_std=fixed_std)
    # print(f'Loss for {args.dataset} dataset: {total_loss:.4f}  reg={reg:.4f}  cls={cls:.4f}')
    return dict(ll=-total_loss, reg_ll=-reg, cls_ll=-cls, full_ll=-total_loss2)


def calculate_metrics(args, plot=False, fixed_std=1.0):
    truths, preds = get_truths_and_preds(args)

    # calculate rmse excluding stable systems
    pred_stable = preds >= 9
    true_stable = truths >= 9
    exclude_stable_preds = preds[~true_stable]
    exclude_stable_truths = truths[~true_stable]
    rmse = np.average(np.square(exclude_stable_truths - exclude_stable_preds))**0.5
    full_rmse = np.average(np.square(truths - preds))**0.5

    acc = np.mean((preds >= 9) == (truths >= 9))
    bias = np.mean(preds[truths < 9] - truths[truths < 9])
    small_bias = np.mean(preds[truths < 5.5] - truths[truths < 5.5])
    large_unstable = (truths > 7.5) & (truths < 9)
    large_bias = np.mean(preds[large_unstable] - truths[large_unstable])

    # positive = unstable
    fpr = np.mean(~pred_stable[true_stable])       # falsely predict unstable among stable
    fnr = np.mean(pred_stable[~true_stable])           # falsely predict stable among unstable
    true_unstable = ~true_stable
    unstable_score = -preds
    roc_auc = (
        roc_auc_score(true_unstable, unstable_score)
        if len(np.unique(true_unstable)) == 2
        else np.nan
    )
    # print(f'RMSE: {rmse:.4f}, Accuracy: {acc:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}')
    if plot:
        plot_path = get_2d_plot_path(args)
        plot_2d(preds, truths, plot_path)
        print(f'Saved 2D comparison plot to {plot_path}')

        stable_truths = truths >= 9
        if len(np.unique(stable_truths)) == 2:
            auc = roc_auc_score(stable_truths, preds)
            plot_roc_curve(args, stable_truths, preds, auc)
        else:
            print(f'Skipping ROC plot for {get_model_name(args)} ({args.dataset}): only one class present')
        # plot_rmse_threshold_curve(args, truths, preds)

    lls = calculate_ll(args, fixed_std=fixed_std)
    ll = lls['ll']
    full_ll = lls['full_ll']

    return {
        'full_rmse': full_rmse,
        'rmse': rmse,
        'acc': acc,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'fnr': fnr,
        'll': ll,
        'full_ll': full_ll,
        'bias': bias,
        'small_bias': small_bias,
        'large_bias': large_bias,
    }


def _format_official_latex_row(latex_name, metrics, ll):
    """Format columns as RMSE/Acc/LL/ROC/FPR/FNR/bias."""
    return (
        f'{latex_name} & {metrics["rmse"]:.2f} & {metrics["acc"]:.2f} & '
        f'{ll} & {metrics["roc_auc"]:.2f} & {metrics["fpr"]:.2f} & '
        f'{metrics["fnr"]:.2f} & {metrics["bias"]:.2f} \\\\'
    )


def single_class_acc(truths, preds):
    thresholds = np.sort(np.unique(preds))[::-1]  # sweep high -> low
    stable_truths = truths >= 9

    def accuracy_at_threshold(t):
        predicted_stable = preds >= t
        acc = np.mean(predicted_stable == stable_truths)
        return acc
    scores = [accuracy_at_threshold(t) for t in thresholds]
    best_threshold = thresholds[np.argmax(scores)]
    return best_threshold, np.max(scores)


def roc_curve_from_scratch(args):
    """
    Educational implementation of ROC curve + AUC.

    The key idea: instead of one fixed threshold (pred >= 9 → "stable"),
    we sweep the threshold across all possible values. At each threshold t:
      - predict stable if pred >= t
      - compute TPR = fraction of true stables we correctly caught
      - compute FPR = fraction of true unstables we wrongly called stable
    Plotting TPR vs FPR traces the ROC curve.
    AUC is just the area under that curve (higher = better; 0.5 = random).
    """
    truths, preds = get_truths_and_preds(args, clip=False, exclude_stable=False)
    stable_truths = truths >= 9

    thresholds = np.sort(np.unique(preds))[::-1]  # sweep high -> low

    tprs = []
    fprs = []
    for t in thresholds:
        predicted_stable = preds >= t
        tpr = np.sum(stable_truths & predicted_stable) / (np.sum(stable_truths) + 1e-8)
        fpr = np.sum(~stable_truths & predicted_stable) / (np.sum(~stable_truths) + 1e-8)
        tprs.append(tpr)
        fprs.append(fpr)

    tprs = np.array(tprs)
    fprs = np.array(fprs)
    auc = np.trapz(tprs, fprs)  # area under curve via trapezoidal rule

    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    ax.plot(fprs, tprs, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve (from scratch) — {get_model_name(args)} ({args.dataset})')
    ax.legend()

    path = f'plots/roc_scratch_{get_model_name(args)}_{args.dataset}.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'AUC = {auc:.4f}')
    print(f'Saved ROC curve to {path}')
    return auc


def plot_roc_curve(args, stable_truths, preds, auc):
    fpr, tpr, _ = roc_curve(stable_truths, preds)

    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {get_model_name(args)} ({args.dataset})')
    ax.legend()

    path = get_roc_plot_path(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved ROC curve to {path}')


def plot_rmse_threshold_curve(args, truths, preds):
    unstable = truths < 9
    errors = np.sqrt(np.square(truths[unstable] - preds[unstable]))
    if len(errors) == 0:
        print(f'Skipping RMSE threshold plot for {get_model_name(args)} ({args.dataset}): no unstable systems')
        return

    thresholds = np.linspace(0, 5, 501)
    pct_below = np.array([np.mean(errors <= threshold) for threshold in thresholds]) * 100

    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    ax.plot(thresholds, pct_below)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 100)
    ax.set_xlabel('RMSE threshold')
    ax.set_ylabel('Percent below threshold')
    ax.set_title(f'RMSE threshold — {get_model_name(args)} ({args.dataset})')
    fig.tight_layout()

    path = get_rmse_threshold_plot_path(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved RMSE threshold plot to {path}')




def const_predict_fn(const):
    return lambda *_: const


def calculate_all_and_store(args):
    path = get_results_path(args)
    results = {}
    # datasets = ['val', 'test', 'random']
    datasets = ['val']

    for dataset in datasets:
        print('Calculating for dataset', dataset)
        print()
        args.dataset = dataset

        if args.eval_type in ['nn', 'petit']:
            metrics = calculate_metrics(args)
            results[dataset] = metrics
        else:
            if args.eval_type == 'pure_sr':
                sr_results = get_pure_sr_results(args.pysr_version)
            else:
                assert args.eval_type == 'pysr'
                sr_results = get_pysr_results(args.pysr_version)

            dataset_results = {}
            if args.pysr_model_selection is None:
                for c in sr_results['complexity']:
                    args.pysr_model_selection = c
                    print('Calculating for complexity', c)
                    metrics = calculate_metrics(args)
                    dataset_results[c] = metrics
            else:
                metrics = calculate_metrics(args)
                dataset_results = metrics

            results[dataset] = dataset_results
    save_pickle(results, path)


def get_results_path(args):
    results_tag = ''
    if args.version is not None:
        results_tag += f'_{args.version}'
    if args.pysr_version is not None:
        results_tag += f'_{args.pysr_version}'

    filename = f'pickles/{args.eval_type}_results_all{results_tag}.pkl'
    return filename


def get_2d_plot_path(args):
    return f'plots/2d/{get_model_name(args)}_{args.dataset}.png'


def get_roc_plot_path(args):
    return f'plots/roc/{get_model_name(args)}_{args.dataset}.png'


def get_rmse_threshold_plot_path(args):
    return f'plots/rmse_threshold/{get_model_name(args)}_{args.dataset}.png'


def get_prediction_fn(args, full=False):
    if args.eval_type == 'petit':
        return petit.tsurv
    elif args.eval_type == 'pure_sr':
        pure_sr_results = get_pure_sr_results(args.pysr_version)
        return pure_sr_predict_fn(pure_sr_results, int(args.pysr_model_selection))
    elif args.eval_type == 'nn':
        model = load(version=args.version)
        return nn_prediction_fn(model, full=full)
    elif args.eval_type == 'pysr':
        model = load_with_pysr_f2(version=args.version, pysr_version=args.pysr_version, pysr_model_selection=args.pysr_model_selection)
        return nn_prediction_fn(model, full=full)
    else:
        raise ValueError(f"Unknown eval_type: {args.eval_type}")


def official_calculations(version_json='official_versions.json'):
    v = load_json(version_json)

    models = [
        ('nn', 'Neural network', dict(eval_type='nn', version=v['nn_version'])),
        ('distilled_eqs', 'Ours', dict(eval_type='pysr', version=v['nn_version'],
                                       pysr_version=v['pysr_version'],
                                       pysr_model_selection=str(v['pysr_model_selection']))),
        ('petit', 'Petit+ 2020', dict(eval_type='petit')),
        ('pure_sr', 'Pure SR', dict(eval_type='pure_sr',
                                    pysr_version=v['pure_sr_version'],
                                    pysr_model_selection=str(v['pure_sr_model_selection']))),
        ('pure_sr2', 'Pure SR 2', dict(eval_type='pysr', version=v['pure_sr2_nn_version'],
                                       pysr_version=v['pure_sr2_version'],
                                       pysr_model_selection=str(v['pure_sr2_model_selection']))),
        # NN trained to predict just from intermediate features
        # ('nn2', 'NN 2', dict(eval_type='nn', version=28114)),
    ]

    datasets = [('train', 'Train'), ('val', 'Val'), ('test', 'test'), ('random', 'random')]
    all_results = {}

    for dataset, _ in datasets:
        for name, _, cfg in models:
            all_results.setdefault(name, {})
            args = SimpleNamespace(dataset=dataset, **cfg)
            fixed_std = 1.47 if name in {'distilled_eqs', 'pure_sr', 'pure_sr2'} else 1.0
            metrics = calculate_metrics(args, fixed_std=fixed_std)
            all_results[name][dataset] = metrics

    # for dataset, label in datasets:
    #     print(label)
    #     for name, _, _ in models:
    #         metrics = all_results[name][dataset]
    #         print(f'[{name:13s}] '
    #             f'Unstable RMSE={metrics["rmse"]:.2f} Full RMSE={metrics["full_rmse"]:.2f} Acc={metrics["acc"]:.2f}  '
    #             f'ROC-AUC={metrics["roc_auc"]:.2f} FPR (positive=unstable) ={metrics["fpr"]:.2f}  FNR={metrics["fnr"]:.2f} '
    #             f'LL={metrics["ll"]:.2f} Full LL={metrics["full_ll"]:.2f}, bias={metrics["bias"]:.2f}'
    #             f' (bias below 5.5={metrics["small_bias"]:.2f}, bias above 7.5={metrics["large_bias"]:.2f})'
    #         )

    print('LaTeX rows')
    print('% columns: RMSE & Acc & LL & ROC & FPR & FNR & bias')
    for dataset, label in [('test', 'resonant'), ('random', 'random')]:
        print(label)
        for name, latex_name, _ in models:
            metrics = all_results[name][dataset]
            ll = '--' if name == 'petit' else f'{metrics["ll"]:.2f}'
            print(_format_official_latex_row(latex_name, metrics, ll))

    return all_results



def _plot_official_2d_grid(columns, output, max_points=35000, seed=0, dpi=600,
                           scale=0.8, layout_ncols=None, rows=None,
                           clip_truths=True, clip_preds=True,
                           show_rmse_in_title=False, wrap_columns=False):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc('font', family='sans-serif')

    if rows is None:
        rows = [
            ('Resonant test set', 'test'),
            ('Random test set', 'random'),
        ]

    if layout_ncols is None:
        layout_ncols = len(columns)

    if wrap_columns:
        n_panels = len(rows) * len(columns)
        n_grid_rows = int(np.ceil(n_panels / layout_ncols))
    else:
        n_grid_rows = len(rows)

    fig = plt.figure(figsize=(scale*6*layout_ncols, scale*6*n_grid_rows), dpi=dpi)
    hspace = 0.3 if wrap_columns else 0.15
    grid = fig.add_gridspec(n_grid_rows, layout_ncols, hspace=hspace, wspace=0.15)
    rng = np.random.default_rng(seed)

    for r, (row_label, dataset) in enumerate(rows):
        for c, (col_title, cfg) in enumerate(columns):
            if wrap_columns:
                panel_ix = r * len(columns) + c
                grid_r, grid_c = divmod(panel_ix, layout_ncols)
            else:
                grid_r, grid_c = r, c
            args = SimpleNamespace(dataset=dataset, **cfg)
            truths_full, preds_full = _load_and_cache(args)
            truths = truths_full.reshape(-1)
            preds = preds_full[:, 0] if preds_full.ndim == 2 else preds_full
            preds = np.repeat(preds, 2)
            truths = clipped(truths) if clip_truths else truths
            preds = clipped(preds) if clip_preds else preds

            title = col_title if r == len(rows) - 1 or wrap_columns else ''
            if title and show_rmse_in_title:
                truths_avg = (
                    np.average(truths_full, axis=1)
                    if truths_full.ndim == 2 else truths_full
                )
                preds_avg = (
                    preds_full[:, 0] if preds_full.ndim == 2 else preds_full
                )
                if clip_preds:
                    preds_avg = clipped(preds_avg)
                unstable = truths_avg < 9
                rmse = np.sqrt(np.mean((preds_avg[unstable] - truths_avg[unstable])**2))
                title = f'{title}, RMSE={rmse:.2f}'
            ax = draw_joint_panel(fig, grid[grid_r, grid_c], truths, preds,
                                  title=title, color=MAIN_COLOR,
                                  max_points=max_points, rng=rng,
                                  show_bias=True)
            for collection in ax.collections:
                collection.set_alpha(0.35)
                collection.set_rasterized(False)
            ax.set_box_aspect(1)
            ax.title.set_fontweight('normal')
            if c == 0 and not wrap_columns:
                ax.annotate(row_label, xy=(-0.32, 0.5),
                            xycoords='axes fraction',
                            ha='center', va='center', rotation=90,
                            fontsize=13)

    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f'Saved official 2D grid figure to {output}')
    return output


def _official_2d_grids(v, output='plots/2d_grid.png',
                       complexities=(1, 3, 7, 14, 26),
                       max_points=None, seed=0, dpi=600):
    """Save the official 2D grid comparing selected equation complexities.

    The first two rows show the 2D truth/prediction panels. The third row
    shows residual trends for the additive terms in the complexity-26 equation.
    """
    complexity_columns = [
        (f'Complexity {complexity}',
         dict(eval_type='pysr', version=v['nn_version'],
              pysr_version=11003,
              pysr_model_selection=str(complexity)))
        for complexity in complexities
    ]
    complexity_columns.append(
        ('Neural network', dict(eval_type='nn', version=v['nn_version']))
    )

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc('font', family='sans-serif')

    scale = 0.8
    fig = plt.figure(figsize=(scale * 18, scale * 15), dpi=dpi)
    top_bottom_grid = fig.add_gridspec(
        2, 1,
        height_ratios=[2.7, 1.0],
        hspace=0.25,
    )
    top_grid = top_bottom_grid[0, 0].subgridspec(
        2, 12,
        height_ratios=[1.2, 1.2],
        hspace=0.35,
        wspace=0,
    )
    rng = np.random.default_rng(seed)

    def square_joint_panel(ax):
        ax.set_box_aspect(1)
        fig.canvas.draw()

        ax_histx = getattr(ax, '_joint_histx', None)
        ax_histy = getattr(ax, '_joint_histy', None)
        if ax_histx is None or ax_histy is None:
            return

        ax_pos = ax.get_position()
        histx_pos = ax_histx.get_position()
        histy_pos = ax_histy.get_position()
        top_gap = 0.01
        side_gap = 0.01

        ax_histx.set_position([
            ax_pos.x0,
            ax_pos.y1 + top_gap,
            ax_pos.width,
            histx_pos.height,
        ])
        ax_histy.set_position([
            ax_pos.x1 + side_gap,
            ax_pos.y0,
            histy_pos.width,
            ax_pos.height,
        ])

    for panel_ix, (col_title, cfg) in enumerate(complexity_columns):
        grid_r = panel_ix // 3
        grid_c = (panel_ix % 3) * 4
        args = SimpleNamespace(dataset='test', **cfg)
        truths_full, preds_full = _load_and_cache(args)
        truths = truths_full.reshape(-1)
        preds = preds_full[:, 0] if preds_full.ndim == 2 else preds_full
        preds = np.repeat(preds, 2)
        truths = clipped(truths)
        preds = clipped(preds)

        truths_avg = (
            np.average(truths_full, axis=1)
            if truths_full.ndim == 2 else truths_full
        )
        preds_avg = preds_full[:, 0] if preds_full.ndim == 2 else preds_full
        preds_avg = clipped(preds_avg)
        unstable = truths_avg < 9
        rmse = np.sqrt(np.mean((preds_avg[unstable] - truths_avg[unstable])**2))
        title = f'{col_title}, RMSE={rmse:.2f}'

        ax = draw_joint_panel(
            fig,
            top_grid[grid_r, grid_c:grid_c + 4],
            truths,
            preds,
            title=title,
            color=MAIN_COLOR,
            max_points=max_points,
            rng=rng,
            show_bias=True,
        )
        for collection in ax.collections:
            collection.set_alpha(0.15)
            collection.set_rasterized(False)
        ax.title.set_fontweight('normal')
        ax.title.set_y(-0.38)
        square_joint_panel(ax)

    bins = _official_subterm_bins(v)
    yticks = np.arange(-1.0, 2.01, 0.5)
    bottom_grid = top_bottom_grid[1, 0].subgridspec(1, 4, wspace=0.28)
    panels = [
        (
            'gated_quiet_term',
            r'$0.084^{\sigma_1}[\sigma_6^{0.36}(\sigma_2+\sigma_4)]^{-0.31}$',
        ),
        (
            'gated_sinusoid_term',
            r'$0.084^{\sigma_1}[-\sin(\mu_2)]$',
        ),
        (
            'gated_mass_eccentricity_term',
            r'$0.084^{\sigma_1}1.2^{-\mu_1}(\mu_7-\sigma_8)$',
        ),
        (
            'eq_pred',
            r'Equation predicted $\log_{10} T_{\rm eq}$',
        ),
    ]
    for i, (quantity, xlabel) in enumerate(panels):
        ax = fig.add_subplot(bottom_grid[0, i])
        panel = bins[bins['quantity'] == quantity].sort_values('x')
        ax.plot(panel['x'], panel['residual'], marker='o', lw=2,
                color='#294475',
                label=r'$\langle \log_{10} T_{\rm eq}-\log_{10} T_{\rm inst}\rangle$')
        ax.axhline(0, color='0.75', lw=0.8)
        ax.set_ylim(-1, 2)
        ax.set_yticks(yticks)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(
                r'$\log_{10} T_{\rm eq}-\log_{10} T_{\rm inst}$'
            )
        else:
            ax.set_yticklabels([])
        ax.grid(True, color='0.9')
        ax.set_box_aspect(1)

    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f'Saved official 2D grid figure to {output}')
    return output


def official_figures(version_json='official_versions.json',
                     max_points=35000, seed=0, dpi=600):
    """Generate official figure set."""
    v = load_json(version_json)

    _official_2d_grids(v, output="plots/2d_grid.pdf", max_points=max_points,
                       seed=seed, dpi=dpi)

    _official_std_figure(v, dpi=dpi)
    _official_roc_figure(v, dpi=dpi)
    _official_rmse_acc_plot(version_json=version_json)


def _complexity26_additive_terms(summary_stats):
    """Additive components of the official complexity-26 equation.

    summary_stats is [N, 20] = [mu_0..mu_9, sigma_0..sigma_9].
    The official equation is

      T_eq = 3.6591659 + 0.0841594^sigma1 *
             (quiet - sin(mu2) + (mu7 - sigma8)/1.2004135^mu1)
    """
    mu1 = summary_stats[:, 1]
    mu2 = summary_stats[:, 2]
    mu7 = summary_stats[:, 7]
    sigma1 = summary_stats[:, 11]
    sigma2 = summary_stats[:, 12]
    sigma4 = summary_stats[:, 14]
    sigma6 = summary_stats[:, 16]
    sigma8 = summary_stats[:, 18]

    gate = 0.0841594**sigma1
    quiet = (sigma6**0.35633504 * (sigma2 + sigma4))**(-0.3054036)
    sinusoid = -np.sin(mu2)
    mass_eccentricity = (mu7 - sigma8) / (1.2004135**mu1)

    return {
        'gated_quiet_term': gate * quiet,
        'gated_sinusoid_term': gate * sinusoid,
        'gated_mass_eccentricity_term': gate * mass_eccentricity,
    }


def _binned_equation_error(truths, preds, x, quantity, nbins=12):
    data = pd.DataFrame({
        'truth': truths,
        'pred': preds,
        'x': x,
    }).replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data['truth'] < 9]
    data['bin'] = pd.qcut(data['x'], q=nbins, duplicates='drop')

    rows = []
    for interval, group in data.groupby('bin', observed=True):
        residual = group['pred'] - group['truth']
        rows.append({
            'quantity': quantity,
            'x': group['x'].median(),
            'x_lo': interval.left,
            'x_hi': interval.right,
            'n': len(group),
            'rmse': np.sqrt(np.mean(residual**2)),
            'residual': np.mean(residual),
        })
    return pd.DataFrame(rows)


def _official_subterm_bins(v):
    nn_cache = _load_or_create_nn_cache_with_summaries(v['nn_version'], 'test')
    eq_args = SimpleNamespace(
        eval_type='pysr',
        version=v['nn_version'],
        pysr_version=v['pysr_version'],
        pysr_model_selection=str(v['pysr_model_selection']),
        dataset='test',
    )
    truths_full, preds_full = _load_and_cache(eq_args)
    truths = np.average(truths_full, axis=1) if truths_full.ndim == 2 else truths_full
    preds = preds_full[:, 0] if preds_full.ndim == 2 else preds_full
    preds = clipped(preds)

    term_values = _complexity26_additive_terms(nn_cache['summary_stats'])
    term_values['eq_pred'] = preds
    return pd.concat([
        _binned_equation_error(truths, preds, values, name)
        for name, values in term_values.items()
    ], ignore_index=True)


def _official_subterm_figure(
    v,
    output='plots/subterm_error.pdf',
    csv_output='pickles/subterm_error.csv',
    dpi=300,
):
    """Official resonant diagnostic for equation compression.

    Plots binned equation RMSE and mean residual against the additive terms
    of the complexity-26 equation and against the equation prediction itself.
    """
    bins = _official_subterm_bins(v)

    os.makedirs(os.path.dirname(csv_output) or '.', exist_ok=True)
    bins.to_csv(csv_output, index=False)

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc('font', family='sans-serif')

    yticks = np.arange(-1.0, 2.01, 0.5)
    fig, axes = plt.subplots(1, 4, figsize=(17.8, 3.9), dpi=dpi, sharey=True)
    panels = [
        (
            'gated_quiet_term',
            r'$0.084^{\sigma_1}[\sigma_6^{0.36}(\sigma_2+\sigma_4)]^{-0.31}$',
            'Gated variability term',
        ),
        (
            'gated_sinusoid_term',
            r'$0.084^{\sigma_1}[-\sin(\mu_2)]$',
            'Gated sinusoid term',
        ),
        (
            'gated_mass_eccentricity_term',
            r'$0.084^{\sigma_1}1.2^{-\mu_1}(\mu_7-\sigma_8)$',
            'Gated mass/eccentricity term',
        ),
        (
            'eq_pred',
            r'Equation predicted $\log_{10} T_{\rm eq}$',
            'Error vs predicted instability time',
        ),
    ]
    for ax, (quantity, xlabel, title) in zip(axes, panels):
        panel = bins[bins['quantity'] == quantity].sort_values('x')
        ax.plot(panel['x'], panel['residual'], marker='o', lw=2,
                color='#294475',
                label=r'$\langle \log_{10} T_{\rm eq}-\log_{10} T_{\rm inst}\rangle$')
        ax.axhline(0, color='0.75', lw=0.8)
        ax.set_ylim(-1, 2)
        ax.set_yticks(yticks)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(
            r'$\log_{10} T_{\rm eq}-\log_{10} T_{\rm inst}$'
        )
        ax.grid(True, color='0.9')
        ax.legend(frameon=False, loc='upper left', fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(output, bbox_inches='tight', dpi=dpi)
    png_output = os.path.splitext(output)[0] + '.png'
    fig.savefig(png_output, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f'Saved official resonant equation-error diagnostic to {output}')
    return output


def _official_rmse_acc_table(version_json='official_versions.json',
                             period_metrics_path='pickles/period_ratio_official_metrics.pkl',
                             metrics=None):
    """Combine resonant/random official metrics with period-ratio grid metrics."""
    if metrics is None:
        metrics = official_calculations(version_json=version_json)

    period_ratio_official = load_pickle(period_metrics_path)

    model_name_map = {
        'nn': 'Neural network',
        'distilled_eqs': 'Distilled equations',
        'pure_sr': 'Pure SR',
        'pure_sr2': 'Pure SR 2',
        'petit': 'Petit+2020',
    }

    splits = ['test', 'random']
    metrics_to_keep = ['rmse', 'acc', 'fpr', 'fnr']
    rows = {}
    for model_key, display_name in model_name_map.items():
        if model_key not in metrics:
            continue
        rows[display_name] = {
            (metric, split): metrics[model_key][split][metric]
            for split in splits
            for metric in metrics_to_keep
        }

    result_table = pd.DataFrame.from_dict(rows, orient='index')
    result_table.columns = pd.MultiIndex.from_tuples(
        result_table.columns,
        names=['metric', 'split'],
    )

    for model_key, display_name in model_name_map.items():
        if model_key not in period_ratio_official:
            continue
        grid_metrics = period_ratio_official[model_key]
        if display_name not in result_table.index:
            result_table.loc[display_name, :] = np.nan
        result_table.loc[display_name, ('rmse', 'grid')] = grid_metrics['unstable_rmse']
        result_table.loc[display_name, ('acc', 'grid')] = grid_metrics['acc']
        result_table.loc[display_name, ('fpr', 'grid')] = grid_metrics['fpr']
        result_table.loc[display_name, ('fnr', 'grid')] = grid_metrics['fnr']

    split_order = ['test', 'random', 'grid']
    metric_order = ['rmse', 'acc', 'fpr', 'fnr']
    ordered_columns = [
        (metric, split)
        for metric in metric_order
        for split in split_order
        if (metric, split) in result_table.columns
    ]
    return result_table.reindex(
        columns=pd.MultiIndex.from_tuples(ordered_columns, names=['metric', 'split'])
    )


def _official_rmse_acc_plot(version_json='official_versions.json',
                            period_metrics_path='pickles/period_ratio_official_metrics.pkl',
                            output='graphics/rmse_acc_plot.svg',
                            metrics=None):
    """Create the rebuttal RMSE/accuracy summary plot."""
    result_table = _official_rmse_acc_table(
        version_json=version_json,
        period_metrics_path=period_metrics_path,
        metrics=metrics,
    )

    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

    methods = ['Neural network', 'Distilled equations', 'Pure SR', 'Petit+2020']
    rmse_datasets = ['Resonant', 'Random', 'Grid']
    acc_datasets = ['Resonant', 'Random', 'Grid']

    def series_for(method, metric):
        split_lookup = [('test', 'Resonant'), ('random', 'Random'), ('grid', 'Grid')]
        return [result_table.loc[method, (metric, split)] for split, _ in split_lookup]

    rmse_dict = {method: series_for(method, 'rmse') for method in methods}
    acc_dict = {method: series_for(method, 'acc') for method in methods}

    cmap = plt.get_cmap('tab10')
    method_colors = dict(zip(methods, [cmap(i) for i in [1, 0, 2, 3]]))

    scale = 0.8
    fig, (ax_rmse, ax_acc) = plt.subplots(1, 2, figsize=(scale * 15, scale * 4))
    fig.subplots_adjust(right=0.88, wspace=0.35)

    bw = 0.23
    x_spacing = 1.1
    legend_handles = []

    x_rmse = np.arange(len(rmse_datasets)) * x_spacing
    for i, method in enumerate(methods):
        offset = (i - 1.5) * bw
        vals = rmse_dict[method]
        ax_rmse.bar(x_rmse + offset, vals, width=bw, color=method_colors[method])
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=method_colors[method], label=method))
        for j, value in enumerate(vals):
            ax_rmse.annotate(
                f'{value:.2f}',
                xy=(x_rmse[j] + offset, value),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=8,
            )

    ax_rmse.grid(True, axis='y', linewidth=0.6)
    ax_rmse.grid(False, axis='x')
    ax_rmse.set_xticks(x_rmse)
    ax_rmse.set_xticklabels(rmse_datasets, fontsize=10)
    ax_rmse.set_ylabel('RMSE (dex)')
    ax_rmse.set_ylim(0, 3.4)
    ax_rmse.yaxis.set_major_locator(MultipleLocator(0.5))
    ax_rmse.tick_params(axis='x', pad=6)

    x_acc = np.arange(len(acc_datasets)) * x_spacing
    for i, method in enumerate(methods):
        offset = (i - 1.5) * bw
        vals = acc_dict[method]
        ax_acc.bar(x_acc + offset, vals, width=bw, color=method_colors[method])
        for j, value in enumerate(vals):
            ax_acc.annotate(
                f'{value:.2f}',
                xy=(x_acc[j] + offset, value),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=8,
            )

    ax_acc.grid(True, axis='y', linewidth=0.6)
    ax_acc.grid(False, axis='x')
    ax_acc.set_xticks(x_acc)
    ax_acc.set_xticklabels(acc_datasets, fontsize=10)
    ax_acc.set_ylabel('Classification Accuracy')
    ax_acc.set_ylim(0, 1.05)
    ax_acc.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_acc.tick_params(axis='x', pad=6)

    fig.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(0.89, 0.93),
        bbox_transform=fig.transFigure,
        frameon=True,
        fontsize=8,
        labelspacing=0.8,
    )
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved RMSE/accuracy plot to {output}')
    return output


def _official_roc_figure(v, output='plots/roc.pdf', dpi=300, positive_is_stable=False):
    """Side-by-side ROC plots: resonant test (left) and random (right).
    Curves: Petit, Distilled equations, NN — colors match five_planet_plot.py.

    If positive_is_stable is False, positive class is unstable and scores are
    flipped so that larger scores indicate greater confidence in instability.
    """
    models = [
        # ('Petit+20', 'tab:red',
        #  dict(eval_type='petit')),
        ('Neural network', 'tab:orange',
         dict(eval_type='nn', version=v['nn_version'])),
        ('Distilled equations', 'tab:blue',
         dict(eval_type='pysr', version=v['nn_version'],
              pysr_version=v['pysr_version'],
              pysr_model_selection=str(v['pysr_model_selection']))),
        ('Pure SR', 'tab:green',
         dict(eval_type='pure_sr',
              pysr_version=v['pure_sr_version'],
              pysr_model_selection=str(v['pure_sr_model_selection']))),
        # ('Pure SR (no intermediate features)', 'tab:purple',
        #  dict(eval_type='pysr', version=v['pure_sr2_nn_version'],
        #       pysr_version=v['pure_sr2_version'],
        #       pysr_model_selection=str(v['pure_sr2_model_selection']))),
    ]
    panels = [('Resonant', 'test'),
              ('Random', 'random')]

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc('font', family='sans-serif')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)
    for ax, (panel_title, dataset) in zip(axes, panels):
        for label, color, cfg in models:
            args = SimpleNamespace(dataset=dataset, **cfg)
            truths, preds = get_truths_and_preds(args, clip=False)
            stable = truths >= 9
            if len(np.unique(stable)) < 2:
                continue
            labels = stable if positive_is_stable else ~stable
            scores = preds if positive_is_stable else -preds
            fpr, tpr, _ = roc_curve(labels, scores)
            auc = roc_auc_score(labels, scores)
            ax.plot(fpr, tpr, color=color, lw=1.6,
                    label=f'{label} (AUC={auc:.3f})')
            pred_stable = preds >= 9
            pred_positive = pred_stable if positive_is_stable else ~pred_stable
            true_positive = stable if positive_is_stable else ~stable
            true_negative = ~stable if positive_is_stable else stable
            tpr_9 = np.sum(pred_positive & true_positive) / max(np.sum(true_positive), 1)
            fpr_9 = np.sum(pred_positive & true_negative) / max(np.sum(true_negative), 1)
            ax.plot(fpr_9, tpr_9, marker='*', color=color,
                    markersize=12, markeredgecolor='black',
                    markeredgewidth=0.6, linestyle='None', zorder=5,
                    clip_on=False)
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title(panel_title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_box_aspect(1)
        ax.legend(loc='lower right', fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(output, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f'Saved official ROC figure to {output}')
    return output


def _official_std_figure(v, output='plots/std_hist.pdf', dpi=300):
    """Side-by-side std histograms: NN (left) vs Distilled equations (right)."""
    panels = [
        ('Neural network', v['nn_version']),
        ('Distilled equations', 15687),
    ]

    def load_std(version, split):
        args = SimpleNamespace(eval_type='nn', version=version, dataset=split)
        _, preds = get_full_truths_and_preds(args)
        return _as_numpy(preds)[:, 1]

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc('font', family='sans-serif')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi, sharey=True)
    stds = [
        (load_std(ver, 'test'), load_std(ver, 'random'), load_std(ver, 'val'))
        for _, ver in panels
    ]
    # print average std for each model/dataset
    for (title, _), (r, s, vstd) in zip(panels, stds):
        print(
            f'{title} - resonant test std: {r.mean():.4f}, '
            f'random test std: {s.mean():.4f}, '
            f'validation std: {vstd.mean():.4f}'
        )

    upper = max(max(r.max(), s.max()) for r, s, _ in stds)
    bins = np.linspace(0, upper, 60)
    xlabels = ["Predicted standard deviation", "NN predicted standard deviation"]

    for ax, (title, _), (res, rand, _), xlabel in zip(axes, panels, stds, xlabels):
        ax.hist(res, bins=bins, density=True, alpha=0.6, label='Resonant')
        ax.hist(rand, bins=bins, density=True, alpha=0.6, label='Random')
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel('Density')

    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(output, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f'Saved official std histogram figure to {output}')
    return output


def _as_numpy(array):
    if hasattr(array, 'detach'):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=None)
    parser.add_argument('--pysr_version', type=int, default=None)
    parser.add_argument('--best_complexity', action='store_true')
    parser.add_argument('--dataset', type=str, default='test', choices=['train','val','test', 'random', 'grid', 'five_planet'])
    parser.add_argument('--eval_type', type=str, default='pysr', choices=['pure_sr', 'pysr', 'nn', 'petit'])
    parser.add_argument('--pysr_model_selection', type=str, default=None, help='"best", "accuracy", "score", or an integer of the pysr equation complexity.')
    parser.add_argument('--include_stable', action='store_true', help='Include stable systems in RMSE calculation (i.e., where ground_truth is 10^9).')
    parser.add_argument('--classification', action='store_true', help='Calculate classification metrics with RMSE.')
    parser.add_argument('--calculate_all', action='store_true', help='Calculate official classification and RMSE metrics on test and random splits and save to pickles/ folder.')
    parser.add_argument('--version_json', type=str, default='official_versions.json', help='JSON file specifying versions to use for official calculations.')
    parser.add_argument('--official', action='store_true')

    args = parser.parse_args()
    if args.pysr_version is None:
        if args.version is not None:
            args.eval_type = 'nn'
        else:
            args.eval_type = 'petit'
    else:
        if args.version is None:
            args.eval_type = 'pure_sr'
        else:
            args.eval_type = 'pysr'

    return args


if __name__ == '__main__':
    args = get_args()

    if args.official:
        official_figures(version_json=args.version_json)
    elif args.best_complexity:
        path = get_results_path(args)
        results = load_pickle(path)
        best_complexity = max(results['val'].items(), key=lambda x: x[1]['ll'])[0]
        print(best_complexity)
    elif args.calculate_all:
        calculate_all_and_store(args)
    else:
        metrics = calculate_metrics(args, plot=True)
        print(metrics)
