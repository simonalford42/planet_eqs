import argparse
import torch
import os
import numpy as np
import spock_reg_model
from tqdm.notebook import tqdm
import petit
from pure_sr_evaluation import pure_sr_predict_fn, get_pure_sr_results
from interpret import get_pysr_results
from utils import assert_equal, load_pickle, save_pickle
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from einops import rearrange
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
    model = spock_reg_model.load(24880)

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


def nn_prediction_fn(model):
    model.eval()
    model.cuda()
    def predict_fn(x):
        pred = model(x.cuda(), noisy_val=False).cpu().detach().numpy()
        # just the mean
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
    ax.plot([4-3, 9+3], [4+0.61-3, 9+0.61+3], color='k', ls='--')
    ax.plot([4-3, 9+3], [4-0.61-3, 9-0.61+3], color='k', ls='--')
    ax.set_xlim(3+0.9, 10-0.9)
    ax.set_ylim(3+0.9, 10-0.9)

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


def get_truths_and_preds(args, clip=True):
    petit = args.eval_type == 'petit'
    dataloader = get_dataloader(args.dataset, petit=petit)
    predict_fn = get_prediction_fn(args)

    truths = []
    preds = []
    for x, y in tqdm(dataloader):
        truths.append(y.numpy())
        preds.append(predict_fn(x))

    truths = np.concatenate(truths)
    preds = np.concatenate(preds)
    assert_equal(truths.shape[0], preds.shape[0])

    if np.sum(np.isnan(preds)) > 0:
        print('num nan:', np.sum(np.isnan(preds)), '/', len(preds))
    preds[np.isnan(preds)] = 4

    if clip:
        preds = np.clip(preds, 4, 9)

    truths = np.average(truths, 1)  # avg the two ground truths for each sim


    return truths, preds

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


def lossfnc(testy, y):
    # y: [B, 2] batch of ground truth means. each input system has two
    #  simulations, one with a small initial perturbation, so the two means
    #  are samples from the distribution of instability times for that
    #  initial system
    # so we just sum over the loss for both of them.
    mu = testy[:, [0]]
    std = testy[:, [1]]

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

    safe_regression_loss = torch.where(
            ~torch.isfinite(regression_loss),
            -torch.ones_like(regression_loss)*100,
            regression_loss)
    safe_classifier_loss = torch.where(
            ~torch.isfinite(classifier_loss),
            -torch.ones_like(classifier_loss)*100,
            classifier_loss)

    total_loss = (
        safe_regression_loss * (~t_greater_9) +
        safe_classifier_loss * ( t_greater_9)
    )
    return -total_loss.sum(1)

    # total_regression = safe_regression_loss * (~t_greater_9)
    # total_classifier = safe_classifier_loss * ( t_greater_9)
    # total_regression = total_regression.sum(1)
    # total_classifier = total_classifier.sum(1)
    # total_loss = total_regression + total_classifier
    # return -total_loss


def calculate_loss(args):
    petit = args.eval_type == 'petit'
    dataloader = get_dataloader(args.dataset, petit=petit)
    model = spock_reg_model.load(version=args.version)
    model.eval()
    model.cuda()

    def predict_fn(x):
        return model(x.cuda(), noisy_val=False, deterministic=True).cpu().detach()

    xs = []
    truths = []
    preds = []
    for x, y in tqdm(dataloader):
        xs.append(x)
        truths.append(y)
        preds.append(predict_fn(x))

    xs = torch.cat(xs, dim=0)
    truths = torch.cat(truths, dim=0)
    preds = torch.cat(preds, dim=0)
    # preds[:, 0] = torch.clip(preds[:, 0], 4, 9)
    assert_equal(truths.shape, preds.shape)
    print('xs mean', xs.mean().item())
    print('truths mean', truths.mean().item())
    print('preds mean', preds[:, 0].mean().item())
    # loss = model._lossfnc(preds, truths).sum() / len(truths)
    loss = lossfnc(truths, preds[:, 0])
    loss = loss.sum() / len(truths)
    print(f'Loss for {args.dataset} dataset: {loss.item()}')


def calculate_metrics(args):
    truths, preds = get_truths_and_preds(args)
    print('truths mean', truths.mean().item())
    print('preds mean', preds.mean().item())

    # calculate rmse excluding stable systems
    stable_ixs = truths >= 9
    exclude_stable_preds = preds[~stable_ixs]
    exclude_stable_truths = truths[~stable_ixs]
    rmse = np.average(np.square(exclude_stable_truths - exclude_stable_preds))**0.5
    print(f'RMSE for {args.dataset} dataset: {rmse}')

    acc = np.mean((preds >= 9) == (truths >= 9))
    stable_only_acc = np.mean((preds[stable_ixs] >= 9) == (truths[stable_ixs] >= 9))
    stable_truths = truths >= 9
    auc = roc_auc_score(stable_truths, preds)
    bias = np.mean(preds - truths)
    print(f'Accuracy: {acc:.4f}, Stable-only Accuracy: {stable_only_acc:.4f}, AUC: {auc:.4f}, Bias: {bias:.4f}')
    plot_roc_curve(args, stable_truths, preds, auc)

    return {
        'rmse': rmse,
        'acc': acc,
        'stable_only_acc': stable_only_acc,
        'auc': auc,
        'bias': bias,
    }


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

    path = f'plots/roc_{get_model_name(args)}_{args.dataset}.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved ROC curve to {path}')




def const_predict_fn(const):
    return lambda *_: const


def calculate_all_and_store(args):
    path = get_results_path(args)
    results = {}
    datasets = ['test', 'random']
    for dataset in datasets:
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


def get_prediction_fn(args):
    if args.eval_type == 'petit':
        return petit.tsurv
    elif args.eval_type == 'pure_sr':
        pure_sr_results = get_pure_sr_results(args.pysr_version)
        return pure_sr_predict_fn(pure_sr_results, int(args.pysr_model_selection))
    elif args.eval_type == 'nn':
        model = spock_reg_model.load(version=args.version)
        return nn_prediction_fn(model)
    elif args.eval_type == 'pysr':
        model = spock_reg_model.load_with_pysr_f2(version=args.version, pysr_version=args.pysr_version, pysr_model_selection=args.pysr_model_selection)
        return nn_prediction_fn(model)
    else:
        raise ValueError(f"Unknown eval_type: {args.eval_type}")


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
    # for dataset in ['val', 'test', 'random']:
    # for dataset in ['val']:
    #     args.dataset = dataset
    #     calculate_loss(args)
    # assert 0

    if args.best_complexity:
        path = get_results_path(args)
        results = load_pickle(path)
        best_complexity = min(results['val'].items(), key=lambda x: x[1])[0]
        print(best_complexity)
    elif args.calculate_all:
        calculate_all_and_store(args)
    else:
        calculate_metrics(args)
