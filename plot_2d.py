import argparse
import os
import pickle

import matplotlib as mpl
mpl.use("agg")
from matplotlib import pyplot as plt
import numpy as np


DEFAULT_PANELS = [
    ("obertas_random", "The model of Obertas et al. (2017)"),
    ("petit_random", "The model of Petit et al. (2020)"),
    ("nn_random", "Tamayo et al. (2020), retrained for regression"),
    ("ours_random", "Our model"),
    ("ours_test", "Our model, applied to the resonant dataset"),
    ("theoretical_random", "Theoretical limit"),
]


MODEL_CACHE = {
    "petit": "cache_truths_preds_petit_{split}.pkl",
    "nn": "cache_truths_preds_nn_24880_{split}.pkl",
    "ours": "cache_truths_preds_pysr_24880_11003_26_{split}.pkl",
}


MAIN_COLOR = np.array([41, 68, 117], dtype=float) / 255.0


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def clipped(values):
    return np.clip(np.asarray(values, dtype=float), 4.0, 9.0)


def load_model_panel(model_key, split, cache_dir):
    cache_name = MODEL_CACHE[model_key].format(split=split)
    cached = load_pickle(os.path.join(cache_dir, cache_name))
    truths = np.asarray(cached["truths_full"], dtype=float)
    preds = np.asarray(cached["preds_full"], dtype=float)

    if preds.ndim == 2:
        preds = preds[:, 0]

    # One model prediction corresponds to the system; plot it against both
    # direct integrations, matching the old comparison-figure treatment.
    truths = truths.reshape(-1)
    preds = np.repeat(preds, 2)
    return clipped(truths), clipped(preds)


def load_theoretical_panel(split, cache_dir):
    cached = load_pickle(os.path.join(cache_dir, f"cache_truths_preds_nn_24880_{split}.pkl"))
    truths = np.asarray(cached["truths_full"], dtype=float)
    return clipped(truths[:, 0]), clipped(truths[:, 1])


def mutual_hill_spacing(a_inner, a_outer, m_inner, m_outer):
    hill = ((m_inner + m_outer) / 3.0) ** (1.0 / 3.0) * 0.5 * (a_inner + a_outer)
    return (a_outer - a_inner) / hill


def load_obertas_panel(data_dir, cache_dir):
    data = load_pickle(os.path.join(data_dir, "random_dataset.pkl"))
    labels = data["labels"]
    x = np.asarray(data["X"], dtype=float)

    def feature(name):
        return x[:, :, labels.index(name)].mean(axis=1)

    a1, a2, a3 = feature("a1"), feature("a2"), feature("a3")
    m1, m2, m3 = feature("m1"), feature("m2"), feature("m3")
    beta12 = mutual_hill_spacing(a1, a2, m1, m2)
    beta23 = mutual_hill_spacing(a2, a3, m2, m3)
    beta = np.minimum(beta12, beta23)

    preds = 0.951 * beta - 1.202
    cached = load_pickle(os.path.join(cache_dir, "cache_truths_preds_nn_24880_random.pkl"))
    truths = np.asarray(cached["truths_full"], dtype=float).reshape(-1)
    preds = np.repeat(preds, 2)
    return clipped(truths), clipped(preds)


def load_panel(panel_key, cache_dir, data_dir):
    if panel_key == "obertas_random":
        return load_obertas_panel(data_dir, cache_dir)
    if panel_key == "theoretical_random":
        return load_theoretical_panel("random", cache_dir)

    model_key, split = panel_key.rsplit("_", 1)
    return load_model_panel(model_key, split, cache_dir)


def subsample(truths, preds, max_points, rng):
    if max_points is None or len(truths) <= max_points:
        return truths, preds
    ixs = rng.choice(len(truths), size=max_points, replace=False)
    return truths[ixs], preds[ixs]


def draw_joint_panel(fig, outer_spec, truths, preds, title=None, color=MAIN_COLOR, max_points=None, rng=None, show_bias=False, n_bias_bins=20):
    if rng is None:
        rng = np.random.default_rng(0)
    truths, preds = subsample(truths, preds, max_points, rng)

    sub = outer_spec.subgridspec(
        2,
        2,
        height_ratios=(0.16, 1.0),
        width_ratios=(1.0, 0.17),
        hspace=0.08,
        wspace=0.08,
    )
    ax_histx = fig.add_subplot(sub[0, 0])
    ax = fig.add_subplot(sub[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(sub[1, 1], sharey=ax)

    bins = np.linspace(4.0, 9.0, 16)
    ax_histx.hist(truths, bins=bins, color=color)
    ax_histy.hist(preds, bins=bins, orientation="horizontal", color=color)

    ax.scatter(truths, preds, s=6, color=color, alpha=0.20, edgecolors="none", rasterized=True)
    ax.plot([1, 12], [1, 12], color="black", lw=1.1)

    if show_bias:
        edges = np.linspace(4.0, 9.0, n_bias_bins + 1)
        idx = np.digitize(truths, edges) - 1
        idx = np.clip(idx, 0, n_bias_bins - 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        means = np.full(n_bias_bins, np.nan)
        for b in range(n_bias_bins):
            mask = idx == b
            if mask.any():
                means[b] = preds[mask].mean()
        valid = ~np.isnan(means)
        ax.plot(centers[valid], means[valid], color="tab:orange", lw=1.6, zorder=5)

    ax.set_xlim(3.9, 9.1)
    ax.set_ylim(3.9, 9.1)
    ax.set_xticks(np.arange(4, 10))
    ax.set_yticks(np.arange(4, 10))
    ax.set_xlabel("Truth")
    ax.set_ylabel("Predicted")
    if title is not None:
        ax.set_title(title, y=-0.31, fontsize=13, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.tick_params(axis="y", left=False, labelleft=False)
    ax_histy.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    for hist_ax in (ax_histx, ax_histy):
        hist_ax.spines["top"].set_visible(False)
        hist_ax.spines["right"].set_visible(False)
        hist_ax.spines["left"].set_visible(False)
        hist_ax.spines["bottom"].set_color("0.35")
    ax_histy.spines["bottom"].set_visible(False)
    ax_histy.spines["left"].set_color("0.35")

    return ax


def plot_2d(pred, truth, path, title=None, max_points=35000, seed=0, dpi=300):
    """Save a single predicted-vs-true 2D comparison panel."""
    rng = np.random.default_rng(seed)
    truth = clipped(truth).reshape(-1)
    pred = clipped(pred).reshape(-1)

    if truth.shape != pred.shape:
        raise ValueError(f"truth and pred must have the same shape, got {truth.shape} and {pred.shape}")

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc("font", family="serif")

    fig = plt.figure(figsize=(4.6, 4.6), dpi=dpi)
    grid = fig.add_gridspec(1, 1)
    draw_joint_panel(fig, grid[0, 0], truth, pred, title=title, max_points=max_points, rng=rng)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_comparison_figure(args):
    rng = np.random.default_rng(args.seed)
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc("font", family="serif")

    fig = plt.figure(figsize=(18, 10), dpi=args.dpi)
    grid = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.22)

    for ix, (panel_key, title) in enumerate(DEFAULT_PANELS):
        truths, preds = load_panel(panel_key, args.cache_dir, args.data_dir)
        row, col = divmod(ix, 3)
        draw_joint_panel(
            fig,
            grid[row, col],
            truths,
            preds,
            title,
            MAIN_COLOR,
            args.max_points,
            rng,
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure to {args.output}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="pickles")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="plots/comparison_figures.png")
    parser.add_argument("--max-points", type=int, default=35000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


if __name__ == "__main__":
    make_comparison_figure(get_args())
