"""Make the 2D prediction-vs-truth plots for the minimal c=26 equation.

This loads the raw resonant/random datasets and predicts from scratch with
``planet_stability.StabilityPredictor``. It does not read or write prediction
caches. The RMSE definition matches ``evaluation.py``: truths are averaged
per system, predictions are clipped to [4, 9], and stable systems
(truth >= 9) are excluded from RMSE.
"""

from __future__ import annotations

import argparse
import os
import pickle

import matplotlib as mpl

mpl.use("agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from planet_stability import StabilityPredictor  # noqa: E402


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "minimal", "2d_plot_c26.pdf")
MAIN_COLOR = np.array([41, 68, 117], dtype=float) / 255.0
ZEROED_INDICES = {7, 3, 6, 1, 2, 4, 5, 38, 39, 40}
KEPT_INDICES = np.array([i for i in range(41) if i not in ZEROED_INDICES])


def clipped(values):
    return np.clip(np.asarray(values, dtype=float), 4.0, 9.0)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dataset_path(name):
    return os.path.join(REPO_ROOT, "data", f"{name}_dataset.pkl")


def evaluation_test_indices(n_items):
    """Match sklearn.train_test_split(..., test_size=0.1, random_state=0).

    evaluation.py uses the held-out 10% final split for the resonant "test"
    set. sklearn's ShuffleSplit takes the first n_test entries from this
    permutation as the test split.
    """
    n_test = int(np.ceil(0.1 * n_items))
    rng = np.random.RandomState(0)
    return rng.permutation(n_items)[:n_test]


def load_raw_split(split):
    if split == "test":
        data = load_pickle(dataset_path("resonant"))
        ixs = evaluation_test_indices(len(data["X"]))
        return data["X"][ixs], data["y"][ixs]
    if split == "random":
        data = load_pickle(dataset_path("random"))
        return data["X"], data["y"]
    raise ValueError(f"unknown split {split!r}")


def predict_dataset(model, X, batch_size=2048):
    preds = np.empty(len(X), dtype=float)
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        raw_31 = X[start:stop, :, KEPT_INDICES]
        for j, features in enumerate(raw_31):
            preds[start + j] = model.predict_from_features(features)
    return preds


def calc_rmse(truths_full, preds):
    truths = np.average(truths_full, axis=1) if truths_full.ndim == 2 else truths_full
    preds = clipped(preds)
    unstable = truths < 9
    return float(np.sqrt(np.mean((truths[unstable] - preds[unstable]) ** 2)))


def scatter_arrays_for_plot(truths_full, preds):
    truths = clipped(truths_full.reshape(-1))
    preds = clipped(np.repeat(preds, truths_full.shape[1] if truths_full.ndim == 2 else 1))
    return truths, preds


def subsample(truths, preds, max_points, rng):
    if max_points is None or len(truths) <= max_points:
        return truths, preds
    ixs = rng.choice(len(truths), size=max_points, replace=False)
    return truths[ixs], preds[ixs]


def draw_joint_panel(fig, outer_spec, truths, preds, title, rng, max_points):
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
    ax_histx.hist(truths, bins=bins, color=MAIN_COLOR)
    ax_histy.hist(preds, bins=bins, orientation="horizontal", color=MAIN_COLOR)

    ax.scatter(
        truths,
        preds,
        s=6,
        color=MAIN_COLOR,
        alpha=0.2,
        edgecolors="none",
        rasterized=False,
    )
    ax.plot([1, 12], [1, 12], color="black", lw=1.1)

    ax.set_xlim(3.9, 9.1)
    ax.set_ylim(3.9, 9.1)
    ax.set_xticks(np.arange(4, 10))
    ax.set_yticks(np.arange(4, 10))
    ax.set_xlabel("Truth")
    ax.set_ylabel("Predicted")
    ax.set_title(title, y=-0.31, fontsize=13)
    ax.set_box_aspect(1)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-points", type=int, default=35000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc("font", family="sans-serif")

    rng = np.random.default_rng(args.seed)
    model = StabilityPredictor.load(complexity=26)

    fig = plt.figure(figsize=(9.2, 4.8), dpi=args.dpi)
    grid = fig.add_gridspec(1, 2, hspace=0.15, wspace=0.22)

    panels = [("test", "Resonant", "resonant"), ("random", "Random", "random")]
    for col, (split, title_label, print_label) in enumerate(panels):
        X, truths_full = load_raw_split(split)
        preds_raw = predict_dataset(model, X)
        rmse = calc_rmse(truths_full, preds_raw)
        truths, preds = scatter_arrays_for_plot(truths_full, preds_raw)
        draw_joint_panel(
            fig,
            grid[0, col],
            truths,
            preds,
            f"{title_label}, RMSE={rmse:.2f}",
            rng,
            args.max_points,
        )
        print(f"{print_label} RMSE: {rmse:.6f}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"Saved 2D plot to {args.out}")


if __name__ == "__main__":
    main()
