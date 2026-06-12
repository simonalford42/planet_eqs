"""Plot the period-ratio figure for the published N=300 grid using
``planet_stability`` for the predictions.

For each cached (already-integrated, already-standardized) feature
tensor from ``figures/period_results/cache_ngrid=300.pkl``, we:

  1. Reverse the StandardScaler (so the features are raw 41-col rebound
     outputs again).
  2. Drop the 10 always-zero input positions, giving a 31-col raw input.
  3. Run ``StabilityPredictor.predict_from_features`` to get
     ``log10(T_inst / P_inner)``.
  4. Plot the 300 * 300 = 90,000 predictions as a pcolormesh, using the
     same color scale (4 <= mu <= 9) and axes as the existing figure.

The output PDF goes to ``minimal/plot_period_ratio_planet_stability.pdf``
so you can A/B it against
``figures/period_results/v=24880_ngrid=300_pysr_f2_v=11003/26.pdf``.

Usage:
    python minimal/plot_period_ratio.py
    python minimal/plot_period_ratio.py --complexity 14   # other eq
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planet_stability import StabilityPredictor, DEFAULT_COMPLEXITY  # noqa: E402


CACHE_PATH = os.path.join(
    REPO_ROOT, "figures", "period_results", "cache_ngrid=300.pkl"
)
NGRID = 300

# Same range as figures/period_ratio_figure.py:get_period_ratios.
P12_RANGE = (0.55, 0.76)
P23_RANGE = (0.55, 0.76)


# =====================================================================
# Inverse-scaler arrays. These are the 41-column training-set mean and
# scale used by the published model. We need them to undo the scaler
# baked into the cached features. (planet_stability itself has the
# scaler folded directly into its weights; this script needs the
# inverse to bridge the cache layout back to raw features.)
# =====================================================================
SSX_MEAN_41 = np.array([
     4.95458585e+03,  5.67411891e-02,  3.83176945e-02,  2.97223474e+00,
     6.29733979e-02,  3.50074471e-02,  6.72845676e-01,  9.92794768e+00,
     9.99628430e-01,  5.39591547e-02,  2.92795061e-02,  2.12480714e-03,
    -1.01500319e-02,  1.82667162e-02,  1.00813201e-02,  5.74404197e-03,
     6.86570242e-03,  1.25316320e+00,  4.76946516e-02,  2.71326280e-02,
     7.02054326e-03,  9.83378673e-03, -5.70616748e-03,  5.50782881e-03,
    -8.44213953e-04,  2.05958338e-03,  1.57866569e+00,  4.31476211e-02,
     2.73316392e-02,  1.05505555e-02,  1.03922250e-02,  7.36865006e-03,
    -6.00523246e-04,  6.53016990e-03, -1.72038113e-03,  1.24807860e-05,
     1.60314173e-05,  1.21732696e-05,  5.67292645e-03,  1.92488263e-01,
     5.08607199e-03,
])
SSX_SCALE_41 = np.array([
    2.88976974e+03, 6.10019661e-02, 4.03849732e-02, 4.81638693e+01,
    6.72583662e-02, 4.17939679e-02, 8.15995339e+00, 2.26871589e+01,
    4.73612029e-03, 7.09223721e-02, 3.06455099e-02, 7.10726478e-01,
    7.03392022e-01, 7.07873597e-01, 7.06030923e-01, 7.04728204e-01,
    7.09420909e-01, 1.90740659e-01, 4.75502285e-02, 2.77188320e-02,
    7.08891412e-01, 7.05214134e-01, 7.09786887e-01, 7.04371833e-01,
    7.04371110e-01, 7.09828420e-01, 3.33589977e-01, 5.20857790e-02,
    2.84763136e-02, 7.02210626e-01, 7.11815232e-01, 7.10512240e-01,
    7.03646004e-01, 7.08017286e-01, 7.06162814e-01, 2.12569430e-05,
    2.35019125e-05, 2.04211110e-05, 7.51048890e-02, 3.94254400e-01,
    7.11351099e-02,
])
# Indices in the 41-col layout that we keep (same as in planet_stability).
ZEROED_INDICES = {7, 3, 6, 1, 2, 4, 5, 38, 39, 40}
KEPT_INDICES = np.array([i for i in range(41) if i not in ZEROED_INDICES])
assert len(KEPT_INDICES) == 31


def cache_entry_to_raw_31(entry) -> np.ndarray:
    """Take one entry from cache_ngrid=300.pkl (a torch tensor of shape
    (1, 100, 41), standardized) and return a (100, 31) raw-feature
    numpy array ready for ``predict_from_features``.
    """
    # Torch -> numpy
    if hasattr(entry, "detach"):
        arr = entry.detach().cpu().numpy()
    else:
        arr = np.asarray(entry)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    assert arr.shape == (100, 41), f"unexpected cache entry shape {arr.shape}"

    # Reverse the scaler so we get raw features.
    raw_41 = arr * SSX_SCALE_41 + SSX_MEAN_41
    return raw_41[:, KEPT_INDICES]                  # (100, 31)


def get_period_ratios(ngrid: int):
    P12s = np.linspace(P12_RANGE[0], P12_RANGE[1], ngrid)
    P23s = np.linspace(P23_RANGE[0], P23_RANGE[1], ngrid)
    return P12s, P23s


def get_centered_grid(xlist, ylist, probs):
    """Same as figures/period_ratio_figure.py:get_centered_grid — bins
    are centered on the gridded values."""
    dx = xlist[1] - xlist[0]
    dy = ylist[1] - ylist[0]
    xgrid = list(xlist - dx / 2) + [xlist[-1] + dx / 2]
    ygrid = list(ylist - dy / 2) + [ylist[-1] + dy / 2]
    X, Y = np.meshgrid(xgrid, ygrid)
    # The original code iterates P12 outer, P23 inner -> reshape to (y, x).
    Z = np.array(probs).reshape(len(xlist), len(ylist)).T
    return X, Y, Z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--complexity", type=int, default=DEFAULT_COMPLEXITY,
        help="PySR equation complexity to use (default: 26)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="output path (default: minimal/plot_period_ratio_c<C>.pdf)",
    )
    parser.add_argument(
        "--png", action="store_true",
        help="save as png instead of pdf",
    )
    args = parser.parse_args()

    if not os.path.exists(CACHE_PATH):
        sys.exit(
            f"cache not found at {CACHE_PATH}.\n"
            f"This script requires figures/period_results/cache_ngrid=300.pkl "
            f"(generated by figures/period_ratio_figure.py --create_input_cache)."
        )

    print(f"Loading cache from {CACHE_PATH}...")
    t0 = time.time()
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    print(f"  loaded {len(cache)} entries in {time.time() - t0:.1f}s")
    assert len(cache) == NGRID * NGRID, \
        f"expected {NGRID * NGRID} entries, got {len(cache)}"

    model = StabilityPredictor.load(complexity=args.complexity)
    print(f"Predictor ready. Complexity={args.complexity}.")
    print(f"  Equation: {model.equation()}")

    print(f"Predicting {len(cache)} entries...")
    t0 = time.time()
    preds = np.full(len(cache), np.nan)
    n_none = 0
    n_done = 0
    for i, entry in enumerate(cache):
        if entry is None:
            n_none += 1
            continue
        X_raw = cache_entry_to_raw_31(entry)
        preds[i] = model.predict_from_features(X_raw)
        n_done += 1
    print(f"  {n_done} predictions, {n_none} skipped (None entries), "
          f"in {time.time() - t0:.1f}s")

    # Build the grid for pcolormesh — same convention as the published figure.
    P12s, P23s = get_period_ratios(NGRID)
    X, Y, Z = get_centered_grid(P12s, P23s, preds)

    # White out NaNs (failed integrations) — mirrors the figure style.
    cmap = plt.cm.plasma.reversed().copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(X, Y, Z, vmin=4, vmax=9, cmap=cmap, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$P_1/P_2$")
    ax.set_ylabel(r"$P_2/P_3$")
    ax.set_title(
        f"planet_stability (complexity={args.complexity}, N={NGRID})"
    )
    ax.set_xticks([0.55, 0.60, 0.65, 0.70, 0.75])
    ax.set_yticks([0.55, 0.60, 0.65, 0.70, 0.75])
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(r"$\log_{10}(T_{\rm inst} / P_{\rm inner})$")

    if args.out is None:
        suffix = "png" if args.png else "pdf"
        args.out = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"plot_period_ratio_c{args.complexity}.{suffix}",
        )
    plt.savefig(args.out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {args.out}")
    print()
    print("Compare side-by-side with the existing figure:")
    print(
        "  figures/period_results/v=24880_ngrid=300_pysr_f2_v=11003/"
        f"{args.complexity}.pdf"
    )


if __name__ == "__main__":
    main()
