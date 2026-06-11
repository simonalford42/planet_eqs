"""NN and distilled-equation predictions on the Dan/Rath resonant grid.

The initial conditions mirror ``figures/resonances.ipynb``: two massive
planets at P=1 yr and P=2.04 yr, and a middle massless body scanned over
resonant angle and orbital period.
"""

import argparse
import os
import pickle
import sys
import time
import warnings

import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
SR_DIR = os.path.join(REPO_ROOT, "sr_results")

for path in (REPO_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


INSTABILITY_TIME_LABEL = r"$\log_{10}(T_{\rm inst})$"
COLOR_MAP = plt.cm.plasma
# Match the notebook's particle insertion order (inner, outer, middle), but
# feed the NN's extended per-planet features in physical inner-middle-outer order.
NN_TRIO_INDICES = [1, 3, 2]

def get_resonant_grid(ngrid, p2_min=1.47, p2_max=1.56):
    angles = np.linspace(-np.pi, np.pi, ngrid)
    middle_periods = np.linspace(p2_min, p2_max, ngrid)
    return angles, middle_periods


def get_parameters(angles, middle_periods):
    return [(angle, period) for period in middle_periods for angle in angles]


def get_centered_grid(xlist, ylist, values):
    if len(xlist) < 2 or len(ylist) < 2:
        raise ValueError("Need at least a 2x2 grid to draw a pcolormesh")

    dx = xlist[1] - xlist[0]
    dy = ylist[1] - ylist[0]
    xgrid = list(xlist - dx / 2) + [xlist[-1] + dx / 2]
    ygrid = list(ylist - dy / 2) + [ylist[-1] + dy / 2]
    X, Y = np.meshgrid(xgrid, ygrid)
    Z = np.asarray(values, dtype=float).reshape(len(ylist), len(xlist))
    return X, Y, Z


def get_simulation(
    par,
    inner_period=1.0,
    outer_period=2.04,
    inner_mass=3e-5,
    middle_mass=0.0,
    outer_mass=3e-5,
    middle_eccentricity=0.05,
    pomega0=0.0,
):
    import rebound

    resonant_angle, middle_period = par

    j1, k1, j2, k2 = 3, 1, 3, 1
    inner_longitude = -np.pi / (j1 - k1)
    middle_longitude = resonant_angle / j1
    outer_longitude = -((k2 + 1) % 2) * np.pi / (j2 + k2)

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.add(m=1.0)
    sim.add(m=inner_mass, P=inner_period, l=inner_longitude)
    sim.add(m=outer_mass, P=outer_period, l=outer_longitude)
    sim.add(
        m=middle_mass,
        P=middle_period,
        l=middle_longitude,
        e=middle_eccentricity,
        pomega=pomega0,
    )

    ps = sim.particles
    for ix in (1, 2, 3):
        if ps[ix].m > 0:
            ps[ix].r = ps[ix].a * (ps[ix].m / (3.0 * ps[0].m)) ** (1.0 / 3.0)

    sim.dt = sim.particles[1].P / 12.0
    sim.collision = "direct"
    sim.move_to_com()
    return sim


def load_regressor(version, seed=0, pysr_version=None, pysr_model_selection=None, cuda=None):
    import spock
    import spock_reg_model
    import torch

    if pysr_version is None:
        model = spock_reg_model.load(version, seed)
    else:
        model = spock_reg_model.load_with_pysr_f2(
            version,
            pysr_version,
            pysr_model_selection=pysr_model_selection,
            pysr_dir=SR_DIR + os.sep,
        )

    use_cuda = torch.cuda.is_available() if cuda is None else cuda
    model.eval()
    return spock.NonSwagFeatureRegressor(model, cuda=use_cuda)


def create_cached_input(sim, regressor, indices=NN_TRIO_INDICES):
    import rebound

    sim = sim.copy()
    sim.dt = 0.05
    sim.init_megno()
    sim.exit_max_distance = 20.0

    try:
        cached_input = regressor.predict_up_to_cached_input(sim, indices=indices)
        if hasattr(cached_input, "detach"):
            cached_input = cached_input.detach().cpu()
        return cached_input
    except (rebound.Escape, rebound.Encounter):
        return None


def predict_from_cached_input(regressor, cached_input):
    import torch

    if cached_input is None:
        return None

    if hasattr(cached_input, "cuda"):
        cached_input = cached_input.cuda() if regressor.cuda else cached_input.cpu()

    with torch.no_grad():
        out_dict = regressor.model(
            cached_input,
            noisy_val=False,
            return_intermediates=True,
            deterministic=True,
        )

    return {
        "mean": float(out_dict["mean"][0, 0].detach().cpu().numpy()),
        "std": float(out_dict["std"][0, 0].detach().cpu().numpy()),
        "f1": out_dict["summary_stats"][0].detach().cpu().numpy(),
    }


def get_results_path(
    ngrid,
    version=None,
    pysr_version=None,
    pysr_model_selection=None,
    input_cache=False,
):
    if input_cache:
        path = os.path.join(DATA_DIR, "resonant_cache_ngrid=300.pkl")
    else:
        path = os.path.join(DATA_DIR, f"resonant_v={version}_ngrid={ngrid}")
        if pysr_version is not None:
            path = os.path.join(
                path + f"_pysr_f2_v={pysr_version}",
                f"{pysr_model_selection}",
            )
        path += ".pkl"
    return path


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print("Saved", path)


def load_or_create_input_cache(args):
    path = get_results_path(args.Ngrid, input_cache=True)
    if os.path.exists(path) and not args.overwrite:
        cache = load_pickle(path)
        expected = args.Ngrid * args.Ngrid
        if len(cache) != expected:
            raise ValueError(f"Cache {path} has {len(cache)} entries, expected {expected}")
        # print("Loaded input cache from", path)
        return cache

    # print("Creating input cache at", path)
    regressor = load_regressor(args.version, args.seed, cuda=args.cuda)
    angles, middle_periods = get_resonant_grid(args.Ngrid, args.p2_min, args.p2_max)
    parameters = get_parameters(angles, middle_periods)
    simulations = [
        get_simulation(
            par,
            inner_period=args.inner_period,
            outer_period=args.outer_period,
            inner_mass=args.inner_mass,
            middle_mass=args.middle_mass,
            outer_mass=args.outer_mass,
            middle_eccentricity=args.middle_eccentricity,
            pomega0=args.pomega0,
        )
        for par in parameters
    ]

    t0 = time.time()
    cache = []
    for i, sim in enumerate(simulations, start=1):
        cache.append(create_cached_input(sim, regressor))
        if i == 1 or i == len(simulations) or i % max(1, len(simulations) // 10) == 0:
            print(f"  cached {i}/{len(simulations)}")

    save_pickle(cache, path)
    print(f"Input cache complete in {time.time() - t0:.1f}s")
    return cache


def compute_predictions(args, pysr=False):
    path = get_results_path(
        args.Ngrid,
        args.version,
        pysr_version=args.pysr_version if pysr else None,
        pysr_model_selection=args.pysr_model_selection if pysr else None,
    )
    if os.path.exists(path) and not args.overwrite:
        print("Prediction cache already exists at", path)
        return load_pickle(path)

    cache = load_or_create_input_cache(args)
    if pysr:
        print(
            "Computing equation predictions "
            f"(NN {args.version}, PySR {args.pysr_version}, complexity {args.pysr_model_selection})"
        )
        regressor = load_regressor(
            args.version,
            args.seed,
            pysr_version=args.pysr_version,
            pysr_model_selection=args.pysr_model_selection,
            cuda=args.cuda,
        )
    else:
        print(f"Computing NN predictions (version {args.version})")
        regressor = load_regressor(args.version, args.seed, cuda=args.cuda)

    t0 = time.time()
    results = []
    for i, cached_input in enumerate(cache, start=1):
        results.append(predict_from_cached_input(regressor, cached_input))
        if i == 1 or i == len(cache) or i % max(1, len(cache) // 10) == 0:
            print(f"  predicted {i}/{len(cache)}")

    save_pickle(results, path)
    print(f"Predictions complete in {time.time() - t0:.1f}s")
    return results


def extract_mean(results):
    return np.array([d["mean"] if d is not None else np.nan for d in results], dtype=float)


def plot_prediction_panel(args, ax, results, title, show_ylabel=False):
    angles, middle_periods = get_resonant_grid(args.Ngrid, args.p2_min, args.p2_max)
    values = extract_mean(results)
    X, Y, Z = get_centered_grid(angles, middle_periods, values)

    cmap = COLOR_MAP.copy().reversed()
    cmap.set_bad(color="white")

    im = ax.pcolormesh(X, Y, Z, vmin=4, vmax=9, cmap=cmap, rasterized=True)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(args.p2_min, args.p2_max)
    ax.set_xlabel("Resonant Angle")
    if show_ylabel:
        ax.set_ylabel("Middle Planet Orbital Period")
    ax.set_title(title)
    ax.set_xticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
    ax.set_xticklabels(['-π','-π/2', '0', 'π/2', 'π'])

    inner_res = 1.5 * args.inner_period
    outer_res = 0.75 * args.outer_period
    resonance_lines = [
        (inner_res, "3:2 MMR"),
        (outer_res, "4:3 MMR"),
    ]
    for y, _ in resonance_lines:
        if args.p2_min <= y <= args.p2_max:
            ax.axhline(y, color="black", lw=0.9, alpha=0.45, ls="--")

    return im


def annotate_resonances_between_panels(args, fig, axes):
    resonances = [
        (1.5 * args.inner_period, "3:2 MMR"),
        (0.75 * args.outer_period, "4:3 MMR"),
    ]
    if not all(args.p2_min <= y <= args.p2_max for y, _ in resonances):
        return

    fig.canvas.draw()
    left_pos = axes[0].get_position()
    right_pos = axes[1].get_position()
    gap_x = 0.5 * (left_pos.x1 + right_pos.x0)

    to_fig = fig.transFigure.inverted()
    line_kwargs = {
        "color": "black",
        "lw": 0.9,
        "alpha": 0.75,
        "transform": fig.transFigure,
        "clip_on": False,
    }
    arrow_kwargs = {
        "arrowstyle": "<|-|>",
        "mutation_scale": 7,
        "color": "black",
        "lw": 0.9,
        "alpha": 0.75,
        "transform": fig.transFigure,
        "clip_on": False,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    half_arrow_length = 0.4 * (right_pos.x0 - left_pos.x1)

    for y, label in resonances:
        fig_y = to_fig.transform(axes[0].transData.transform((np.pi, y)))[1]
        text_y = fig_y + 0.043
        stem_top_y = fig_y + 0.028

        fig.text(
            gap_x,
            text_y,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )
        fig.add_artist(
            mpl.lines.Line2D(
                [gap_x, gap_x],
                [stem_top_y, fig_y],
                **line_kwargs,
            )
        )
        fig.add_artist(
            mpl.patches.FancyArrowPatch(
                (gap_x - half_arrow_length, fig_y),
                (gap_x + half_arrow_length, fig_y),
                **arrow_kwargs,
            )
        )


def plot_combined_predictions(args, nn_results, eq_results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    im = plot_prediction_panel(
        args,
        axes[0],
        nn_results,
        "Neural network",
        show_ylabel=True,
    )
    plot_prediction_panel(
        args,
        axes[1],
        eq_results,
        "Distilled equations",
    )

    cb = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cb.set_label(INSTABILITY_TIME_LABEL)
    annotate_resonances_between_panels(args, fig, axes)

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure to", output_path)


def plot_results(args):
    nn_path = get_results_path(args.Ngrid, args.version)
    eq_path = get_results_path(
        args.Ngrid,
        args.version,
        pysr_version=args.pysr_version,
        pysr_model_selection=args.pysr_model_selection,
    )

    missing = [path for path in (nn_path, eq_path) if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing prediction cache(s):\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\nRun with --compute first, or run without --plot/--compute flags to do both."
        )

    nn_results = load_pickle(nn_path)
    eq_results = load_pickle(eq_path)

    filename = "resonant.pdf"
    output = os.path.join(filename)
    plot_combined_predictions(
        args,
        nn_results,
        eq_results,
        output,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ngrid", "-n", type=int, default=300)
    parser.add_argument("--version", "-v", type=int, default=24880)
    parser.add_argument("--pysr_version", type=int, default=11003)
    parser.add_argument("--pysr_model_selection", type=int, default=26)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--create_input_cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dpi", type=int, default=500)

    parser.add_argument("--p2_min", type=float, default=1.47)
    parser.add_argument("--p2_max", type=float, default=1.56)
    parser.add_argument("--inner_period", type=float, default=1.0)
    parser.add_argument("--outer_period", type=float, default=2.04)
    parser.add_argument("--inner_mass", type=float, default=3e-5)
    parser.add_argument("--middle_mass", type=float, default=0.0)
    parser.add_argument("--outer_mass", type=float, default=3e-5)
    parser.add_argument("--middle_eccentricity", type=float, default=0.05)
    parser.add_argument("--pomega0", type=float, default=0.0)
    parser.add_argument("--cuda", action="store_true", default=None)
    parser.add_argument("--cpu", dest="cuda", action="store_false")

    args = parser.parse_args()
    if not args.compute and not args.plot and not args.create_input_cache:
        args.compute = True
        args.plot = True
    return args


def main():
    args = get_args()
    start = time.time()

    if args.create_input_cache:
        load_or_create_input_cache(args)
    elif args.compute:
        compute_predictions(args, pysr=False)
        compute_predictions(args, pysr=True)

    if args.plot:
        plot_results(args)

    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
