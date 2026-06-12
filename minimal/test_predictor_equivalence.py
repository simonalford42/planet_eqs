"""Verify that the new planet_eqs.StabilityPredictor produces *identical*
predictions to the existing pipeline used in figures/period_ratio_figure.py.

Runs predictions on a handful of rebound simulations through both paths
and asserts numerical equality (bitwise — both paths execute the same
code, just with different wrappers).

Usage:
    python test_predictor_equivalence.py
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import torch

# Ensure repo root + figures/ are importable, mirroring what
# period_ratio_figure.py does at top-of-file.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "figures"))

import rebound  # noqa: E402

import spock_reg_model  # noqa: E402
import modules  # noqa: E402
from figures import spock  # noqa: E402
from planet_eqs import StabilityPredictor  # noqa: E402


def get_simulation(P12: float, P23: float) -> rebound.Simulation:
    """Same construction as figures/period_ratio_figure.py:get_simulation."""
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.add(m=1.0)
    sim.add(m=1e-4, P=1.0, theta="uniform")
    sim.add(m=1e-4, P=1.0 / P12, theta="uniform")
    sim.add(m=1e-4, P=1.0 / P12 / P23, theta="uniform")
    sim.move_to_com()
    return sim


def existing_predict_nn(model_regressor, sim):
    """Mirror figures/period_ratio_figure.py:get_model_prediction (NN path).

    The existing pipeline preps the sim (dt, MEGNO init, exit distance)
    before calling model.predict — we mirror that here.
    """
    sim = sim.copy()
    sim.dt = 0.05
    sim.init_megno()
    sim.exit_max_distance = 20.0
    try:
        out_dict = model_regressor.predict(sim)
    except rebound.Escape:
        return None
    if out_dict is None:
        return None
    return {
        "mean": float(out_dict["mean"][0, 0].detach().cpu().numpy()),
        "std": float(out_dict["std"][0, 0].detach().cpu().numpy()),
    }


def make_test_sims(seed: int = 0):
    """Test cases sampled from the period_ratio_figure grid.

    The figure scans P12, P23 in linspace(0.55, 0.76, Ngrid). We pick a
    handful of (P12, P23) pairs from that range — covering corners,
    midpoint, and a couple of off-diagonal points — so we exercise the
    same regime users will hit when reproducing the plot.
    """
    import random

    pairs = [
        (0.55, 0.55),   # tight–tight   (likely escape)
        (0.55, 0.76),   # tight–wide
        (0.76, 0.55),   # wide–tight
        (0.76, 0.76),   # wide–wide     (likely stable, mu hits ceiling)
        (0.655, 0.655), # exact center
        (0.60, 0.70),   # off-diagonal
        (0.70, 0.60),   # off-diagonal
        (0.62, 0.62),   # tight-ish near diagonal
    ]
    # rebound's theta='uniform' draws from python's `random`; seed it for
    # reproducibility of the test sim construction.
    random.seed(seed)
    sims = [get_simulation(p12, p23) for (p12, p23) in pairs]
    return pairs, sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--complexity",
        type=str,
        default=None,
        help="If set, also test the NN+PySR path at this complexity (e.g. 26).",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA (default: auto-detect).",
    )
    args = parser.parse_args()

    use_cuda = args.cuda or torch.cuda.is_available()

    with open(os.path.join(REPO_ROOT, "official_versions.json")) as f:
        versions = json.load(f)
    nn_version = versions["nn_version"]
    pysr_version = versions["pysr_version"]
    default_complexity = versions["pysr_model_selection"]

    pairs, sims = make_test_sims(seed=0)

    # ---------- NN-only path ------------------------------------------------
    print("=" * 60)
    print(f"NN-only path  (nn_version={nn_version})")
    print("=" * 60)
    existing_model = spock_reg_model.load(nn_version)
    existing_reg = spock.NonSwagFeatureRegressor(existing_model, cuda=use_cuda)
    new_pred = StabilityPredictor.load(cuda=use_cuda)

    n_ok = 0
    n_total = 0
    for (p12, p23), sim in zip(pairs, sims):
        sim_a = sim.copy()
        sim_b = sim.copy()

        # set the same rebound rng before each call so any internal sampling
        # is identical between the two paths.
        torch.manual_seed(0)
        old_out = existing_predict_nn(existing_reg, sim_a)

        torch.manual_seed(0)
        new_out = new_pred.predict(sim_b)

        n_total += 1
        if old_out is None and new_out is None:
            print(f"  ({p12:.2f}, {p23:.2f})  both: None (escape/encounter)")
            n_ok += 1
            continue
        if old_out is None or new_out is None:
            print(
                f"  ({p12:.2f}, {p23:.2f})  MISMATCH (one None): "
                f"old={old_out}  new={new_out}"
            )
            continue

        same_mean = np.isclose(
            old_out["mean"], new_out["mean"], rtol=0, atol=0
        )
        same_std = np.isclose(old_out["std"], new_out["std"], rtol=0, atol=0)
        status = "OK " if (same_mean and same_std) else "MISMATCH"
        print(
            f"  ({p12:.2f}, {p23:.2f})  {status}  "
            f"old=({old_out['mean']:+.6f}, {old_out['std']:.6f})  "
            f"new=({new_out['mean']:+.6f}, {new_out['std']:.6f})"
        )
        if same_mean and same_std:
            n_ok += 1
    print(f"NN-only path: {n_ok}/{n_total} exact match")

    nn_pass = n_ok == n_total

    # ---------- NN + PySR f2 path ------------------------------------------
    complexity = args.complexity if args.complexity is not None else default_complexity
    print()
    print("=" * 60)
    print(
        f"NN + PySR f2 path  "
        f"(nn_version={nn_version}, pysr_version={pysr_version}, "
        f"complexity={complexity})"
    )
    print("=" * 60)

    existing_eq_model = spock_reg_model.load_with_pysr_f2(
        nn_version,
        pysr_version,
        pysr_model_selection=complexity,
        pysr_dir=os.path.join(REPO_ROOT, "sr_results/"),
    )
    existing_eq_reg = spock.NonSwagFeatureRegressor(
        existing_eq_model, cuda=use_cuda
    )
    new_eq_pred = StabilityPredictor.load(complexity=complexity, cuda=use_cuda)

    n_ok = 0
    n_total = 0
    for (p12, p23), sim in zip(pairs, sims):
        sim_a = sim.copy()
        sim_b = sim.copy()

        torch.manual_seed(0)
        old_out = existing_predict_nn(existing_eq_reg, sim_a)

        torch.manual_seed(0)
        new_out = new_eq_pred.predict(sim_b)

        n_total += 1
        if old_out is None and new_out is None:
            print(f"  ({p12:.2f}, {p23:.2f})  both: None (escape/encounter)")
            n_ok += 1
            continue
        if old_out is None or new_out is None:
            print(
                f"  ({p12:.2f}, {p23:.2f})  MISMATCH (one None): "
                f"old={old_out}  new={new_out}"
            )
            continue
        same_mean = np.isclose(
            old_out["mean"], new_out["mean"], rtol=0, atol=0
        )
        same_std = np.isclose(old_out["std"], new_out["std"], rtol=0, atol=0)
        status = "OK " if (same_mean and same_std) else "MISMATCH"
        print(
            f"  ({p12:.2f}, {p23:.2f})  {status}  "
            f"old=({old_out['mean']:+.6f}, {old_out['std']:.6f})  "
            f"new=({new_out['mean']:+.6f}, {new_out['std']:.6f})"
        )
        if same_mean and same_std:
            n_ok += 1
    print(f"NN + PySR path: {n_ok}/{n_total} exact match")
    eq_pass = n_ok == n_total

    print()
    print("=" * 60)
    if nn_pass and eq_pass:
        print("ALL EQUIVALENCE CHECKS PASSED")
        sys.exit(0)
    else:
        print("FAILURES: see mismatches above")
        sys.exit(1)


if __name__ == "__main__":
    main()
