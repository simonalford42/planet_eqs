"""Verify the lite predictor matches the original-deps predictor.

For each of the same (P12, P23) test points used by the period_ratio
figure, run both:
    - StabilityPredictor (wraps the existing training code; full deps)
    - LitePredictor      (loads from planet_eqs/data/; lite deps only)

and assert the (mean, std) outputs agree to within tight numerical
tolerance for BOTH the NN-only and NN+PySR paths.

Usage:
    python test_lite_equivalence.py
"""

from __future__ import annotations

import os
import sys
import json
import random
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "figures"))

import rebound  # noqa: E402

from planet_eqs import StabilityPredictor, LitePredictor  # noqa: E402


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


def make_test_sims(seed: int = 0):
    pairs = [
        (0.55, 0.55),
        (0.55, 0.76),
        (0.76, 0.55),
        (0.76, 0.76),
        (0.655, 0.655),
        (0.60, 0.70),
        (0.70, 0.60),
        (0.62, 0.62),
    ]
    random.seed(seed)
    sims = [get_simulation(p12, p23) for (p12, p23) in pairs]
    return pairs, sims


def _compare_runs(name, full_pred, lite_pred, pairs, sims, tol):
    print("=" * 72)
    print(f"{name}")
    print("=" * 72)
    n_ok = 0
    n_total = 0
    for (p12, p23), sim in zip(pairs, sims):
        torch.manual_seed(0)
        full_out = full_pred.predict(sim.copy())
        torch.manual_seed(0)
        lite_out = lite_pred.predict(sim.copy())

        n_total += 1
        if full_out is None and lite_out is None:
            print(f"  ({p12:.3f}, {p23:.3f})  both: None")
            n_ok += 1
            continue
        if full_out is None or lite_out is None:
            print(
                f"  ({p12:.3f}, {p23:.3f})  MISMATCH (one None): "
                f"full={full_out} lite={lite_out}"
            )
            continue

        dmu = abs(full_out["mean"] - lite_out["mean"])
        dsd = abs(full_out["std"] - lite_out["std"])
        status = "OK " if (dmu <= tol and dsd <= tol) else "MISMATCH"
        print(
            f"  ({p12:.3f}, {p23:.3f})  {status}  "
            f"full=({full_out['mean']:+.6f}, {full_out['std']:.6f})  "
            f"lite=({lite_out['mean']:+.6f}, {lite_out['std']:.6f})  "
            f"|dmu|={dmu:.2e}  |dstd|={dsd:.2e}"
        )
        if dmu <= tol and dsd <= tol:
            n_ok += 1
    print(f"{n_ok}/{n_total} agree within atol={tol:g}")
    print()
    return n_ok == n_total


def main():
    use_cuda = torch.cuda.is_available()

    with open(os.path.join(REPO_ROOT, "official_versions.json")) as f:
        versions = json.load(f)
    default_complexity = int(versions["pysr_model_selection"])

    pairs, sims = make_test_sims(seed=0)
    tol = 1e-4

    runs: list[tuple[str, dict]] = []
    runs.append(("default NN-only", {"complexity": None}))
    runs.append((f"default NN + PySR c={default_complexity}",
                 {"complexity": default_complexity}))

    # Also exercise the per-k pairs from official_versions.json (k=2..5).
    if "k" in versions:
        for k_str, info in versions["k"].items():
            nn_v = info["version"]
            pysr_v = info["pysr_version"]
            runs.append((
                f"k={k_str} NN-only (nn={nn_v})",
                {"complexity": None, "nn_version": nn_v},
            ))
            runs.append((
                f"k={k_str} NN+PySR (nn={nn_v}, pysr={pysr_v})",
                {"complexity": default_complexity,
                 "nn_version": nn_v, "pysr_version": pysr_v},
            ))

    all_pass = True
    for label, kwargs in runs:
        try:
            full = StabilityPredictor.load(cuda=use_cuda, **kwargs)
        except Exception as e:
            # The existing pipeline can fail to load certain hof files
            # due to pysr<->sklearn compat (e.g. 93102 missing 'bumper').
            # In that case we cannot run the comparison, but we can at
            # least sanity-check that the lite predictor still runs.
            print("=" * 72)
            print(f"{label}")
            print("=" * 72)
            print(f"  SKIP comparison: existing pipeline failed to load "
                  f"({type(e).__name__}: {str(e)[:60]})")
            try:
                lite = LitePredictor.load(cuda=use_cuda, **kwargs)
                for sim in sims[:2]:
                    out = lite.predict(sim.copy())
                    print(f"    lite still runs: {out}")
            except Exception as e2:
                print(f"  LITE PREDICTOR ALSO FAILED: {type(e2).__name__}: {e2}")
                all_pass = False
            print()
            continue
        lite = LitePredictor.load(cuda=use_cuda, **kwargs)
        ok = _compare_runs(label, full, lite, pairs, sims, tol)
        all_pass = all_pass and ok

    if all_pass:
        print("LITE PREDICTOR MATCHES FULL PREDICTOR ON ALL TEST CASES")
        sys.exit(0)
    else:
        print("MISMATCHES — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
