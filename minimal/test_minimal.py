"""Verify minimal/planet_stability.py matches the existing pipeline.

Compares against the existing StabilityPredictor (from planet_eqs) on:
  - the default complexity (26)
  - a sweep across other complexities (1, 7, 14, 22, 29)

For each, the minimal predictor must return the same `mean` as the
existing predictor's predict()['mean'].
"""

from __future__ import annotations

import os
import sys
import random

import numpy as np
import rebound

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINIMAL_ROOT = os.path.join(REPO_ROOT, "minimal")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "figures"))
sys.path.insert(0, MINIMAL_ROOT)

from planet_eqs import StabilityPredictor as FullPredictor  # noqa: E402
from planet_stability import StabilityPredictor as MinimalPredictor  # noqa: E402


def make_sims(seed=0):
    random.seed(seed)
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
    sims = []
    for p12, p23 in pairs:
        sim = rebound.Simulation()
        sim.integrator = "whfast"
        sim.ri_whfast.safe_mode = 0
        sim.add(m=1.0)
        sim.add(m=1e-4, P=1.0, theta="uniform")
        sim.add(m=1e-4, P=1.0 / p12, theta="uniform")
        sim.add(m=1e-4, P=1.0 / p12 / p23, theta="uniform")
        sim.move_to_com()
        sims.append(sim)
    return pairs, sims


def run_one_complexity(c, pairs, sims, tol=1e-4):
    print("=" * 60)
    print(f"complexity = {c}")
    print("=" * 60)
    full = FullPredictor.load(complexity=c)
    minimal = MinimalPredictor.load(
        os.path.join(REPO_ROOT, "minimal", "planet_stability.npz"),
        complexity=c,
    )
    n_ok = 0
    n_total = 0
    for (p12, p23), sim in zip(pairs, sims):
        f_out = full.predict(sim.copy())
        m_out = minimal.predict(sim.copy())
        n_total += 1
        if f_out is None and m_out is None:
            print(f"  ({p12:.3f}, {p23:.3f})  both None")
            n_ok += 1
            continue
        if (f_out is None) != (m_out is None):
            print(
                f"  ({p12:.3f}, {p23:.3f})  MISMATCH (one None): "
                f"full={f_out}  minimal={m_out}"
            )
            continue
        # `full` returns dict with mean+std; `minimal` returns just the
        # mean (float). Compare means; ignore std.
        dmu = abs(f_out["mean"] - m_out)
        ok = dmu < tol
        if ok:
            n_ok += 1
        status = "OK " if ok else "MISMATCH"
        print(
            f"  ({p12:.3f}, {p23:.3f})  {status}  "
            f"full_mean={f_out['mean']:+.6f}  "
            f"minimal={m_out:+.6f}  |dmu|={dmu:.2e}"
        )
    print(f"  -> {n_ok}/{n_total}")
    return n_ok == n_total


def main():
    pairs, sims = make_sims(seed=0)
    complexities_to_test = [1, 7, 14, 22, 26, 29]
    print(f"Testing minimal predictor at complexities {complexities_to_test}\n")
    results = {c: run_one_complexity(c, pairs, sims) for c in complexities_to_test}
    print()
    print("=" * 60)
    print("summary:")
    for c, ok in results.items():
        print(f"  c={c:>3d}  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
