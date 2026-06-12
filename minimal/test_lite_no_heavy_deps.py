"""Confirm LitePredictor runs *without* pysr / pytorch_lightning / sklearn.

We do this by installing import-blocker hooks at the very start of the
process, before any planet_eqs / torch import. Any attempt to import the
blocked modules raises ImportError, simulating a fresh environment with
only ``torch numpy rebound sympy`` available.

Usage:
    python test_lite_no_heavy_deps.py
"""

from __future__ import annotations

import sys


# Block heavy deps BEFORE any other imports.
_BLOCKED = {"pysr", "pytorch_lightning", "sklearn"}


class _Blocker:
    @classmethod
    def find_module(cls, name, path=None):
        # py2-style — needed for some old import systems
        top = name.split(".", 1)[0]
        return cls if top in _BLOCKED else None

    def load_module(self, name):
        raise ImportError(
            f"planet_eqs lite-deps check: import of {name!r} is BLOCKED. "
            f"The lite predictor must not need this module at runtime."
        )

    @classmethod
    def find_spec(cls, name, path, target=None):
        top = name.split(".", 1)[0]
        if top in _BLOCKED:
            # Build a spec whose loader raises on import.
            import importlib.machinery
            loader = cls()
            return importlib.machinery.ModuleSpec(name, loader)
        return None


# Insert at the FRONT so we beat the normal finders.
sys.meta_path.insert(0, _Blocker)

# Also strip any cached imports of the blocked modules.
for mod in list(sys.modules):
    if mod.split(".", 1)[0] in _BLOCKED:
        del sys.modules[mod]


# ----- now do the actual test ----------------------------------------------

import os
import json
import random

import numpy as np  # noqa: E402
import rebound  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from planet_eqs import LitePredictor  # noqa: E402


def get_simulation(P12, P23):
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.add(m=1.0)
    sim.add(m=1e-4, P=1.0, theta="uniform")
    sim.add(m=1e-4, P=1.0 / P12, theta="uniform")
    sim.add(m=1e-4, P=1.0 / P12 / P23, theta="uniform")
    sim.move_to_com()
    return sim


def main():
    # Sanity: confirm the import blocks are active.
    for name in _BLOCKED:
        try:
            __import__(name)
        except ImportError:
            print(f"  block confirmed: {name!r} is unimportable")
        else:
            print(f"  ERROR: {name!r} should be unimportable")
            sys.exit(1)
    print()

    random.seed(0)
    sims = [get_simulation(p12, p23) for (p12, p23) in
            [(0.55, 0.55), (0.655, 0.655), (0.60, 0.70)]]

    # NN-only
    print("Loading LitePredictor (NN-only)...")
    nn_only = LitePredictor.load(cuda=False)
    for sim in sims:
        out = nn_only.predict(sim)
        print(f"  predict -> {out}")
    print()

    # NN + PySR
    print("Loading LitePredictor (NN + PySR complexity=26)...")
    with_eq = LitePredictor.load(complexity=26, cuda=False)
    for sim in sims:
        out = with_eq.predict(sim)
        print(f"  predict -> {out}")
    print(f"  equation:\n{with_eq.equation()}")
    print()

    print("LITE PREDICTOR WORKS WITHOUT pysr / pytorch_lightning / sklearn")


if __name__ == "__main__":
    main()
