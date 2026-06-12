"""Single-file predictor for planetary-system instability time.

Pinned to the published default model:
    NN checkpoint 24880,
    PySR hall-of-fame 11003,
    selectable equation complexity (default: 26).

Usage:

    >>> import rebound
    >>> from planet_stability import StabilityPredictor
    >>> model = StabilityPredictor.load()    # uses default complexity=26
    >>> # ...build a 3-planet rebound sim...
    >>> model.predict(sim)
    8.395...

Returns ``log10(T_inst / P_inner)``, clamped to [4, 12]. ``None`` if the
rebound short integration aborted (collision / escape).

Runtime dependencies: numpy, rebound.

Files this module needs (defaults look next to the .py):
    planet_stability.npz             - f1 weight (10, 31) + bias (10,)
    planet_stability_equations.json  - {complexity: python_expr_str}

Architecture note
-----------------
The published training-time pipeline builds a 41-element feature vector
(time + e+/e-/MMR for two pairs + MEGNO + 3 planets * 6 orbital elements
with angles split into cos/sin + 3 masses + 3 NaN flags), standardizes
it with a fixed mean/scale, then runs a 41 -> 10 masked linear layer.

We compress all of that:

  1. The 10 input positions whose contribution to the final answer is
     zero (e+/e-, MMR strength, MEGNO, NaN flags) are dropped from
     every layer of the pipeline. We never compute them.
  2. The fixed standardization ``(x - mean) / scale`` is folded into
     the masked linear layer:
         feats = (x - mean)/scale @ W.T    =>    feats = x @ W'.T + b
     where W' = W / scale and b = -W' @ mean. So mu1 ends up as e.g.
     ``+209.781 * (a1/a1_init) + 85414.854 * m1 - 210.769``,
     reading the trained "physics units" off the weights directly.

Pipeline, top-to-bottom:
  1. Rebound short integration recording orbital elements only
  2. Angle expansion (Omega, pomega, theta -> cos/sin pairs) + masses
  3. One numpy matmul + bias (the NN feature extractor) + summary stats
  4. eval() of the chosen symbolic equation
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import rebound


_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(_HERE, "planet_stability.npz")
DEFAULT_EQUATIONS = os.path.join(_HERE, "planet_stability_equations.json")
DEFAULT_COMPLEXITY = 26


def load_equations(path: str = DEFAULT_EQUATIONS) -> dict[int, str]:
    """Load the equations JSON. Keys are int complexity, values are
    Python expressions in m0..m9, s0..s9, optionally calling ``sin``."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# Column indices in the per-timestep recording (before the angle
# expansion + mass append) that hold angles to be split into (cos, sin).
# The 19-column recording is:
#   0:           t / min_P
#   1, 2, 3:     planet 1 (a/a1_init, e, i)
#   4, 5, 6:     planet 1 angles (Omega, pomega, theta)
#   7, 8, 9:     planet 2 (a/a1_init, e, i)
#   10, 11, 12:  planet 2 angles
#   13, 14, 15:  planet 3 (a/a1_init, e, i)
#   16, 17, 18:  planet 3 angles
_ANGLE_RAW_INDICES = (4, 5, 6, 10, 11, 12, 16, 17, 18)


# =====================================================================
# Rebound integration: 10000 inner-orbits of WHFast, sampled 1000 times,
# subsampled to 100. We only record orbital elements per planet — no
# MEGNO, no MMR strength, no e+/e-. Returns the (100, 31) *scaled*
# feature matrix, or None on collision / escape.
# =====================================================================
def _populate_row(sim: rebound.Simulation, trio, row: np.ndarray,
                  a1_init: float, min_P: float) -> None:
    row[0] = sim.t / min_P
    orbits = sim.calculate_orbits()
    for j, k in enumerate(trio):
        o = orbits[k - 1]
        base = 1 + 6 * j
        row[base    ] = o.a / a1_init
        row[base + 1] = o.e
        row[base + 2] = o.inc
        row[base + 3] = o.Omega
        row[base + 4] = o.pomega
        row[base + 5] = o.theta


def _extract_features(sim: rebound.Simulation) -> np.ndarray | None:
    """Integrate + extract the (100, 31) scaled feature matrix.

    Uses the first trio (planets 1, 2, 3). Returns None on
    collision / escape.
    """
    if sim.N_real < 4:
        raise ValueError("StabilityPredictor needs 1 star + 3 planets minimum")
    sim = sim.copy()
    sim.dt = 0.05
    sim.exit_max_distance = 20.0
    # init_megno() adds variational ("ghost") particles. We do NOT use
    # the MEGNO value (it's a zeroed feature), but the ghost particles
    # subtly change the integration of the real planets at floating-
    # point precision. Keeping init_megno() preserves bitwise equivalence
    # to the published pipeline; without it, chaotic systems can diverge
    # by ~0.1 in the predicted log10(T_inst) on borderline trajectories.
    sim.init_megno()

    trio = [1, 2, 3]
    a1_init = sim.particles[trio[0]].a
    min_P = min(p.P for p in sim.particles[1: sim.N_real])

    Nout = 1000
    times = np.linspace(0, 10_000 * abs(min_P), Nout)
    raw = np.zeros((Nout, 19))
    for i, t in enumerate(times):
        try:
            sim.integrate(t, exact_finish_time=0)
        except (rebound.Collision, rebound.Escape):
            return None
        _populate_row(sim, trio, raw[i], a1_init, min_P)

    # Subsample 1000 -> 100 timesteps.
    cur = raw[::10]                                       # (100, 19)

    # Append the 3 mass ratios (broadcast over time).
    masses = np.array([sim.particles[j].m / sim.particles[0].m for j in trio])
    X = np.concatenate([cur, np.tile(masses, (100, 1))], axis=1)  # (100, 22)

    # Expand the 9 angle columns into (cos, sin) pairs.
    cols = []
    for j in range(X.shape[1]):
        if j in _ANGLE_RAW_INDICES:
            cols.append(np.cos(X[:, j:j+1]))
            cols.append(np.sin(X[:, j:j+1]))
        else:
            cols.append(X[:, j:j+1])
    X = np.concatenate(cols, axis=1)                      # (100, 31)
    assert X.shape == (100, 31)
    return X


# =====================================================================
# The predictor. The NN is one numpy matmul, the equation is one
# eval() of a Python expression.
# =====================================================================
class StabilityPredictor:
    """Predict log10(T_inst / P_inner) for a 3-planet rebound simulation."""

    MIN_PRED = 4.0
    MAX_PRED = 12.0

    # Limit eval() lookups to a known-safe namespace.
    _SAFE_GLOBALS = {"__builtins__": {}, "sin": np.sin, "cos": np.cos,
                     "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                     "abs": np.abs}

    def __init__(
        self,
        f1_weight: np.ndarray,
        f1_bias: np.ndarray,
        equations: dict[int, str],
        complexity: int,
    ):
        # The f1 layer is the masked-linear of the published model with
        # both the input mask and the StandardScaler folded in:
        #     feats = X @ f1_weight.T + f1_bias
        # See the module docstring for the derivation.
        if f1_weight.shape != (10, 31) or f1_bias.shape != (10,):
            raise ValueError(
                f"expected shapes (10, 31) and (10,), got "
                f"{f1_weight.shape} and {f1_bias.shape}"
            )
        self.f1_weight = f1_weight
        self.f1_bias = f1_bias
        self.equations = equations
        self.complexity = complexity
        self._equation_code = compile(
            equations[complexity], f"<equation c={complexity}>", "eval"
        )

    @classmethod
    def load(
        cls,
        weights_path: str = DEFAULT_WEIGHTS,
        equations_path: str = DEFAULT_EQUATIONS,
        complexity: int = DEFAULT_COMPLEXITY,
    ) -> "StabilityPredictor":
        """Load the predictor.

        Args:
            weights_path: ``planet_stability.npz``. Defaults to the file
                next to this module.
            equations_path: ``planet_stability_equations.json``. Defaults
                to the file next to this module.
            complexity: which mean-equation to use. Defaults to 26 —
                the official complexity in the paper.

        See ``StabilityPredictor.available_complexities()`` for choices.
        """
        with np.load(weights_path) as blob:
            f1_weight = np.array(blob["f1_weight"], dtype=np.float64)
            f1_bias = np.array(blob["f1_bias"], dtype=np.float64)
        equations = load_equations(equations_path)
        if complexity not in equations:
            raise ValueError(
                f"unknown complexity {complexity!r}. Available: "
                f"{sorted(equations)}"
            )
        return cls(f1_weight, f1_bias, equations, complexity)

    def set_complexity(self, complexity: int) -> None:
        """Switch the active equation without reloading weights."""
        if complexity not in self.equations:
            raise ValueError(
                f"unknown complexity {complexity!r}. Available: "
                f"{sorted(self.equations)}"
            )
        self.complexity = complexity
        self._equation_code = compile(
            self.equations[complexity],
            f"<equation c={complexity}>",
            "eval",
        )

    def available_complexities(self) -> list[int]:
        return sorted(self.equations)

    def equation(self) -> str:
        """Return the Python source of the equation in use."""
        return self.equations[self.complexity]

    def _summary_stats(self, X_raw: np.ndarray) -> np.ndarray:
        """X_raw: (100, 31), unscaled. Returns (20,) summary stats —
        10 means then 10 stds of the learned features."""
        feats = X_raw @ self.f1_weight.T + self.f1_bias    # (100, 10)
        m = feats.mean(axis=0)
        # Match torch.std (unbiased, ddof=1), then add EPSILON in
        # variance space before sqrt — same as the published model.
        var = feats.var(axis=0, ddof=1)
        s = np.sqrt(np.abs(var) + 1e-5)
        return np.concatenate([m, s])                      # (20,)

    def predict(self, sim: rebound.Simulation) -> float | None:
        """Run the full pipeline. Returns log10(T_inst / P_inner) or
        ``None`` if the rebound short integration aborted.
        """
        X = _extract_features(sim)
        if X is None:
            return None
        return self.predict_from_features(X)

    def predict_from_features(self, X: np.ndarray) -> float:
        """Predict from a pre-extracted ``(100, 31)`` raw-feature matrix.

        This skips the rebound integration step — useful for batch
        evaluation when the features were precomputed.

        Args:
            X: shape ``(100, 31)``, columns in the layout described in
               the module docstring. *Not* standardized — the scaler is
               already folded into ``f1_weight`` / ``f1_bias``.
        """
        if X.shape != (100, 31):
            raise ValueError(f"expected X shape (100, 31), got {X.shape}")
        summary = self._summary_stats(X)
        local_vars = {f"m{i}": summary[i]      for i in range(10)}
        local_vars.update({f"s{i}": summary[10 + i] for i in range(10)})
        mean_pred = float(eval(self._equation_code, self._SAFE_GLOBALS, local_vars))
        return float(np.clip(mean_pred, self.MIN_PRED, self.MAX_PRED))


__all__ = ["StabilityPredictor", "DEFAULT_COMPLEXITY", "load_equations"]


# =====================================================================
# Example: build a 3-planet system in a mildly-resonant configuration
# and predict its log10(T_inst / P_inner).
# =====================================================================
if __name__ == "__main__":
    model = StabilityPredictor.load()
    print(f"Using equation complexity = {model.complexity}")
    print(f"  {model.equation()}")
    print()

    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.add(m=1.0)                                  # star (solar mass)
    sim.add(m=1e-4, P=1.0,         theta="uniform")  # planet 1 (P=1)
    sim.add(m=1e-4, P=1.0 / 0.655, theta="uniform")  # planet 2 (P=1.527)
    sim.add(m=1e-4, P=1.0 / 0.655 / 0.655, theta="uniform")  # planet 3
    sim.move_to_com()

    log_t_inst = model.predict(sim)
    print(f"log10(T_inst / P_inner) = {log_t_inst:.4f}")
    print(f"  (means the system is expected to go unstable after about")
    print(f"   10**{log_t_inst:.2f} ~= {10**log_t_inst:.2e} inner-planet periods)")
