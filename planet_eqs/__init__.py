"""Predict planetary-system instability time from a rebound simulation.

Recommended public API for external users:

    >>> from planet_eqs import LitePredictor
    >>> model = LitePredictor.load(complexity=26)   # or load() for the BNN
    >>> out = model.predict(sim)                    # {'mean': ..., 'std': ...}
    >>> print(model.equation())                     # the symbolic formula

LitePredictor needs only torch, numpy, rebound, sympy — no pysr,
pytorch_lightning, or sklearn. Artifacts live under planet_eqs/data/
and are produced by scripts/build_artifacts.py.

StabilityPredictor is a thin wrapper around the existing training-time
code path (spock_reg_model.VarModel + figures.spock.NonSwagFeatureRegressor).
It is kept for development / debugging — it produces bitwise-identical
predictions to LitePredictor but pulls in the full dep stack.
"""

from .predictor import StabilityPredictor
from ._lite import LitePredictor

__all__ = ["LitePredictor", "StabilityPredictor"]
