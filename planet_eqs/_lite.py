"""Lite (deps-minimized) inference path.

Reads the artifacts produced by ``scripts/build_artifacts.py`` and exposes
the same ``predict(sim)`` interface as ``StabilityPredictor``. Runtime
deps: numpy, torch, rebound, sympy.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import sympy
import torch
from torch import nn

from . import _features
from ._eq_compiler import compile_expr
from ._model import MinimalVarModel


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_manifest(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "manifest.json")) as f:
        return json.load(f)


def _build_nn(nn_version: int, *, data_dir: str) -> MinimalVarModel:
    manifest = _load_manifest(data_dir)
    fname = manifest["nn_files"][str(nn_version)]
    path = os.path.join(data_dir, fname)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    hparams = blob["hparams"]
    state_dict = blob["state_dict"]

    model = MinimalVarModel(hparams)
    # Pytorch Lightning's state_dict prefix is the same as a plain
    # nn.Module's (no "model." prefix), so this matches directly.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in checkpoint state_dict for nn_version="
            f"{nn_version}: {unexpected}. The lite MinimalVarModel does "
            f"not match the stored architecture — check hparams."
        )
    # `missing` is allowed to include training-only buffers we don't
    # bother registering. Print a one-line warning if anything besides
    # those goes missing.
    if missing:
        # The official checkpoint has no extras beyond the inference path.
        # If we see misses, surface them so we can fix _model.py.
        raise RuntimeError(
            f"Missing keys when loading nn_version={nn_version}: {missing}"
        )
    model.eval()
    return model


def _SymbolicHead(equation_strs, feature_names) -> nn.Module:
    """Build the (B, summary_dim) -> (B, 2) head from the saved sympy
    equations. If there's only one head emitted by PySR (regression on
    the mean), we emit a dummy std=1 column to match the (B, 2) shape
    that the original ``modules.PySRNet`` produces."""

    # Parse strings back into sympy expressions. The save format always
    # uses str(expr); sympy.sympify should round-trip.
    exprs = []
    for s in equation_strs:
        try:
            exprs.append(sympy.sympify(s))
        except (sympy.SympifyError, SyntaxError) as e:
            raise RuntimeError(
                f"Could not parse stored equation back into sympy: {s!r}\n"
                f"sympy says: {e}"
            )

    compiled = nn.ModuleList(
        [compile_expr(e, feature_names) for e in exprs]
    )

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.compiled = compiled
            self.n_heads = len(compiled)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            outs = [m(x) for m in self.compiled]
            stacked = torch.stack(outs, dim=-1)   # (B, n_heads)
            if self.n_heads == 1:
                # The existing PySRNet pads with a dummy std=1 column.
                ones = torch.ones_like(stacked)
                stacked = torch.cat([stacked, ones], dim=-1)
            return stacked

    return _Head()


class LitePredictor:
    """Predictor backed by *only* the exported artifacts.

    Same surface as ``StabilityPredictor`` (``load`` / ``predict`` /
    ``predict_batch``). Does not import ``pysr`` or ``pytorch_lightning``
    or ``sklearn``. The repo source tree is irrelevant at runtime.
    """

    def __init__(
        self,
        nn_model: MinimalVarModel,
        *,
        nn_version: int,
        pysr_version: int | None,
        complexity: int | None,
        device: torch.device,
    ):
        self._nn = nn_model.to(device).eval()
        self.nn_version = nn_version
        self.pysr_version = pysr_version
        self.complexity = complexity
        self.device = device

    # --------------------------------------------------------------- load
    @classmethod
    def load(
        cls,
        complexity: int | None = None,
        *,
        nn_version: int | None = None,
        pysr_version: int | None = None,
        data_dir: str = _DATA_DIR,
        cuda: bool | None = None,
    ) -> "LitePredictor":
        manifest = _load_manifest(data_dir)

        # Resolve defaults from the manifest: the first pair is "default".
        default_pair = manifest["pairs"][0]
        if nn_version is None:
            nn_version = default_pair["nn_version"]
        if complexity is not None and pysr_version is None:
            # If we're using a complexity, default to the pysr_version
            # paired with our nn_version in the manifest; otherwise fall
            # back to the default pair's pysr_version.
            paired = None
            for p in manifest["pairs"]:
                if p["nn_version"] == nn_version:
                    paired = p["pysr_version"]
                    break
            pysr_version = paired or default_pair["pysr_version"]

        device = torch.device(
            "cuda" if (cuda if cuda is not None else torch.cuda.is_available())
            else "cpu"
        )
        nn_model = _build_nn(int(nn_version), data_dir=data_dir)

        if complexity is not None:
            # Plug in the symbolic head.
            eqs_meta = manifest["pysr_files"][str(int(pysr_version))]
            with open(os.path.join(data_dir, eqs_meta["file"])) as f:
                eqs = json.load(f)
            feature_names = eqs["feature_names"]
            # The closest-complexity rule, applied at *export time* per head.
            avail = sorted(int(c) for c in eqs["complexities"])
            closest = min(avail, key=lambda c: abs(c - int(complexity)))
            head_map = eqs["complexities"][str(closest)]
            equation_strs = [
                head_map[f"head_{i}"] for i in range(eqs["n_heads"])
            ]
            head = _SymbolicHead(equation_strs, feature_names)
            nn_model.set_regress_nn(head, is_symbolic=True)
            complexity = closest

        return cls(
            nn_model,
            nn_version=int(nn_version),
            pysr_version=int(pysr_version) if pysr_version is not None else None,
            complexity=complexity,
            device=device,
        )

    # ------------------------------------------------------------- predict
    def predict(
        self,
        sim,
        *,
        indices=None,
        return_features: bool = False,
    ) -> dict[str, Any] | None:
        """Predict log10(T_inst) for ``sim``. Same contract as
        ``StabilityPredictor.predict``."""
        import rebound  # local import; only needed for the Escape catch.

        sim_prepped = sim.copy()
        sim_prepped.dt = 0.05
        sim_prepped.init_megno()
        sim_prepped.exit_max_distance = 20.0

        try:
            X = _features.features_from_sim(sim_prepped, indices=indices)
        except rebound.Escape:
            return None
        if X is None:
            return None

        X_scaled = _features.standard_scale(X)
        x_t = torch.from_numpy(X_scaled.astype(np.float32)).to(self.device)

        with torch.no_grad():
            out = self._nn(x_t, return_intermediates=True)

        mean = float(out["mean"][0, 0].detach().cpu().numpy())
        std = float(out["std"][0, 0].detach().cpu().numpy())
        result: dict[str, Any] = {"mean": mean, "std": std}
        if return_features:
            result["summary_stats"] = (
                out["summary_stats"][0].detach().cpu().numpy()
            )
            result["f1"] = result["summary_stats"]
        return result

    def predict_batch(self, sims, *, indices=None):
        return [self.predict(sim, indices=indices) for sim in sims]

    # ---------------------------------------------------------- equation
    def equation(self) -> str | None:
        if self.complexity is None or self.pysr_version is None:
            return None
        manifest = _load_manifest(_DATA_DIR)
        eqs_meta = manifest["pysr_files"][str(self.pysr_version)]
        with open(os.path.join(_DATA_DIR, eqs_meta["file"])) as f:
            eqs = json.load(f)
        head_map = eqs["complexities"][str(int(self.complexity))]
        lines = []
        for i in range(eqs["n_heads"]):
            lines.append(f"out_{i} = {head_map[f'head_{i}']}")
        return "\n".join(lines)
