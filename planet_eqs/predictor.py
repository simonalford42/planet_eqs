"""StabilityPredictor: clean inference API for the planet-equations model.

Wraps the existing training-time code (spock_reg_model.VarModel,
modules.PySRNet, figures.spock.NonSwagFeatureRegressor) without modifying
it. Intended for external users who have a rebound Simulation and want a
log10(T_inst) prediction.

Pipeline:
    rebound.Simulation
      -> short integration + tseries  (figures.spock.tseries_feature_functions)
      -> StandardScaler                (figures.spock.regression, hardcoded)
      -> f1 feature NN                 (VarModel.feature_nn)
      -> mean/std summary stats        (VarModel.compute_summary_stats)
      -> f2 head                       (VarModel.regress_nn, OR PySRNet)
      -> (mu, std) of log10(T_inst)
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_VERSIONS_JSON = os.path.join(_REPO_ROOT, "official_versions.json")
_DEFAULT_SR_DIR = os.path.join(_REPO_ROOT, "sr_results")


def _ensure_repo_on_path() -> None:
    """The existing modules use top-level imports (`import modules`,
    `import spock_reg_model`, etc.). Make sure the repo root is importable
    regardless of CWD."""
    import sys

    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    figures_dir = os.path.join(_REPO_ROOT, "figures")
    if figures_dir not in sys.path:
        sys.path.insert(0, figures_dir)


class StabilityPredictor:
    """Predict log10(T_inst) for a 3+ planet system from a rebound Simulation.

    Use ``StabilityPredictor.load()`` rather than the constructor directly.

    Example:
        >>> import rebound
        >>> from planet_eqs import StabilityPredictor
        >>> model = StabilityPredictor.load(complexity=26)
        >>> sim = rebound.Simulation()
        >>> sim.add(m=1.0)
        >>> sim.add(m=1e-4, P=1.0, theta='uniform')
        >>> sim.add(m=1e-4, P=1.3, theta='uniform')
        >>> sim.add(m=1e-4, P=1.69, theta='uniform')
        >>> sim.move_to_com()
        >>> out = model.predict(sim)
        >>> out['mean'], out['std']
    """

    def __init__(self, regressor, *, nn_version, pysr_version, complexity):
        # regressor is a figures.spock.NonSwagFeatureRegressor wrapping a
        # VarModel (with regress_nn optionally replaced by PySRNet).
        self._regressor = regressor
        self.nn_version = nn_version
        self.pysr_version = pysr_version
        self.complexity = complexity

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        complexity: int | str | None = None,
        *,
        nn_version: int | None = None,
        pysr_version: int | None = None,
        versions_json: str = _DEFAULT_VERSIONS_JSON,
        sr_dir: str = _DEFAULT_SR_DIR,
        cuda: bool | None = None,
    ) -> "StabilityPredictor":
        """Build a predictor.

        Args:
            complexity: which PySR equation to use for the f2 head.
                - None (default): use the trained BNN f2 head.
                - int: use the PySR equation closest to this complexity.
                - "best" / "accuracy" / "score": PySR's own selection criteria.
            nn_version: override the f1+f2 NN checkpoint id.
                Defaults to ``nn_version`` in ``official_versions.json``.
            pysr_version: override the PySR hall-of-fame id.
                Defaults to ``pysr_version`` in ``official_versions.json``.
                Only used when ``complexity`` is not None.
            versions_json: path to the versions json. Defaults to
                ``official_versions.json`` in the repo root.
            sr_dir: directory containing ``{pysr_version}.pkl``.
                Defaults to ``sr_results/`` in the repo root.
            cuda: force cuda on/off. Default: use cuda if available.
        """
        _ensure_repo_on_path()

        import spock_reg_model  # noqa: E402  (repo-root import)
        from figures import spock  # noqa: E402

        if nn_version is None or (complexity is not None and pysr_version is None):
            with open(versions_json) as f:
                versions = json.load(f)
            if nn_version is None:
                nn_version = versions["nn_version"]
            if complexity is not None and pysr_version is None:
                pysr_version = versions["pysr_version"]

        if complexity is None:
            model = spock_reg_model.load(nn_version)
        else:
            model = spock_reg_model.load_with_pysr_f2(
                nn_version,
                pysr_version,
                pysr_model_selection=complexity,
                pysr_dir=sr_dir + ("/" if not sr_dir.endswith("/") else ""),
            )

        use_cuda = torch.cuda.is_available() if cuda is None else cuda
        regressor = spock.NonSwagFeatureRegressor(model, cuda=use_cuda)
        return cls(
            regressor,
            nn_version=nn_version,
            pysr_version=pysr_version,
            complexity=complexity,
        )

    # --------------------------------------------------------------- predict
    def predict(
        self,
        sim,
        *,
        indices: list[int] | None = None,
        return_features: bool = False,
    ) -> dict[str, Any]:
        """Predict log10(T_inst) for ``sim``.

        Args:
            sim: a ``rebound.Simulation`` with at least one star + 3 planets.
                The sim is copied internally and prepped (dt=0.05, MEGNO init,
                exit_max_distance=20) before integration; your sim is left
                untouched.
            indices: optionally restrict to a specific trio of planet indices.
            return_features: also return intermediate f1 outputs and summary
                stats (useful for symbolic-regression follow-ups).

        Returns:
            ``{'mean': float, 'std': float}``, or ``None`` if the rebound
            short integration aborted (escape / encounter) before features
            could be extracted. With ``return_features=True``, also includes
            ``'f1'`` (shape (100, n_features)) and ``'summary_stats'``
            (shape (n_summary,)).
        """
        import rebound  # local import; only needed for the Escape exception

        sim_prepped = sim.copy()
        sim_prepped.dt = 0.05
        sim_prepped.init_megno()
        sim_prepped.exit_max_distance = 20.0

        try:
            out_dict = self._regressor.predict(sim_prepped, indices=indices)
        except rebound.Escape:
            return None
        if out_dict is None:
            return None

        mean = out_dict["mean"][0, 0].detach().cpu().numpy()
        std = out_dict["std"][0, 0].detach().cpu().numpy()
        result: dict[str, Any] = {
            "mean": float(mean),
            "std": float(std),
        }
        if return_features:
            result["f1"] = out_dict["summary_stats"][0].detach().cpu().numpy()
            result["summary_stats"] = result["f1"]
        return result

    def predict_batch(
        self,
        sims,
        *,
        indices: list[int] | None = None,
    ) -> list[dict[str, float] | None]:
        """Convenience: run ``predict`` on a list of sims. Returns a list.

        Note: not yet vectorized across sims — each sim still goes through
        its own rebound short integration. This is here for API
        completeness; speedups will come later.
        """
        return [self.predict(sim, indices=indices) for sim in sims]

    # ------------------------------------------------------------- equations
    def equation(self) -> str | None:
        """Return the symbolic PySR equation(s) as text, or ``None`` if this
        predictor uses the BNN f2 head.
        """
        if self.complexity is None:
            return None
        _ensure_repo_on_path()
        import pickle

        path = os.path.join(_DEFAULT_SR_DIR, f"{self.pysr_version}.pkl")
        with open(path, "rb") as f:
            reg = pickle.load(f)

        eqs = reg.equations_
        eqs_list = eqs if isinstance(eqs, list) else [eqs]

        out_lines = []
        for i, df in enumerate(eqs_list):
            if isinstance(self.complexity, str):
                # PySR's own selection criteria — let PySR pick.
                try:
                    sel = reg.get_best(index=i) if len(eqs_list) > 1 else reg.get_best()
                    row = sel
                except Exception:
                    ix = df["complexity"].idxmax()
                    row = df.loc[ix]
            else:
                ix = (df["complexity"] - int(self.complexity)).abs().idxmin()
                row = df.loc[ix]
            out_lines.append(f"out_{i} = {row['equation']}")
        return "\n".join(out_lines)
