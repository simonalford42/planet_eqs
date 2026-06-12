"""Minimal VarModel rebuild — *inference only*.

Re-implements the forward path of ``spock_reg_model.VarModel`` as a plain
``nn.Module``, so we don't depend on ``pytorch_lightning`` at runtime.
Mirrors ``VarModel.forward(x, noisy_val=False, deterministic=True)``
exactly for the hparams used in the official checkpoint.

Only the hparams that actually affect inference are honored here; all
the training-side knobs (optimizer, SWA, augmentation, KL terms, noise
injection) are dropped.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

EPSILON = 1e-5


def _soft_clamp(x: torch.Tensor, lo: float, high: float) -> torch.Tensor:
    return 0.5 * (torch.tanh(x) + 1.0) * (high - lo) + lo


def _hard_clamp(x: torch.Tensor, lo: float, high: float) -> torch.Tensor:
    return torch.clamp(x, lo, high)


class _MaskedLinear(nn.Module):
    """Same shape contract as ``modules.MaskedLinear`` so state_dict keys
    match: ``linear.weight``, ``linear.bias`` (optional), ``mask``."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.mask = nn.Parameter(
            torch.ones(out_features, in_features), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)


def _build_feature_nn(hparams: dict[str, Any], n_features: int, latent: int) -> nn.Module:
    """Reconstruct the architecture of ``VarModel.feature_nn`` from
    hparams. Only the variants used by the official checkpoint are
    supported here — others raise NotImplementedError.
    """
    variant = hparams.get("f1_variant", "mlp")

    if variant == "linear":
        # All published linear-f1 checkpoints save under the MaskedLinear
        # layout (feature_nn.linear.weight + feature_nn.mask), even when
        # `prune_f1_topk=None`. The mask is just all-ones in that case.
        # Build MaskedLinear unconditionally so the state_dict loads.
        # bias=False per VarModel.get_feature_nn (the linear branch).
        return _MaskedLinear(n_features, latent, bias=False)
    if variant == "identity":
        return nn.Identity()
    if variant == "mlp":
        # We replicate modules.mlp here to avoid importing it (it pulls in
        # spock_reg_model). Note: the existing get_feature_nn() never
        # forwards hparams['act'] to modules.mlp, so the activation is
        # always ReLU in practice — match that exactly.
        act_cls = nn.ReLU
        depth = hparams["f1_depth"]
        hidden = hparams["hidden_dim"]
        if depth == -1:
            return nn.Linear(n_features, latent)
        layers = [nn.Linear(n_features, hidden), act_cls()]
        for _ in range(depth):
            layers += [nn.Linear(hidden, hidden), act_cls()]
        layers += [nn.Linear(hidden, latent)]
        return nn.Sequential(*layers)

    raise NotImplementedError(
        f"f1_variant={variant!r} is not yet supported in the lite model. "
        f"Either extend planet_eqs._model._build_feature_nn or load via the "
        f"original full pipeline."
    )


def _build_regress_nn(hparams: dict[str, Any], summary_dim: int) -> nn.Module:
    """Reconstruct ``VarModel.regress_nn`` from hparams. Officials use
    ``f2_variant='mlp'``; PySR is plugged in later by the lite loader."""
    variant = hparams.get("f2_variant", "mlp")

    if variant == "linear":
        return nn.Linear(summary_dim, 2)
    if variant == "mlp":
        # Same caveat as _build_feature_nn — get_regress_nn() never
        # forwards hparams['act'], so the activation is always ReLU.
        act_cls = nn.ReLU
        depth = hparams["f2_depth"]
        hidden = hparams["hidden_dim"]
        if depth == -1:
            return nn.Linear(summary_dim, 2)
        layers = [nn.Linear(summary_dim, hidden), act_cls()]
        for _ in range(depth):
            layers += [nn.Linear(hidden, hidden), act_cls()]
        layers += [nn.Linear(hidden, 2)]
        return nn.Sequential(*layers)

    raise NotImplementedError(
        f"f2_variant={variant!r} is not yet supported in the lite model."
    )


class MinimalVarModel(nn.Module):
    """Inference-only stand-in for ``spock_reg_model.VarModel``.

    Forward path mirrors ``VarModel.forward(x, noisy_val=False,
    deterministic=True)`` for hparams seen on the official model. Builds
    only the modules needed to satisfy the saved state_dict.

    Attributes:
        feature_nn: f1
        regress_nn: f2 (replaceable with a PySR module — see
            ``.set_regress_nn``)
    """

    # Feature index conventions in the (B, T, F) input tensor. Mirror
    # VarModel.__init__.
    megno_location = 7
    mmr_location = [3, 6]
    nan_location = [38, 39, 40]
    eplusminus_location = [1, 2, 4, 5]
    theta_locations = [15, 16, 24, 25, 33, 34]

    def __init__(self, hparams: dict[str, Any]):
        super().__init__()
        self.hparams_dict = dict(hparams)

        # n_features mirrors VarModel.__init__.
        time_series_features = hparams.get("time_series_features", 38 + 3)
        if time_series_features == 82:
            time_series_features = 41
        include_derivatives = hparams.get("include_derivatives", False)
        self.n_features = time_series_features * (1 + int(include_derivatives))
        if hparams.get("combined_mass_feature", False):
            self.n_features += 1

        latent = hparams["latent"]
        self.latent = latent
        self.feature_nn = _build_feature_nn(hparams, self.n_features, latent)

        summary_dim = latent * 2
        if hparams.get("no_std", False) or hparams.get("no_mean", False):
            summary_dim = latent
        if hparams.get("fix_megno", False):
            summary_dim += 2
        self.summary_dim = summary_dim

        self.regress_nn = _build_regress_nn(hparams, summary_dim)

        # These two tensors are present in the official state_dict but
        # not used during deterministic inference. We register them so
        # state_dict loading succeeds without strict=False trickery.
        self.input_noise_logvar = nn.Parameter(
            torch.zeros(self.n_features) - 2.0
        )
        self.summary_noise_logvar = nn.Parameter(
            torch.zeros(summary_dim) - 2.0
        )

        # Bookkeeping for the few conditional ops in forward.
        self.fix_megno = bool(hparams.get("fix_megno", False))
        self.fix_megno2 = bool(hparams.get("fix_megno2", False))
        self.include_mmr = bool(hparams.get("include_mmr", True))
        self.include_nan = bool(hparams.get("include_nan", True))
        self.include_eplusminus = bool(hparams.get("include_eplusminus", True))
        self.combined_mass_feature = bool(
            hparams.get("combined_mass_feature", False)
        )
        self.zero_theta_list = hparams.get("zero_theta", 0)
        self.max_pred = float(hparams.get("max_pred", 12.0))
        self.lowest = 0.1 if hparams.get("lower_std", False) else 0.5

        self._regress_is_symbolic = False

    # ------------------------------------------------------------------- io
    def set_regress_nn(self, module: nn.Module, *, is_symbolic: bool) -> None:
        """Replace the f2 head. ``is_symbolic=True`` switches the clamp
        from soft (NN) to hard (matches VarModel.predict_instability)."""
        self.regress_nn = module
        self._regress_is_symbolic = bool(is_symbolic)

    # ----------------------------------------------------- input prep
    def _zero_indices(self, x: torch.Tensor, indices) -> torch.Tensor:
        # Match VarModel.zero_megno / zero_mmr / ... which zero out *only*
        # those feature indices, preserving the rest.
        mask = torch.zeros_like(x)
        if isinstance(indices, int):
            indices = [indices]
        for ix in indices:
            mask[..., ix] = x[..., ix]
        return x - mask

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if self.fix_megno or self.fix_megno2:
            x = self._zero_indices(x, self.megno_location)
        if not self.include_mmr:
            x = self._zero_indices(x, self.mmr_location)
        if not self.include_nan:
            x = self._zero_indices(x, self.nan_location)
        if not self.include_eplusminus:
            x = self._zero_indices(x, self.eplusminus_location)
        if self.zero_theta_list and self.zero_theta_list != 0:
            # zero_theta is a list of 1..6 indices
            for ix in self.zero_theta_list:
                assert 1 <= ix <= 6
                x = self._zero_indices(x, self.theta_locations[ix - 1])
        if self.combined_mass_feature:
            m1_ix, m2_ix, m3_ix = 35, 36, 37
            combined = x[..., m1_ix] + x[..., m2_ix] + x[..., m3_ix]
            x = torch.cat([x, combined.unsqueeze(-1)], dim=-1)
        return x

    # ------------------------------------------------------- summary stats
    def _summary_stats(self, x: torch.Tensor) -> torch.Tensor:
        # Mirrors VarModel.compute_summary_stats(deterministic=True) with
        # no_summary_sample=False, f1_variant != 'mean_cov'.
        feats = self.feature_nn(x)  # (B, T, latent) — assumes T on dim 1
        # When feature_nn is nn.Linear, it broadcasts over the time dim
        # by default (nn.Linear acts on the last axis).
        sample_mu = torch.mean(feats, dim=1)
        sample_var = torch.std(feats, dim=1) ** 2
        std_sample = torch.sqrt(torch.abs(sample_var) + EPSILON)
        if self.hparams_dict.get("no_std", False):
            return sample_mu
        if self.hparams_dict.get("no_mean", False):
            return std_sample
        return torch.cat((sample_mu, std_sample), dim=1)

    # ----------------------------------------------------- predict head
    def _predict_instability(self, summary_stats: torch.Tensor):
        testy = self.regress_nn(summary_stats)
        # Symbolic modules emit a single column for the mean; if we plugged
        # in a CompiledSymbolicModule directly here, the lite loader will
        # wrap it in a small adapter that emits (B, 2). That's handled
        # outside.
        if testy.shape[-1] != 2:
            raise RuntimeError(
                f"regress_nn must emit (B, 2); got {tuple(testy.shape)}"
            )
        if self._regress_is_symbolic:
            mu = _hard_clamp(testy[:, [0]], 4.0, self.max_pred)
            std = _hard_clamp(testy[:, [1]], self.lowest, 6.0)
        else:
            mu = _soft_clamp(testy[:, [0]], 4.0, self.max_pred)
            std = _soft_clamp(testy[:, [1]], self.lowest, 6.0)
        return mu, std

    # ------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        x = self._prepare_inputs(x)
        summary_stats = self._summary_stats(x)
        mu, std = self._predict_instability(summary_stats)
        out = torch.cat((mu, std), dim=1)
        if return_intermediates:
            return {
                "inputs": x,
                "summary_stats": summary_stats,
                "mean": mu,
                "std": std,
                "prediction": out,
            }
        return out
