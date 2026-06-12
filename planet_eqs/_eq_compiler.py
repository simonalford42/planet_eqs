"""Compile a sympy expression into a torch ``nn.Module``.

Used to materialize PySR-distilled equations at inference time *without*
needing pysr installed. The expected ops cover what PySR uses in the
shipped models (`+ - * / ^ sin`), plus a small extra set for robustness.

Usage:
    >>> import sympy
    >>> from planet_eqs._eq_compiler import compile_expr
    >>> x_syms = sympy.symbols("m0 m1 s0 s1")
    >>> expr = x_syms[0] + sympy.sin(x_syms[2]) * x_syms[1]
    >>> mod = compile_expr(expr, feature_names=["m0", "m1", "s0", "s1"])
    >>> import torch
    >>> mod(torch.randn(8, 4))   # -> shape (8,)
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


# Map from sympy func name (.func.__name__) to torch function. Add ops here
# if a future PySR run uses something new — kept conservative for now.
_UNARY = {
    "sin": torch.sin,
    "cos": torch.cos,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "Abs": torch.abs,
    "tan": torch.tan,
    "asin": torch.asin,
    "acos": torch.acos,
    "atan": torch.atan,
    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "tanh": torch.tanh,
}


def _eval(expr, x: torch.Tensor, name_to_idx: dict[str, int]):
    """Recursive evaluator. Returns either a Python scalar (for pure
    constants) or a torch.Tensor of shape ``(B,)``."""
    import sympy  # local; only this module needs sympy

    if expr.is_Symbol:
        name = expr.name
        if name not in name_to_idx:
            raise ValueError(
                f"Symbol {name!r} not in feature_names={list(name_to_idx)}"
            )
        return x[:, name_to_idx[name]]

    if expr.is_Number:
        # Rational / Float / Integer all coerce cleanly to float.
        return float(expr)

    if expr.is_Add:
        out = _eval(expr.args[0], x, name_to_idx)
        for child in expr.args[1:]:
            out = out + _eval(child, x, name_to_idx)
        return out

    if expr.is_Mul:
        out = _eval(expr.args[0], x, name_to_idx)
        for child in expr.args[1:]:
            out = out * _eval(child, x, name_to_idx)
        return out

    if expr.is_Pow:
        base = _eval(expr.args[0], x, name_to_idx)
        exp = _eval(expr.args[1], x, name_to_idx)
        return base ** exp

    # Unary functions like sin, cos, exp, log, ...
    fname = expr.func.__name__
    if fname in _UNARY:
        assert len(expr.args) == 1, f"{fname} expected 1 arg"
        return _UNARY[fname](_eval(expr.args[0], x, name_to_idx))

    raise NotImplementedError(
        f"sympy expression type not supported: {type(expr).__name__} "
        f"(func={fname}, expr={expr!r}). "
        "Add it to planet_eqs._eq_compiler._UNARY if it's a unary op."
    )


class CompiledSymbolicModule(nn.Module):
    """Wraps a sympy expression as an ``nn.Module``.

    The expression is *interpreted* on every forward pass (no JIT). For
    PySR-sized equations (< ~50 nodes) this is fast enough that it's not
    a bottleneck next to the rebound short integration.

    Input shape: ``(B, n_features)``. Output shape: ``(B,)``.
    """

    def __init__(self, expr, feature_names: Sequence[str]):
        super().__init__()
        self._expr_repr = str(expr)  # for debugging / equation()
        self._feature_names = list(feature_names)
        self._name_to_idx = {n: i for i, n in enumerate(feature_names)}
        # Keep the sympy expression itself attached for inspection. We do
        # not pickle/serialize this module via state_dict — it has no
        # learnable parameters, so saving the expression string is enough.
        self._expr = expr

    def extra_repr(self) -> str:
        return f"expr={self._expr_repr!r}"

    @property
    def equation_str(self) -> str:
        return self._expr_repr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(
                f"CompiledSymbolicModule expects input (B, n_features), got "
                f"shape {tuple(x.shape)}"
            )
        out = _eval(self._expr, x, self._name_to_idx)
        if not torch.is_tensor(out):
            # Pure-constant expression (no Symbols referenced). Broadcast
            # to batch shape.
            out = torch.full((x.shape[0],), float(out), device=x.device,
                             dtype=x.dtype)
        return out


def compile_expr(expr, feature_names: Sequence[str]) -> CompiledSymbolicModule:
    """Compile a sympy expression into an ``nn.Module``.

    Args:
        expr: a ``sympy.Expr``. Allowed nodes: Symbol, Number, Add, Mul,
            Pow, and any unary listed in ``_UNARY``.
        feature_names: the names of the input variables, in the order
            they appear in the input tensor's last dim.
    """
    return CompiledSymbolicModule(expr, feature_names)


def compile_expr_list(exprs, feature_names: Sequence[str]) -> nn.Module:
    """Compile a list of sympy expressions into a single module that
    emits a tensor of shape ``(B, len(exprs))``."""

    class _Stack(nn.Module):
        def __init__(self, mods):
            super().__init__()
            self.mods = nn.ModuleList(mods)

        def forward(self, x):
            outs = [m(x) for m in self.mods]
            return torch.stack(outs, dim=-1)

        @property
        def equation_strs(self):
            return [m.equation_str for m in self.mods]

    return _Stack([compile_expr(e, feature_names) for e in exprs])
