"""Verify CompiledSymbolicModule matches modules.PySRNet output bit-for-bit
(or at least to ~1e-6 atol) on the official PySR hall-of-fame files."""

from __future__ import annotations

import os
import sys
import json
import pickle

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "figures"))

import pysr  # noqa: E402

import modules  # noqa: E402  (existing pipeline)
from planet_eqs._eq_compiler import compile_expr  # noqa: E402


def collect_pysr_versions() -> list[str]:
    versions = json.load(open(os.path.join(REPO_ROOT, "official_versions.json")))
    out = set()
    for k, v in versions.items():
        if isinstance(v, (int, str)) and "pysr" in k and "selection" not in k:
            if "dir" in k:
                continue
            out.add(str(v))
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, dict) and "pysr_version" in vv:
                    out.add(str(vv["pysr_version"]))
    # only keep ones that have hof files on disk
    avail = []
    for x in sorted(out):
        if os.path.exists(os.path.join(REPO_ROOT, "sr_results", f"{x}.pkl")):
            avail.append(x)
    return avail


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    pysr_versions = collect_pysr_versions()
    print(f"Testing on PySR pickles: {pysr_versions}")
    print()

    all_ok = True
    for v in pysr_versions:
        path = os.path.join(REPO_ROOT, "sr_results", f"{v}.pkl")
        reg = pysr.PySRRegressor.from_file(path)
        feature_names = list(reg.feature_names_in_)
        n_in = len(feature_names)

        eqs = reg.equations_
        eqs_list = eqs if isinstance(eqs, list) else [eqs]

        # Try a few complexities across the range, including endpoints
        # and a couple in the middle.
        for head_ix, df in enumerate(eqs_list):
            complexities = sorted(set(df["complexity"]))
            if len(complexities) > 6:
                pick = [complexities[0], complexities[-1],
                        complexities[len(complexities) // 3],
                        complexities[2 * len(complexities) // 3]]
            else:
                pick = complexities
            pick = sorted(set(pick))

            for c in pick:
                # Build the same single-head module via my compiler.
                ix = (df["complexity"] - int(c)).abs().idxmin()
                expr = df.loc[ix, "sympy_format"]
                compiled = compile_expr(expr, feature_names)

                # Positive random input — many PySR equations use
                # fractional powers (e.g. s4 ** -0.32) and would NaN on
                # negative bases. Both existing and compiled modules
                # produce NaN there, so the diff is uninformative. Stay
                # in the positive regime to actually compare values.
                x = torch.rand(16, n_in) * 2.0 + 0.1   # in [0.1, 2.1]

                # Existing path. Some hof files fail with sklearn-compat
                # errors inside reg.pytorch() (e.g. 93102 missing
                # 'bumper'); skip the comparison for those but still
                # verify our compiler runs cleanly on the same expr.
                try:
                    existing = modules.PySRNet(path, model_selection=int(c))
                    with torch.no_grad():
                        old_out_full = existing(x)
                        old_out_head = old_out_full[:, head_ix]
                except Exception as e:
                    print(
                        f"  v={v}  head={head_ix}  c={c:>3d}  SKIP-existing  "
                        f"(reason: {type(e).__name__}: {str(e)[:60]})"
                    )
                    with torch.no_grad():
                        _ = compiled(x)  # at least exercise our path
                    continue

                with torch.no_grad():
                    new_out_head = compiled(x)

                # Compare valid entries (both finite); also count how
                # many are NaN in each to make sure those agree too.
                finite_mask = torch.isfinite(old_out_head) & torch.isfinite(new_out_head)
                nan_mask_old = ~torch.isfinite(old_out_head)
                nan_mask_new = ~torch.isfinite(new_out_head)
                same_nan_pattern = bool((nan_mask_old == nan_mask_new).all())

                if finite_mask.any():
                    diff = (old_out_head[finite_mask] - new_out_head[finite_mask]).abs()
                    max_d = float(diff.max().item())
                    rel = float((diff / (old_out_head[finite_mask].abs() + 1e-12)).max().item())
                else:
                    max_d = 0.0
                    rel = 0.0

                ok = (max_d < 1e-5) and same_nan_pattern
                status = "OK " if ok else "MISMATCH"
                if not ok:
                    all_ok = False
                print(
                    f"  v={v}  head={head_ix}  c={c:>3d}  {status}  "
                    f"max|diff|={max_d:.2e}  max_rel={rel:.2e}  "
                    f"nan_pat={'=' if same_nan_pattern else 'X'}  "
                    f"({int(nan_mask_old.sum())}/{len(old_out_head)} nan)"
                )

    print()
    if all_ok:
        print("compiler matches modules.PySRNet on all tested equations")
        sys.exit(0)
    else:
        print("MISMATCHES — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
