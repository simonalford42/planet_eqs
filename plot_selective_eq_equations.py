"""Render the Pareto-frontier equations from each selective_eq run as LaTeX
into a single PDF (one page per p).
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import pickle
import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pysr.export_latex import sympy2latex

from interpret import paretoize


SELECTIVE_EQ_RUNS = OrderedDict([
    (4414,  0.10),
    (28125, 0.25),
    (16119, 0.50),
    (3834,  0.75),
])


def latex_for_eq(expr):
    s = sympy2latex(expr, prec=3)
    s = s.replace('m_{', r'\mu_{').replace('s_{', r'\sigma_{')
    return s


def pareto_rows(pysr_version):
    with open(f'sr_results/{pysr_version}.pkl', 'rb') as f:
        reg = pickle.load(f)
    eqs = reg.equations_
    df = eqs[0] if isinstance(eqs, list) else eqs
    complexities = df['complexity'].tolist()
    losses = df['loss'].tolist()
    pareto_c, _ = paretoize(complexities, losses, replace=False)
    rows = []
    for c in pareto_c:
        row = df[df['complexity'] == c].iloc[0]
        rows.append({
            'complexity': int(c),
            'loss': float(row['loss']),
            'latex': latex_for_eq(row['sympy_format']),
        })
    return rows


def render_page(pdf, pysr_version, p, rows, best_c=None):
    # one row per equation. ample vertical space so the math doesn't get cropped.
    n = len(rows)
    fig_height = max(4.0, 0.85 * n + 1.6)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    title = (f'selective_eq  pysr={pysr_version}  p={p:.2f}'
             f'   ({n} Pareto-frontier equations)')
    if best_c is not None:
        title += f'\nbest val complexity: c*={best_c}'
    ax.text(0.5, 0.97, title, ha='center', va='top', fontsize=12,
            fontfamily='serif', weight='bold')

    # Header row
    y0 = 0.90
    dy = 0.86 / max(n, 1)
    ax.text(0.04, y0, r'$c$', ha='left', va='center', fontsize=11, fontfamily='serif')
    ax.text(0.12, y0, 'loss', ha='left', va='center', fontsize=11, fontfamily='serif')
    ax.text(0.22, y0, 'equation', ha='left', va='center', fontsize=11, fontfamily='serif')
    ax.plot([0.03, 0.97], [y0 - 0.5 * dy, y0 - 0.5 * dy], color='k', lw=0.5)

    for i, row in enumerate(rows):
        y = y0 - (i + 1) * dy
        is_best = (best_c is not None and row['complexity'] == best_c)
        weight = 'bold' if is_best else 'normal'
        marker = r' $\bigstar$' if is_best else ''
        ax.text(0.04, y, f"${row['complexity']}$" + marker, ha='left', va='center',
                fontsize=11, fontfamily='serif', fontweight=weight)
        ax.text(0.12, y, f"{row['loss']:.3f}", ha='left', va='center',
                fontsize=11, fontfamily='serif', fontweight=weight)
        ax.text(0.22, y, f"$s(x) = {row['latex']}$",
                ha='left', va='center', fontsize=11, fontfamily='serif',
                fontweight=weight)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='figures/selective_eq_equations.pdf')
    parser.add_argument('--eval_pickle', default='pickles/eval_selective_eq.pkl')
    args = parser.parse_args()

    # Best complexities for star-marking the chosen one
    best_c_by_v = {}
    if os.path.exists(args.eval_pickle):
        e = pickle.load(open(args.eval_pickle, 'rb'))
        best_c_by_v = {int(v): info['best_complexity'] for v, info in e.items()}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with PdfPages(args.out) as pdf:
        for pysr_version, p in SELECTIVE_EQ_RUNS.items():
            rows = pareto_rows(pysr_version)
            print(f'pysr={pysr_version} p={p:.2f}: {len(rows)} Pareto equations '
                  f'(c={[r["complexity"] for r in rows]})')
            render_page(pdf, pysr_version, p, rows,
                        best_c=best_c_by_v.get(pysr_version))
    print('Saved to', args.out)


if __name__ == '__main__':
    main()
