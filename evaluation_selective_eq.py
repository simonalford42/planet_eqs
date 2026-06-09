"""Evaluate selective_eq runs.

A selective_eq run trains a score equation s(x) such that systems with the
lowest s(x) are the ones where a frozen baseline equation (here:
pysr_version=11003, complexity=26) predicts most accurately.

We evaluate each selective_eq run by:
  1. For each complexity c in the run, compute s_c(x) on the validation set.
  2. Take the lowest-p fraction of points (where p is the training fraction).
  3. Compute baseline-equation RMSE and LL2 (full_ll) on that subset.
  4. Choose the complexity that minimizes val LL2.
  5. Re-apply on test and random splits with the chosen complexity.

For a sanity check we also report:
  - Baseline metrics on the full split.
  - Baseline metrics on a randomly-chosen p-fraction of the split.
A working selector should beat the random p-fraction (and ideally beat the
full-split numbers too).

The selective_eq baseline is hardcoded to match sr.py: pysr_version=11003,
complexity=26.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse
import pickle
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pysr

import evaluation
import sr
from utils import save_pickle, load_pickle


BASELINE_PYSR_VERSION = 11003
BASELINE_COMPLEXITY = 26

# pysr_version -> training fraction used for selective_eq
SELECTIVE_EQ_RUNS = OrderedDict([
    (4414,  0.10),
    (28125, 0.25),
    (16119, 0.50),
    (3834,  0.75),
])


def _load_split(version, split):
    """Return (truths_full [N,2], summary_stats [N,D])."""
    args = SimpleNamespace(eval_type='nn', version=version, dataset=split)
    cache = evaluation._load_or_create_nn_cache_with_summaries(version, split)
    return np.asarray(cache['truths_full']), np.asarray(cache['summary_stats'])


def _get_equation(reg, complexity):
    eqs = reg.equations_
    if isinstance(eqs, list):
        eqs = eqs[0]
    ix = int(np.argmin(np.abs(eqs['complexity'].values - int(complexity))))
    return eqs.iloc[ix]


def _baseline_predict_fn():
    reg = pysr.PySRRegressor.from_file(f'sr_results/{BASELINE_PYSR_VERSION}.pkl')
    eq = _get_equation(reg, BASELINE_COMPLEXITY)
    print(f'baseline (pysr {BASELINE_PYSR_VERSION}, complexity={int(eq["complexity"])}, '
          f'train_loss={eq["loss"]:.4f}): {eq["equation"]}')
    return lambda X: eq['lambda_format'](X)


def _ll2_metrics(mu, truths_full):
    """Per-system mean LL2 (negative log-likelihood, sigma=1), averaged
    over the two ground truths per system. Matches sr.py's selective_eq
    training loss exactly. Numerically this is half of
    evaluation.lossfnc.total_loss2 (which sums over both GTs and divides
    by N_systems).
    """
    ll_1 = sr.ll2_per_sample(mu, truths_full[:, 0], sigma=1.0)
    ll_2 = sr.ll2_per_sample(mu, truths_full[:, 1], sigma=1.0)
    return (ll_1 + ll_2) / 2.0  # per-system mean NLL


def _metrics_subset(mu, truths_full, idx, clip=True):
    """Metrics for the subset of points indexed by idx."""
    if clip:
        mu_eval = np.clip(mu, 4, 9)
    else:
        mu_eval = mu
    mu_sub = mu_eval[idx]
    truths_sub = truths_full[idx]
    truths_avg = np.average(truths_sub, axis=1)

    unstable = truths_avg < 9
    if unstable.sum() == 0:
        rmse = float('nan')
    else:
        rmse = float(np.sqrt(np.mean((truths_avg[unstable] - mu_sub[unstable]) ** 2)))
    full_rmse = float(np.sqrt(np.mean((truths_avg - mu_sub) ** 2)))

    pred_stable = mu_sub >= 9
    true_stable = truths_avg >= 9
    acc = float(np.mean(pred_stable == true_stable))
    fpr = float(np.mean(~pred_stable[true_stable])) if true_stable.any() else float('nan')
    fnr = float(np.mean(pred_stable[~true_stable])) if (~true_stable).any() else float('nan')

    # LL2 per-system (uses unclipped mu so it matches sr's training loss)
    per_sys_ll2 = _ll2_metrics(mu[idx], truths_full[idx])
    full_ll = float(-per_sys_ll2.mean())  # match evaluation.calculate_ll convention

    return dict(
        n=int(len(idx)),
        rmse=rmse, full_rmse=full_rmse, acc=acc, fpr=fpr, fnr=fnr,
        full_ll=full_ll,
    )


def _selector_subset(scores, p):
    n = len(scores)
    k = max(1, int(np.floor(p * n)))
    order = np.argsort(scores)
    return order[:k]


def _random_subset(n, p, rng):
    k = max(1, int(np.floor(p * n)))
    return rng.choice(n, size=k, replace=False)


def evaluate_run(pysr_version, p, baseline_predict, splits, rng_seed=0):
    print(f'\n=== selective_eq pysr_version={pysr_version}  p={p:.2f} ===')
    reg = pysr.PySRRegressor.from_file(f'sr_results/{pysr_version}.pkl')
    eqs = reg.equations_
    if isinstance(eqs, list):
        eqs = eqs[0]
    complexities = [int(c) for c in eqs['complexity'].values]
    train_loss = {int(c): float(l) for c, l in zip(eqs['complexity'].values, eqs['loss'].values)}
    equations = {int(c): eqs.iloc[i]['equation'] for i, c in enumerate(eqs['complexity'].values)}

    # Precompute baseline predictions per split
    baseline_preds = {}
    for split, (truths, X) in splits.items():
        mu = baseline_predict(X)
        baseline_preds[split] = np.asarray(mu)

    results = {split: {} for split in splits}
    rng = np.random.RandomState(rng_seed)
    for c in complexities:
        score_eq = eqs.iloc[complexities.index(c)]['lambda_format']
        for split, (truths, X) in splits.items():
            scores = np.asarray(score_eq(X))
            sub_idx = _selector_subset(scores, p)
            results[split][c] = _metrics_subset(baseline_preds[split], truths, sub_idx)
        v = results['val'][c]
        t = results['test'][c]
        r = results['random'][c]
        print(f'  c={c:>2}  '
              f'val LL2={v["full_ll"]:>7.4f} rmse={v["rmse"]:>6.4f} acc={v["acc"]:>6.4f}   '
              f'test LL2={t["full_ll"]:>7.4f} rmse={t["rmse"]:>6.4f}   '
              f'rand LL2={r["full_ll"]:>7.4f} rmse={r["rmse"]:>6.4f}')

    # baseline reference: full and random-p subsets
    baseline_refs = {}
    for split, (truths, X) in splits.items():
        full_idx = np.arange(len(truths))
        rand_idx = _random_subset(len(truths), p, rng)
        baseline_refs[split] = dict(
            full=_metrics_subset(baseline_preds[split], truths, full_idx),
            random_subset=_metrics_subset(baseline_preds[split], truths, rand_idx),
        )

    # pick best complexity on val LL2 (= maximum full_ll, since full_ll is the negative loss)
    best_c = max(complexities, key=lambda c: results['val'][c]['full_ll'])
    print(f'  -> best complexity (max val full_ll): {best_c}')
    return dict(
        pysr_version=pysr_version,
        p=p,
        complexities=complexities,
        train_loss=train_loss,
        equations=equations,
        metrics=results,
        baseline=baseline_refs,
        best_complexity=best_c,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=24880)
    parser.add_argument('--out', default='pickles/eval_selective_eq.pkl')
    args = parser.parse_args()

    splits = {s: _load_split(args.version, s) for s in ('val', 'test', 'random')}
    for s, (t, X) in splits.items():
        print(f'{s}: truths={t.shape}, summary_stats={X.shape}')

    baseline_predict = _baseline_predict_fn()

    all_results = {}
    for pysr_version, p in SELECTIVE_EQ_RUNS.items():
        all_results[pysr_version] = evaluate_run(pysr_version, p, baseline_predict, splits)

    # Summary table
    print('\n=== Summary (chosen complexity per run) ===')
    print(f'{"pysr":>6} {"p":>5} {"c*":>4} {"val_LL2":>9} {"test_LL2":>9} {"rand_LL2":>9} '
          f'{"val_rmse":>9} {"test_rmse":>9} {"rand_rmse":>9}')
    for v, r in all_results.items():
        c = r['best_complexity']
        m = {s: r['metrics'][s][c] for s in ('val', 'test', 'random')}
        print(f'{v:>6} {r["p"]:>5.2f} {c:>4} '
              f'{m["val"]["full_ll"]:>9.4f} {m["test"]["full_ll"]:>9.4f} {m["random"]["full_ll"]:>9.4f} '
              f'{m["val"]["rmse"]:>9.4f} {m["test"]["rmse"]:>9.4f} {m["random"]["rmse"]:>9.4f}')

    print('\n=== Baseline reference (full split / random p-fraction) ===')
    print(f'{"pysr":>6} {"p":>5} '
          f'{"val_full_LL2":>12} {"val_rand_LL2":>12} {"val_full_rmse":>13} {"val_rand_rmse":>13}')
    for v, r in all_results.items():
        b = r['baseline']
        print(f'{v:>6} {r["p"]:>5.2f} '
              f'{b["val"]["full"]["full_ll"]:>12.4f} {b["val"]["random_subset"]["full_ll"]:>12.4f} '
              f'{b["val"]["full"]["rmse"]:>13.4f} {b["val"]["random_subset"]["rmse"]:>13.4f}')

    print('\n=== Improvement vs random p-fraction (val LL2) ===')
    for v, r in all_results.items():
        c = r['best_complexity']
        sel = r['metrics']['val'][c]['full_ll']
        rand = r['baseline']['val']['random_subset']['full_ll']
        full = r['baseline']['val']['full']['full_ll']
        delta_rand = sel - rand
        delta_full = sel - full
        print(f'  p={r["p"]:.2f} c={c}: '
              f'selector LL2={sel:>7.4f}  random LL2={rand:>7.4f}  full LL2={full:>7.4f}  '
              f'(vs rand: +{delta_rand:>6.4f}, vs full: +{delta_full:>6.4f})')

    save_pickle(all_results, args.out)
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
