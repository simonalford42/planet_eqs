"""Evaluate the nn_std-trained pysr model (51222) across complexities and splits.

Picks best complexity on val LL, reports held-out test/random metrics.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse
import pickle
from types import SimpleNamespace

import pysr

import evaluation


def sweep_complexities(version, pysr_version, datasets=('val', 'test', 'random')):
    reg = pysr.PySRRegressor.from_file(f'sr_results/{pysr_version}.pkl')
    eqs = reg.equations_
    if isinstance(eqs, list):
        eqs = eqs[0]
    complexities = [int(c) for c in eqs['complexity'].values]
    losses = {int(c): float(l) for c, l in zip(eqs['complexity'].values, eqs['loss'].values)}

    all_results = {ds: {} for ds in datasets}
    for c in complexities:
        for ds in datasets:
            args = SimpleNamespace(
                eval_type='pysr',
                version=version,
                pysr_version=pysr_version,
                pysr_model_selection=str(c),
                dataset=ds,
            )
            metrics = evaluation.calculate_metrics(args, plot=False)
            all_results[ds][c] = metrics
            print(f'c={c:>3} {ds:>6}  rmse={metrics["rmse"]:.4f}  '
                  f'full_rmse={metrics["full_rmse"]:.4f}  '
                  f'acc={metrics["acc"]:.4f}  '
                  f'll={metrics["ll"]:.4f}  '
                  f'full_ll={metrics["full_ll"]:.4f}')

    return complexities, losses, all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=24880)
    parser.add_argument('--pysr_version', type=int, default=51222)
    parser.add_argument('--out', default='pickles/eval_nn_std_51222.pkl')
    args = parser.parse_args()

    complexities, losses, all_results = sweep_complexities(args.version, args.pysr_version)

    # choose best complexity on val LL (full_ll = censored)
    val = all_results['val']
    best_c = max(val.keys(), key=lambda c: val[c]['full_ll'])

    print('\n=== Summary ===')
    print(f'best complexity by val full_ll: {best_c}')
    for ds in all_results:
        m = all_results[ds][best_c]
        print(f'  {ds:>6}: rmse={m["rmse"]:.4f}  full_rmse={m["full_rmse"]:.4f}  '
              f'acc={m["acc"]:.4f}  ll={m["ll"]:.4f}  full_ll={m["full_ll"]:.4f}  '
              f'fpr={m["fpr"]:.4f}  fnr={m["fnr"]:.4f}')

    out = {
        'pysr_version': args.pysr_version,
        'nn_version': args.version,
        'complexities': complexities,
        'train_losses': losses,
        'metrics': all_results,
        'best_complexity_val_full_ll': best_c,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
