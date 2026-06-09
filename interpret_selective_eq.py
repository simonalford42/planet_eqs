"""Interpret the best equation from each selective_eq run.

A selective_eq equation is a *score* over the 20 NN summary stats (m0..m9
= time-averaged means of each latent feature, s0..s9 = time-std). Sorting by
score ascending and taking the lowest-p fraction is supposed to select the
points where the frozen baseline equation predicts accurately.

This script:
  - loads the chosen complexity per run from `pickles/eval_selective_eq.pkl`
  - prints each equation with the latent variables annotated by their
    dominant physical inputs (from the NN feature linear layer of nn 24880)
  - prints which latent variables show up across runs
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import collections
import pickle
import re

import numpy as np

import interpret


def latent_top_terms(feature_nn, i, k=2, threshold=0.05):
    w = feature_nn.input_linear[i]
    order = np.argsort(-np.abs(w))
    parts = []
    for j in order[:k]:
        if abs(w[j]) >= threshold:
            parts.append(f'{w[j]:+.2f}*{interpret.LABELS[j]}')
    return ' '.join(parts) if parts else '(no significant inputs)'


def main():
    r = pickle.load(open('pickles/eval_selective_eq.pkl', 'rb'))

    feature_nn = interpret.get_feature_nn(24880)
    n_latent = feature_nn.input_linear.shape[0]

    print('=== Latent feature dictionary (nn 24880) ===')
    print('m_i = time-average of latent i (over 100 time steps);  s_i = its temporal std.')
    print()
    for i in range(n_latent):
        print(f'  latent {i}: {latent_top_terms(feature_nn, i, k=2)}')
    print()

    print('=== Best equation per selective_eq run ===')
    counter = collections.Counter()
    for v, info in r.items():
        c = info['best_complexity']
        eq = info['equations'][c]
        vars_used = sorted(set(re.findall(r'[ms]\d+', eq)))
        for x in vars_used:
            counter[x] += 1
        print(f'pysr_version={v}  p={info["p"]:.2f}  c*={c}')
        print(f'  eq: {eq}')
        print(f'  vars: {vars_used}')
        print()

    print('=== Variable frequency across chosen equations ===')
    for x, n in counter.most_common():
        idx = int(x[1:])
        kind = 'mean' if x.startswith('m') else 'std '
        terms = latent_top_terms(feature_nn, idx, k=2)
        print(f'  {x} ({kind} of latent {idx} ~ {terms}): used in {n}/4 runs')


if __name__ == '__main__':
    main()
