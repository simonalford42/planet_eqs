## Equation-Driven Failure Mode Analysis

I tested concrete hypotheses derived from the selected distilled equation, complexity 26:

```text
T ~= ((((s4 + s2) * s6^0.356)^-0.305)
      + ((m7 - s8) / 1.200^m1)
      - sin(m2))
     * 0.084^s1
     + 3.659
```

The strongest prediction from the equation structure is that it should fail when learned short-term variability features are small. In particular, the term

```text
((s4 + s2) * s6^0.356)^-0.305
```

gets large when the learned std features are small. Since this term increases the predicted log instability time, the equation should overpredict stability for dynamically quiet-looking systems.

### Hypothesis 1: Worse when the low-variability long-time term is large

Prediction: systems with large values of `((s4 + s2) * s6^0.356)^-0.305` should have worse RMSE and classification accuracy.

Validated on top/bottom deciles:

| split | low term RMSE | high term RMSE | low acc | high acc |
|---|---:|---:|---:|---:|
| train | 1.097 | 1.899 | 0.960 | 0.702 |
| held-out test | 1.123 | 1.965 | 0.958 | 0.713 |

This is the largest gap found. On held-out test, high-term systems have FNR `0.374` versus `0.010` for low-term systems, meaning the equation often predicts stable for systems that are actually unstable.

### Hypothesis 2: Worse when `s6` is small

Feature interpretation suggests `s6` is roughly the std of a learned inclination combination, mostly involving `i2` and `i3`.

Prediction: low `s6` inflates the long-time term and should cause over-optimistic predictions.

Validated:

| split | low `s6` RMSE | high `s6` RMSE | low `s6` acc | high `s6` acc |
|---|---:|---:|---:|---:|
| train | 1.812 | 1.060 | 0.744 | 0.913 |
| held-out test | 1.811 | 1.104 | 0.772 | 0.903 |

This supports the interpretation that the equation relies on short-term dynamical variability: when the short integration looks too quiet, the equation predicts too long a survival time.

### Hypothesis 3: Worse when the mass/eccentricity offset term is high

The equation also contains:

```text
(m7 - s8) / 1.200^m1
```

`m7` is roughly a learned mass/eccentricity feature, and `s8` is roughly std of an `e1 - e3` feature.

Prediction: large positive values of this term boost predicted instability time and should worsen performance.

Validated:

| split | low term RMSE | high term RMSE | low acc | high acc |
|---|---:|---:|---:|---:|
| train | 1.248 | 1.818 | 0.927 | 0.748 |
| held-out test | 1.215 | 1.852 | 0.912 | 0.748 |

On held-out test, high-term FNR is `0.270` versus `0.019` for low-term systems.

### Physical Corollary

A direct physical split gives a weaker but consistent result: high-eccentricity systems are harder. On held-out test, top-decile max eccentricity RMSE is `1.569`, compared with `1.217` for bottom-decile max eccentricity.

## Summary

The clearest failure mode is:

> The distilled equation performs worst on systems that look dynamically quiet during the short integration, especially systems with low learned inclination variability `s6`.

Mechanism: the equation treats low short-term variability as evidence for long-term survival. When that assumption is wrong, it overpredicts instability time and misclassifies unstable systems as stable.

This gives a concrete, equation-derived prediction:

- Best performance: systems with high short-term learned variability, especially high `s6`.
- Worst performance: systems with low short-term learned variability but eventual instability.
