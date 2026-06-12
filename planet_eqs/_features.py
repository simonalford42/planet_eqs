"""Rebound short-integration and feature-extraction, copied from
``figures/spock/tseries_feature_functions.py`` and the StandardScaler from
``figures/spock/regression.py``. Only the inference path is included.

Runtime deps: ``rebound``, ``numpy``. No numba (the original ``@numba.njit``
on ``data_setup_kernel`` was already commented out), no scipy, no pandas.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import rebound


# ====================================================================== ssX
# StandardScaler statistics, lifted verbatim from
# figures/spock/regression.py:NonSwagFeatureRegressor.__init__. These are
# the empirical mean/scale of the 41 input features computed on the
# training set; they don't change between checkpoints.
SSX_MEAN = np.array([
     4.95458585e+03,  5.67411891e-02,  3.83176945e-02,  2.97223474e+00,
     6.29733979e-02,  3.50074471e-02,  6.72845676e-01,  9.92794768e+00,
     9.99628430e-01,  5.39591547e-02,  2.92795061e-02,  2.12480714e-03,
    -1.01500319e-02,  1.82667162e-02,  1.00813201e-02,  5.74404197e-03,
     6.86570242e-03,  1.25316320e+00,  4.76946516e-02,  2.71326280e-02,
     7.02054326e-03,  9.83378673e-03, -5.70616748e-03,  5.50782881e-03,
    -8.44213953e-04,  2.05958338e-03,  1.57866569e+00,  4.31476211e-02,
     2.73316392e-02,  1.05505555e-02,  1.03922250e-02,  7.36865006e-03,
    -6.00523246e-04,  6.53016990e-03, -1.72038113e-03,  1.24807860e-05,
     1.60314173e-05,  1.21732696e-05,  5.67292645e-03,  1.92488263e-01,
     5.08607199e-03,
])
SSX_SCALE = np.array([
    2.88976974e+03, 6.10019661e-02, 4.03849732e-02, 4.81638693e+01,
    6.72583662e-02, 4.17939679e-02, 8.15995339e+00, 2.26871589e+01,
    4.73612029e-03, 7.09223721e-02, 3.06455099e-02, 7.10726478e-01,
    7.03392022e-01, 7.07873597e-01, 7.06030923e-01, 7.04728204e-01,
    7.09420909e-01, 1.90740659e-01, 4.75502285e-02, 2.77188320e-02,
    7.08891412e-01, 7.05214134e-01, 7.09786887e-01, 7.04371833e-01,
    7.04371110e-01, 7.09828420e-01, 3.33589977e-01, 5.20857790e-02,
    2.84763136e-02, 7.02210626e-01, 7.11815232e-01, 7.10512240e-01,
    7.03646004e-01, 7.08017286e-01, 7.06162814e-01, 2.12569430e-05,
    2.35019125e-05, 2.04211110e-05, 7.51048890e-02, 3.94254400e-01,
    7.11351099e-02,
])


def standard_scale(X: np.ndarray) -> np.ndarray:
    """Apply the StandardScaler used during training. ``X`` has shape
    ``(..., 41)``; returns the same shape."""
    return (X - SSX_MEAN) / SSX_SCALE


# ============================================================ MMR utilities
# (Verbatim from celmech via figures/spock/tseries_feature_functions.py.)

def _farey_sequence(n: int):
    a, b, c, d = 0, 1, 1, n
    seq = [(a, b)]
    while c <= n:
        k = int((n + b) / d)
        a, b, c, d = c, d, (k * c - a), (k * d - b)
        seq.append((a, b))
    return seq


def _resonant_period_ratios(min_per_ratio: float, max_per_ratio: float, order: int):
    if min_per_ratio < 0.0:
        raise AttributeError(
            f"min_per_ratio of {min_per_ratio} can't be < 0"
        )
    if max_per_ratio >= 1.0:
        raise AttributeError(
            f"max_per_ratio of {max_per_ratio} can't be >= 1"
        )
    minJ = int(np.floor(1.0 / (1.0 - min_per_ratio)))
    maxJ = int(np.ceil(1.0 / (1.0 - max_per_ratio)))
    res = [(minJ - 1, minJ)]
    for j in range(minJ, maxJ):
        res = res + [
            (x[1] * j - x[1] + x[0], x[1] * j + x[0])
            for x in _farey_sequence(order)[1:]
        ]
    res = np.array(res)
    msk = np.array(
        [min_per_ratio < x[0] / float(x[1]) < max_per_ratio for x in res]
    )
    return res[msk]


# ==================================================== rebound feature pipe

def _get_pairs(sim: rebound.Simulation, indices):
    ps = sim.particles
    sortedindices = sorted(indices, key=lambda i: ps[i].a)
    em_inner = (ps[sortedindices[1]].a - ps[sortedindices[0]].a) / ps[sortedindices[0]].a
    em_outer = (ps[sortedindices[2]].a - ps[sortedindices[1]].a) / ps[sortedindices[1]].a
    if em_inner < em_outer:
        return [["near", sortedindices[0], sortedindices[1]],
                ["far",  sortedindices[1], sortedindices[2]]]
    return [["near", sortedindices[1], sortedindices[2]],
            ["far",  sortedindices[0], sortedindices[1]]]


def _find_strongest_MMR(sim, i1, i2):
    ps = sim.particles
    n1 = ps[i1].n
    n2 = ps[i2].n
    m1 = ps[i1].m / ps[0].m
    m2 = ps[i2].m / ps[0].m
    Pratio = n2 / n1
    delta = 0.03
    if Pratio < 0 or Pratio > 1:
        return np.nan, np.nan, np.nan
    minp = max(Pratio - delta, 0.0)
    maxp = min(Pratio + delta, 0.99)
    res = _resonant_period_ratios(minp, maxp, order=2)
    EM = np.sqrt(
        (ps[i1].e * np.cos(ps[i1].pomega) - ps[i2].e * np.cos(ps[i2].pomega)) ** 2
        + (ps[i1].e * np.sin(ps[i1].pomega) - ps[i2].e * np.sin(ps[i2].pomega)) ** 2
    )
    EMcross = (ps[i2].a - ps[i1].a) / ps[i1].a
    j, k, maxstrength = np.nan, np.nan, 0.0
    for a, b in res:
        nres = (b * n2 - a * n1) / n1
        if nres == 0:
            s = np.inf
        else:
            s = np.abs(np.sqrt(m1 + m2) * (EM / EMcross) ** ((b - a) / 2.0) / nres)
        if s > maxstrength:
            j = b
            k = b - a
            maxstrength = s
    if maxstrength == 0:
        maxstrength = np.nan
    return j, k, maxstrength


def _populate_extended_trio(sim, trio, pairs, tseries, i, a10):
    Ns = 3
    ps = sim.particles
    for q, (_label, i1, i2) in enumerate(pairs):
        m1 = ps[i1].m
        m2 = ps[i2].m
        e1x = ps[i1].e * np.cos(ps[i1].pomega)
        e1y = ps[i1].e * np.sin(ps[i1].pomega)
        e2x = ps[i2].e * np.cos(ps[i2].pomega)
        e2y = ps[i2].e * np.sin(ps[i2].pomega)
        tseries[i, Ns * q + 1] = np.sqrt((e2x - e1x) ** 2 + (e2y - e1y) ** 2)
        tseries[i, Ns * q + 2] = (
            np.sqrt((m1 * e1x + m2 * e2x) ** 2 + (m1 * e1y + m2 * e2y) ** 2)
            / (m1 + m2)
        )
        _j, _k, tseries[i, Ns * q + 3] = _find_strongest_MMR(sim, i1, i2)

    tseries[i, 7] = sim.calculate_megno()

    orbits = sim.calculate_orbits()
    for j_idx, k in enumerate(trio):
        o = orbits[k - 1]
        tseries[i,  8 + 6 * j_idx] = o.a / a10
        tseries[i,  9 + 6 * j_idx] = o.e
        tseries[i, 10 + 6 * j_idx] = o.inc
        tseries[i, 11 + 6 * j_idx] = o.Omega
        tseries[i, 12 + 6 * j_idx] = o.pomega
        tseries[i, 13 + 6 * j_idx] = o.theta


def get_extended_tseries(sim: rebound.Simulation, indices=None):
    """Integrate ``sim`` for ~10000 inner-planet orbits and return the
    1000-sample feature time series for the trio of adjacent planets.

    Returns ``(triotseries, stable)``. If a Collision is hit during
    integration, ``stable=False`` and the partial tseries is returned.

    Mirrors ``figures/spock/tseries_feature_functions.get_extended_tseries``.
    """
    Norbits = 10_000
    Nout = 1000
    if indices is not None:
        if len(indices) != 3:
            raise ValueError("indices must be a list of 3 particle indices")
        trios = [list(indices)]
    else:
        trios = [[i, i + 1, i + 2] for i in range(1, sim.N_real - 2)]

    a10s = [sim.particles[trio[0]].a for trio in trios]
    minP = float(np.min([p.P for p in sim.particles[1: sim.N_real]]))
    times = np.linspace(0, Norbits * np.abs(minP), Nout)

    triopairs = [_get_pairs(sim, trio) for trio in trios]
    triotseries = [np.full((Nout, 26), np.nan) for _ in trios]

    for i, t in enumerate(times):
        try:
            sim.integrate(t, exact_finish_time=0)
        except rebound.Collision:
            return triotseries, False
        for tr, trio in enumerate(trios):
            triotseries[tr][i, 0] = sim.t / minP
            _populate_extended_trio(
                sim, trio, triopairs[tr], triotseries[tr], i, a10s[tr]
            )
    return triotseries, True


# ====================================================== data_setup_kernel

# Indices on the raw 26-column tseries that hold angles, expanded to
# (cos, sin) pairs during feature construction. Verbatim from
# figures/spock/regression.py:data_setup_kernel.
_ANGLE_AXES_IN_OLD = [11, 12, 13, 17, 18, 19, 23, 24, 25]


def data_setup_kernel(mass_array: np.ndarray, cur_tseries: np.ndarray) -> np.ndarray:
    """Build the 41-feature input expected by the NN.

    Args:
        mass_array: shape (3,), the m_i / m_star ratios for the trio.
        cur_tseries: shape (1, 100, 26), the time-series block (already
            subsampled — caller takes every 10th row out of 1000).
    Returns:
        shape (1, 100, 41).
    """
    mass = np.tile(mass_array[None], (100, 1))[None]   # (1, 100, 3)
    X0 = np.concatenate((cur_tseries, mass), axis=2)   # (1, 100, 29)

    isnotfinite = lambda v: ~np.isfinite(v)
    X0 = np.concatenate((X0, isnotfinite(X0[:, :, [3]]).astype(float)), axis=2)
    X0 = np.concatenate((X0, isnotfinite(X0[:, :, [6]]).astype(float)), axis=2)
    X0 = np.concatenate((X0, isnotfinite(X0[:, :, [7]]).astype(float)), axis=2)
    X0[..., :] = np.nan_to_num(X0[..., :], posinf=0.0, neginf=0.0)

    cols = []
    for j in range(X0.shape[-1]):
        if j in _ANGLE_AXES_IN_OLD:
            cols.append(np.cos(X0[:, :, [j]]))
            cols.append(np.sin(X0[:, :, [j]]))
        else:
            cols.append(X0[:, :, [j]])
    X = np.concatenate(cols, axis=2)
    if X.shape[-1] != 41:
        raise NotImplementedError(
            "expected 41 features after expansion; got "
            f"{X.shape[-1]}"
        )
    return X


# ========================================================= one-shot wrap

def features_from_sim(sim: rebound.Simulation, *, indices=None):
    """Run the full feature pipeline on a rebound sim.

    Returns a numpy array of shape ``(1, 100, 41)``, ready to be passed
    to the StandardScaler and then the NN. Returns ``None`` if the sim
    aborted (Collision or rebound.Escape during integration).
    """
    try:
        triotseries, stable = get_extended_tseries(sim, indices=indices)
    except rebound.Escape:
        return None
    if not stable:
        return None

    # Pick the first trio (the figure code only uses one). Subsample
    # 1000 → 100 rows by taking every 10th (matches NonSwagFeatureRegressor).
    tseries = np.array(triotseries)
    cur = tseries[None, 0, ::10]   # (1, 100, 26)

    if indices is None:
        trio = [1, 2, 3] if sim.N_real >= 4 else []
    else:
        trio = list(indices)
    mass_array = np.array(
        [sim.particles[j].m / sim.particles[0].m for j in trio]
    )

    return data_setup_kernel(mass_array, cur)
