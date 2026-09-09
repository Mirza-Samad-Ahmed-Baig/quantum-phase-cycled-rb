"""Connected phase-cycle estimator with paired block bootstrap and a finite bound.

Input columns: ++,+-,-+,--,1+,1-,2+,2-. Each row is an independent
Clifford/trajectory block. Shot noise and within-block dependencies are allowed.
The target is (4/9) sin(beta)^2 Cov(sin(phi_1), sin(phi_2)) under the
dephasing/twirl model; it is not a quantum-memory classifier.
"""
from __future__ import annotations

import numpy as np


def block_components(survivals):
    p = np.asarray(survivals, float)
    if p.ndim != 2 or p.shape[1] != 8 or len(p) < 2:
        raise ValueError("need at least two blocks and exactly eight phase settings")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("survivals must be finite probabilities")
    return np.column_stack(((p[:, 0]-p[:, 1]-p[:, 2]+p[:, 3])/2,
                            p[:, 4]-p[:, 5], p[:, 6]-p[:, 7]))


def connected_estimate(components):
    """Unbiased connected moment using distinct blocks for the product of means."""
    c = np.asarray(components, float)
    n = len(c)
    if n < 2:
        raise ValueError("at least two independent blocks are required")
    joint, first, second = c.T
    product = (first.sum()*second.sum()-np.dot(first, second)) / (n*(n-1))
    return float(joint.mean()-product)


def phase_cycle_witness(survivals, n_boot=1000, alpha=0.05, seed=314159):
    """Return point estimate, empirical bootstrap CI and conservative Hoeffding CI.

    The latter covers the population connected moment at >=1-alpha for independent
    identically distributed bounded blocks; no asymptotic normality is invoked.
    It need not be centered on the U-statistic. It remains conservative at small n.
    Bootstrap coverage is empirical and must not be called a finite-sample guarantee.
    """
    if not 0 < alpha < 1 or n_boot < 20:
        raise ValueError("require 0 < alpha < 1 and n_boot >= 20")
    c = block_components(survivals)
    n = len(c)
    rng = np.random.default_rng(seed)
    boots = np.array([connected_estimate(c[rng.integers(0, n, n)]) for _ in range(n_boot)])
    empirical = np.quantile(boots, [alpha/2, 1-alpha/2])
    # Each component is in [-1,1]. Union bound over the three two-sided events.
    eps = np.sqrt(2*np.log(6/alpha)/n)
    lo, hi = np.maximum(-1, c.mean(0)-eps), np.minimum(1, c.mean(0)+eps)
    products = [a*b for a in (lo[1], hi[1]) for b in (lo[2], hi[2])]
    rigorous = [float(lo[0]-max(products)), float(hi[0]-min(products))]
    return {"connected": connected_estimate(c), "bootstrap_ci": empirical.tolist(),
            "bootstrap_stderr": float(boots.std(ddof=1)), "finite_sample_ci": rigorous,
            "bootstrap_detected": bool(empirical[0] > 0 or empirical[1] < 0),
            "finite_sample_detected": bool(rigorous[0] > 0 or rigorous[1] < 0),
            "n_blocks": n, "n_boot": n_boot, "alpha": alpha,
            "assumptions": "independent identically distributed bounded blocks; ideal twirl/dephasing interpretation"}
