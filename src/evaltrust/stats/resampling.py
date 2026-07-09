"""Paired resampling: bootstrap confidence intervals and permutation tests.

Both operate on the per-example differences (score_B - score_A on the same
example). Everything is seeded, so the auditor is reproducible.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _sp


def bootstrap_ci(
    differences: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
    method: str = "percentile",
) -> tuple[float, float]:
    """Bootstrap CI for the mean of paired differences; excludes 0 -> distinguishable.

    ``method`` is ``"percentile"`` or ``"bca"`` (bias-corrected, more accurate on
    skewed data; falls back to percentile on degenerate samples).
    """
    if method not in ("percentile", "bca"):
        raise ValueError(
            f"method must be 'percentile' or 'bca', got {method!r}"
        )

    diffs = np.asarray(differences, dtype=float)
    if diffs.size == 0:
        raise ValueError("bootstrap_ci requires at least one difference")

    rng = np.random.default_rng(seed)
    n = diffs.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = diffs[idx].mean(axis=1)

    alpha = 1.0 - confidence
    lo_q, hi_q = alpha / 2, 1.0 - alpha / 2

    if method == "bca":
        adjusted = _bca_quantiles(diffs, means, lo_q, hi_q)
        if adjusted is not None:
            lo_q, hi_q = adjusted
        # else: BCa is undefined for this sample (see the docstring); fall
        # through to the percentile quantiles.

    lo = float(np.percentile(means, 100 * lo_q))
    hi = float(np.percentile(means, 100 * hi_q))
    return lo, hi


def _bca_quantiles(
    data: np.ndarray,
    boot_means: np.ndarray,
    lo_q: float,
    hi_q: float,
) -> tuple[float, float] | None:
    """BCa-adjusted lower/upper quantiles for the mean, or ``None`` when undefined.

    Mirrors ``scipy.stats.bootstrap(method="BCa")``.
    """
    n = data.size
    if n < 2:
        return None

    theta_hat = float(data.mean())

    # Bias-correction z0. When every resample mean is on one side of the observed
    # mean the fraction is 0 or 1, making z0 +/-inf (undefined).
    below = float(np.mean(boot_means < theta_hat))
    if not 0.0 < below < 1.0:
        return None
    z0 = float(_sp.norm.ppf(below))

    # Acceleration a: jackknife skewness of the mean.
    jackknife = (data.sum() - data) / (n - 1)
    centered = jackknife.mean() - jackknife
    denom = float(np.sum(centered ** 2))
    if denom == 0.0:
        return None
    accel = float(np.sum(centered ** 3)) / (6.0 * denom ** 1.5)
    if not np.isfinite(accel):
        return None

    def adjust(q: float) -> float:
        z = float(_sp.norm.ppf(q))
        shifted = z0 + z
        return float(_sp.norm.cdf(z0 + shifted / (1.0 - accel * shifted)))

    lo_adj, hi_adj = adjust(lo_q), adjust(hi_q)
    if not (np.isfinite(lo_adj) and np.isfinite(hi_adj)):
        return None
    return lo_adj, hi_adj


def permutation_test(
    differences: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided paired permutation test that the mean difference is zero.

    Compares the observed |mean| against its distribution under random sign flips.
    Monte-Carlo p-value with the (count + 1) / (N + 1) correction, so never 0.
    """
    diffs = np.asarray(differences, dtype=float)
    if diffs.size == 0:
        raise ValueError("permutation_test requires at least one difference")

    observed = abs(float(diffs.mean()))
    rng = np.random.default_rng(seed)
    n = diffs.size
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, n))
    permuted = np.abs((signs * diffs).mean(axis=1))

    count = int(np.count_nonzero(permuted >= observed))
    return (count + 1) / (n_resamples + 1)
