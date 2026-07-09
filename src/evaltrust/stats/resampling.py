"""Paired resampling: bootstrap confidence intervals and permutation tests.

Both operate on the vector of *per-example differences* (score_B - score_A on the
same example). Pairing is what makes an eval comparison powerful: the same items
are scored by both models, so we can look at the difference example by example
instead of comparing two noisy averages.

Everything is seeded, so the auditor is itself reproducible.
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
    """Bootstrap CI for the mean of paired differences.

    Resamples examples with replacement ``n_resamples`` times, recomputes the
    mean difference each time, and reads a confidence interval off the resulting
    bootstrap distribution. If the interval excludes 0, the two models are
    distinguishable at this level.

    ``method`` selects how the interval is read from the bootstrap distribution:

    - ``"percentile"`` (default): the empirical percentile interval. Simple and
      unchanged; it is the historical behaviour.
    - ``"bca"``: the bias-corrected and accelerated interval. It shifts the
      percentiles to correct for median bias (``z0``, from the fraction of
      bootstrap means below the observed mean) and for skew (the acceleration
      ``a``, from the jackknife skewness of the mean). BCa is second-order
      accurate and noticeably more faithful than the percentile interval on
      skewed data; on symmetric data the two nearly coincide.

    Both methods share the same seeded resampling, so the auditor stays
    reproducible. The percentile path is byte-identical to before — passing
    ``method="bca"`` is purely additive.

    BCa is undefined for a few degenerate samples; in each case it falls back to
    the percentile interval (never a silent ``NaN``):

    - a single observation (``n == 1``): the jackknife acceleration needs at
      least two points;
    - zero variance / all-identical differences: the jackknife denominator is 0
      and the bootstrap distribution is a point mass;
    - a bootstrap distribution entirely on one side of the observed mean, so the
      bias-correction ``z0`` would be ``+/-inf``.
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
    """BCa-adjusted lower/upper quantiles (in ``[0, 1]``) for the mean.

    Returns ``None`` when the bias-correction or acceleration is undefined, so
    the caller can fall back to the percentile interval. Mirrors the bias
    correction and acceleration of ``scipy.stats.bootstrap(method="BCa")``.
    """
    n = data.size
    if n < 2:
        return None  # jackknife acceleration needs at least two observations

    theta_hat = float(data.mean())

    # Bias-correction z0: how far the bootstrap median sits from the observed
    # mean, in normal-quantile units. If every resample mean is on one side of
    # the observed mean the fraction is 0 or 1 and z0 is +/-inf -> undefined.
    below = float(np.mean(boot_means < theta_hat))
    if not 0.0 < below < 1.0:
        return None
    z0 = float(_sp.norm.ppf(below))

    # Acceleration a: jackknife skewness of the statistic. For the mean the
    # leave-one-out values are (sum - x_i) / (n - 1).
    jackknife = (data.sum() - data) / (n - 1)
    centered = jackknife.mean() - jackknife
    denom = float(np.sum(centered ** 2))
    if denom == 0.0:
        return None  # zero variance -> acceleration undefined
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

    Under the null hypothesis the two models are exchangeable on each example,
    so the sign of every difference could equally have been flipped. We compare
    the observed |mean| against the distribution of |mean| under random sign
    flips. Assumption-light and exact in the limit — no normality assumed.

    Returns a Monte-Carlo p-value using the standard (count + 1) / (N + 1)
    correction, which keeps the test valid (never reports p == 0).
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
