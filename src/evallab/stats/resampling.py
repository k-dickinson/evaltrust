"""Paired resampling: bootstrap confidence intervals and permutation tests.

Both operate on the vector of *per-example differences* (score_B - score_A on the
same example). Pairing is what makes an eval comparison powerful: the same items
are scored by both models, so we can look at the difference example by example
instead of comparing two noisy averages.

Everything is seeded, so the auditor is itself reproducible.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    differences: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences.

    Resamples examples with replacement ``n_resamples`` times, recomputes the
    mean difference each time, and returns the empirical percentile interval.
    If the interval excludes 0, the two models are distinguishable at this level.
    """
    diffs = np.asarray(differences, dtype=float)
    if diffs.size == 0:
        raise ValueError("bootstrap_ci requires at least one difference")

    rng = np.random.default_rng(seed)
    n = diffs.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = diffs[idx].mean(axis=1)

    alpha = 1.0 - confidence
    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


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
