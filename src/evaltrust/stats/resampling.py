"""Paired resampling: bootstrap confidence intervals and permutation tests.

Both operate on the per-example differences (score_B - score_A on the same
example). Everything is seeded, so the auditor is reproducible.

Resamples are drawn in memory-bounded blocks rather than one ``(n_resamples, n)``
matrix, so large evaluations don't exhaust memory. Because each block is drawn
from the same generator, the block sequence reproduces one big block exactly, so
results are byte-identical regardless of ``n`` or the block size.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _sp

# Cap on the (rows * n) working set of a single resample block. A block of this
# many cells at float64 is a few tens of MB; peak memory is bounded by it no
# matter how large the evaluation is.
_MAX_RESAMPLE_CELLS = 4_000_000


def _chunk_rows(n: int, n_resamples: int) -> int:
    """Resamples to draw per block so the ``rows * n`` working set stays bounded.

    Small ``n`` fits every resample in one block (identical to the unchunked
    path); large ``n`` shrinks the block, down to a single row when ``n`` alone
    exceeds the cap.
    """
    if n <= 0:
        return n_resamples
    return max(1, min(n_resamples, _MAX_RESAMPLE_CELLS // n))


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
    means = _bootstrap_means(diffs, n_resamples, rng)

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


def _bootstrap_means(diffs: np.ndarray, n_resamples: int, rng) -> np.ndarray:
    """Mean of each bootstrap resample, drawn in memory-bounded blocks.

    Equivalent to ``diffs[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)``
    but never materializes the full ``(n_resamples, n)`` matrix.
    """
    n = diffs.size
    means = np.empty(n_resamples, dtype=float)
    rows = _chunk_rows(n, n_resamples)
    pos = 0
    while pos < n_resamples:
        block = min(rows, n_resamples - pos)
        idx = rng.integers(0, n, size=(block, n))
        means[pos:pos + block] = diffs[idx].mean(axis=1)
        pos += block
    return means


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

    # Count resample means at least as extreme as observed, drawing the sign
    # flips in memory-bounded blocks instead of one (n_resamples, n) matrix.
    rows = _chunk_rows(n, n_resamples)
    count = 0
    pos = 0
    while pos < n_resamples:
        block = min(rows, n_resamples - pos)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(block, n))
        permuted = np.abs((signs * diffs).mean(axis=1))
        count += int(np.count_nonzero(permuted >= observed))
        pos += block
    return (count + 1) / (n_resamples + 1)
