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


def bootstrap_statistic_ci(
    values: np.ndarray,
    statistic,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for an arbitrary statistic of a paired sample.

    Resamples the sample's indices with replacement in blocks bounded by the
    module's memory cap and applies ``statistic`` to each block. ``statistic``
    must accept a 2-D array of shape ``(rows, n)`` — one bootstrap resample per
    row — and return a 1-D array with the statistic for each row (i.e. it reduces
    the last axis), the same vectorized contract ``scipy.stats.bootstrap`` uses.
    This keeps the CI fast enough to run inside every audit.

    Seeded and deterministic. Reused for Cohen's *d* on the paired differences
    and for the paired risk difference (the mean) on pass/fail data.

    Provided ``statistic`` itself never returns ``NaN`` — as the mean and
    Cohen's *d* used here do not — the returned interval is never a silent
    ``NaN``. When the statistic is merely non-finite on some resamples (Cohen's
    *d* on a zero-variance resample is ``+/-inf``), a plain linear percentile
    would interpolate ``inf - inf`` to ``NaN``, so the interval is read with a
    non-interpolating percentile there instead, returning actual resample
    estimates (finite or ``+/-inf``).
    """
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        raise ValueError("bootstrap_statistic_ci requires at least one value")

    rng = np.random.default_rng(seed)
    n = vals.size
    estimates = np.empty(n_resamples, dtype=float)
    rows = _chunk_rows(n, n_resamples)
    pos = 0
    while pos < n_resamples:
        block = min(rows, n_resamples - pos)
        idx = rng.integers(0, n, size=(block, n))
        estimates[pos:pos + block] = np.asarray(
            statistic(vals[idx]), dtype=float,
        )
        pos += block

    alpha = 1.0 - confidence
    lo_pct, hi_pct = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    if np.isfinite(estimates).all():
        lo = float(np.percentile(estimates, lo_pct))
        hi = float(np.percentile(estimates, hi_pct))
    else:
        # Avoid inf - inf interpolation: pick actual order statistics instead.
        lo = float(np.percentile(estimates, lo_pct, method="lower"))
        hi = float(np.percentile(estimates, hi_pct, method="higher"))
    return lo, hi

def bootstrap_ci_clustered(
    cluster_diffs: list[np.ndarray],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI that resamples whole clusters, not individual rows.

    Each resample draws ``len(cluster_diffs)`` clusters with replacement and
    pools their differences, then takes the mean. This preserves within-cluster
    correlation so the interval reflects between-cluster variance.
    """
    if not cluster_diffs:
        raise ValueError("bootstrap_ci_clustered requires at least one cluster")

    rng = np.random.default_rng(seed)
    k = len(cluster_diffs)
    boot_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        chosen = rng.integers(0, k, size=k)
        pooled = np.concatenate([cluster_diffs[j] for j in chosen])
        boot_means[i] = float(pooled.mean())

    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2)))
    return lo, hi


def permutation_test_clustered(
    cluster_diffs: list[np.ndarray],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> float:
    """Cluster-aware permutation test: permute cluster labels, not rows.

    Each permutation randomly flips the sign of each cluster's differences
    (equivalent to reassigning which model is A and which is B for that
    cluster). The p-value is the fraction of permutations whose |mean| is
    at least as large as the observed |mean|, with the (count+1)/(N+1)
    correction so it is never exactly 0.
    """
    if not cluster_diffs:
        raise ValueError(
            "permutation_test_clustered requires at least one cluster"
        )

    all_diffs = np.concatenate(cluster_diffs)
    observed = abs(float(all_diffs.mean()))

    # Precompute per-cluster means so each permutation is O(k) not O(n).
    cluster_means = np.array(
        [float(c.mean()) for c in cluster_diffs], dtype=float
    )
    cluster_sizes = np.array([len(c) for c in cluster_diffs], dtype=float)
    total_n = float(cluster_sizes.sum())

    rng = np.random.default_rng(seed)
    k = len(cluster_diffs)
    count = 0
    for _ in range(n_resamples):
        signs = rng.choice(np.array([-1.0, 1.0]), size=k)
        perm_mean = abs(
            float(np.sum(signs * cluster_means * cluster_sizes)) / total_n
        )
        if perm_mean >= observed:
            count += 1
    return (count + 1) / (n_resamples + 1)

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
