"""Tests for the resampling core: paired bootstrap CI and permutation test.

These functions are the heart of EvalTrust's statistical claims, so they are
validated against known-correct analytic values AND cross-checked against
scipy's own implementations.
"""

import numpy as np
import pytest
from scipy import stats as sp

from evaltrust.stats.resampling import (
    bootstrap_ci,
    bootstrap_statistic_ci,
    permutation_test,
)


# ---------------------------------------------------------------------------
# bootstrap_ci: percentile CI of the mean of paired differences
# ---------------------------------------------------------------------------

def test_bootstrap_ci_of_all_zero_differences_is_zero():
    diffs = np.zeros(50)
    lo, hi = bootstrap_ci(diffs, confidence=0.95, n_resamples=2000, seed=0)
    assert lo == 0.0
    assert hi == 0.0


def test_bootstrap_ci_of_constant_differences_is_that_constant():
    # Resampling a constant vector always yields the same mean, so the CI
    # collapses to the constant itself.
    diffs = np.full(30, 2.0)
    lo, hi = bootstrap_ci(diffs, confidence=0.95, n_resamples=2000, seed=0)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(2.0)


def test_bootstrap_ci_brackets_the_sample_mean():
    rng = np.random.default_rng(1)
    diffs = rng.normal(0.3, 1.0, size=400)
    lo, hi = bootstrap_ci(diffs, confidence=0.95, n_resamples=5000, seed=0)
    assert lo < diffs.mean() < hi


def test_bootstrap_ci_excludes_zero_for_clean_separation():
    rng = np.random.default_rng(2)
    diffs = rng.normal(1.0, 0.5, size=200)  # strongly positive
    lo, hi = bootstrap_ci(diffs, confidence=0.95, n_resamples=5000, seed=0)
    assert lo > 0.0


def test_bootstrap_ci_is_deterministic_for_fixed_seed():
    rng = np.random.default_rng(3)
    diffs = rng.normal(0.2, 1.0, size=100)
    a = bootstrap_ci(diffs, n_resamples=3000, seed=42)
    b = bootstrap_ci(diffs, n_resamples=3000, seed=42)
    assert a == b


def test_bootstrap_ci_matches_scipy_reference():
    rng = np.random.default_rng(4)
    diffs = rng.normal(0.4, 1.2, size=300)
    lo, hi = bootstrap_ci(diffs, confidence=0.95, n_resamples=9000, seed=7)
    ref = sp.bootstrap(
        (diffs,), np.mean, confidence_level=0.95, n_resamples=9000,
        method="percentile", random_state=7,
    )
    # Different RNG streams, so allow Monte Carlo slack.
    assert lo == pytest.approx(ref.confidence_interval.low, abs=0.05)
    assert hi == pytest.approx(ref.confidence_interval.high, abs=0.05)


# ---------------------------------------------------------------------------
# bootstrap_ci: BCa (bias-corrected accelerated) interval
# ---------------------------------------------------------------------------

def test_bca_matches_scipy_on_symmetric_data():
    diffs = np.random.default_rng(1).normal(0.4, 1.2, size=300)
    lo, hi = bootstrap_ci(diffs, confidence=0.95, n_resamples=9000, seed=7,
                          method="bca")
    ref = sp.bootstrap(
        (diffs,), np.mean, confidence_level=0.95, n_resamples=9000,
        method="BCa", random_state=7,
    )
    # Independent RNG streams -> Monte-Carlo slack. Measured endpoint gap on
    # this data was <= 0.005 at 9000 resamples; 0.02 leaves a 4x margin.
    assert lo == pytest.approx(ref.confidence_interval.low, abs=0.02)
    assert hi == pytest.approx(ref.confidence_interval.high, abs=0.02)


def test_bca_matches_scipy_on_right_skewed_data():
    diffs = np.random.default_rng(2).lognormal(0.0, 0.7, size=200) - 1.0
    lo, hi = bootstrap_ci(diffs, confidence=0.95, n_resamples=9000, seed=7,
                          method="bca")
    ref = sp.bootstrap(
        (diffs,), np.mean, confidence_level=0.95, n_resamples=9000,
        method="BCa", random_state=7,
    )
    assert lo == pytest.approx(ref.confidence_interval.low, abs=0.02)
    assert hi == pytest.approx(ref.confidence_interval.high, abs=0.02)


def test_bca_diverges_from_percentile_on_strong_skew_and_matches_scipy():
    # Strong right skew: BCa must actually shift the interval relative to the
    # percentile method (it is not a no-op), while still matching scipy's BCa.
    diffs = np.random.default_rng(42).lognormal(0.0, 1.0, size=60)
    bca = bootstrap_ci(diffs, confidence=0.95, n_resamples=20000, seed=5,
                       method="bca")
    perc = bootstrap_ci(diffs, confidence=0.95, n_resamples=20000, seed=5,
                        method="percentile")
    # Same seed -> same bootstrap draw, so any endpoint gap is the BCa
    # adjustment itself, not RNG noise. Measured shift was ~0.04 (lo) / ~0.09 (hi).
    assert abs(bca[0] - perc[0]) > 0.02 or abs(bca[1] - perc[1]) > 0.02
    ref = sp.bootstrap(
        (diffs,), np.mean, confidence_level=0.95, n_resamples=20000,
        method="BCa", random_state=5,
    )
    assert bca[0] == pytest.approx(ref.confidence_interval.low, abs=0.02)
    assert bca[1] == pytest.approx(ref.confidence_interval.high, abs=0.02)


def test_bca_is_deterministic_for_fixed_seed():
    diffs = np.random.default_rng(3).normal(0.2, 1.0, size=100)
    a = bootstrap_ci(diffs, n_resamples=4000, seed=42, method="bca")
    b = bootstrap_ci(diffs, n_resamples=4000, seed=42, method="bca")
    assert a == b


def test_percentile_is_the_default_and_bca_is_opt_in():
    diffs = np.random.default_rng(9).lognormal(0.0, 1.0, size=80)
    default = bootstrap_ci(diffs, n_resamples=8000, seed=1)
    percentile = bootstrap_ci(diffs, n_resamples=8000, seed=1, method="percentile")
    bca = bootstrap_ci(diffs, n_resamples=8000, seed=1, method="bca")
    assert default == percentile              # default did not change
    assert bca != percentile                  # BCa is a distinct interval


def test_bca_n1_falls_back_to_percentile_without_crashing():
    # A single observation: the jackknife acceleration is undefined, so BCa
    # falls back to the percentile interval (which for n=1 is the value itself).
    diffs = np.array([3.0])
    bca = bootstrap_ci(diffs, n_resamples=2000, seed=0, method="bca")
    perc = bootstrap_ci(diffs, n_resamples=2000, seed=0, method="percentile")
    assert bca == perc
    assert bca == (pytest.approx(3.0), pytest.approx(3.0))
    assert all(np.isfinite(bca))              # never a silent NaN


def test_bca_zero_variance_falls_back_without_nan():
    # All-identical (here all-zero) differences: the bootstrap distribution is a
    # point mass, so z0 -> +/-inf and the jackknife denominator is 0. BCa must
    # degrade to the percentile interval, not return NaN.
    diffs = np.zeros(50)
    bca = bootstrap_ci(diffs, n_resamples=2000, seed=0, method="bca")
    assert bca == (0.0, 0.0)
    assert all(np.isfinite(bca))


def test_bca_constant_nonzero_falls_back_without_nan():
    diffs = np.full(30, 2.0)
    bca = bootstrap_ci(diffs, n_resamples=2000, seed=0, method="bca")
    assert bca == (pytest.approx(2.0), pytest.approx(2.0))
    assert all(np.isfinite(bca))


def test_bootstrap_ci_rejects_unknown_method():
    with pytest.raises(ValueError):
        bootstrap_ci(np.zeros(10), method="bogus")


# ---------------------------------------------------------------------------
# bootstrap_statistic_ci: percentile CI for an arbitrary (vectorized) statistic
# ---------------------------------------------------------------------------

def _mean_axis(matrix):
    return matrix.mean(axis=-1)


def _cohens_d_reference(matrix, axis=-1):
    # Independent reference for Cohen's d over rows (does NOT call the library).
    m = np.asarray(matrix, dtype=float)
    return m.mean(axis=axis) / m.std(axis=axis, ddof=1)


def test_bootstrap_statistic_ci_mean_matches_scipy():
    diffs = np.random.default_rng(11).normal(0.4, 1.2, size=300)
    lo, hi = bootstrap_statistic_ci(diffs, _mean_axis, confidence=0.95,
                                    n_resamples=9000, seed=7)
    ref = sp.bootstrap(
        (diffs,), np.mean, confidence_level=0.95, n_resamples=9000,
        method="percentile", random_state=7,
    )
    assert lo == pytest.approx(ref.confidence_interval.low, abs=0.03)
    assert hi == pytest.approx(ref.confidence_interval.high, abs=0.03)


def test_bootstrap_statistic_ci_cohens_d_matches_scipy():
    diffs = np.random.default_rng(12).normal(0.5, 1.0, size=200)
    lo, hi = bootstrap_statistic_ci(diffs, _cohens_d_reference, confidence=0.95,
                                    n_resamples=9000, seed=7)
    ref = sp.bootstrap(
        (diffs,), _cohens_d_reference, confidence_level=0.95, n_resamples=9000,
        method="percentile", random_state=7, vectorized=True,
    )
    assert lo == pytest.approx(ref.confidence_interval.low, abs=0.03)
    assert hi == pytest.approx(ref.confidence_interval.high, abs=0.03)


def test_bootstrap_statistic_ci_of_mean_equals_bootstrap_ci():
    # For the mean, the general CI must reduce to bootstrap_ci exactly (same
    # seeded resampling scheme).
    diffs = np.random.default_rng(1).normal(0.3, 1.0, size=150)
    assert bootstrap_statistic_ci(diffs, _mean_axis, n_resamples=4000, seed=9) == \
        bootstrap_ci(diffs, n_resamples=4000, seed=9)


def test_bootstrap_statistic_ci_is_deterministic_for_fixed_seed():
    diffs = np.random.default_rng(3).normal(0.2, 1.0, size=100)
    a = bootstrap_statistic_ci(diffs, _mean_axis, n_resamples=3000, seed=42)
    b = bootstrap_statistic_ci(diffs, _mean_axis, n_resamples=3000, seed=42)
    assert a == b


def test_bootstrap_statistic_ci_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_statistic_ci(np.array([]), _mean_axis)


def test_bootstrap_statistic_ci_degenerate_never_returns_nan():
    from evaltrust.stats.effect import cohens_d_paired_along_rows
    # Zero variance -> Cohen's d is +/-inf on every resample; percentile
    # interpolation between infinities must NOT produce a silent NaN.
    lo, hi = bootstrap_statistic_ci(np.full(20, 2.0), cohens_d_paired_along_rows,
                                    n_resamples=1000, seed=0)
    assert not np.isnan(lo) and not np.isnan(hi)
    assert np.isinf(lo) and np.isinf(hi)
    # All-zero differences -> Cohen's d is exactly 0.
    assert bootstrap_statistic_ci(np.zeros(20), cohens_d_paired_along_rows,
                                  n_resamples=1000, seed=0) == (0.0, 0.0)
    # n == 1 -> a documented degenerate return, never a crash or NaN.
    lo1, hi1 = bootstrap_statistic_ci(np.array([0.7]), _mean_axis,
                                      n_resamples=500, seed=0)
    assert (lo1, hi1) == (pytest.approx(0.7), pytest.approx(0.7))


def test_bootstrap_statistic_ci_handles_both_signed_infinities():
    from evaltrust.stats.effect import cohens_d_paired_along_rows
    # Differences with both signs and small n: some resamples are all-identical
    # positive (Cohen's d = +inf) and some all-identical negative (-inf). The
    # non-interpolating guard must pick real order statistics, never inf - inf.
    diffs = np.array([1.0, 1.0, -1.0, -1.0])
    lo, hi = bootstrap_statistic_ci(diffs, cohens_d_paired_along_rows,
                                    n_resamples=2000, seed=0)
    assert not np.isnan(lo) and not np.isnan(hi)


# ---------------------------------------------------------------------------
# permutation_test: two-sided paired (sign-flip) test that mean diff == 0
# ---------------------------------------------------------------------------

def test_permutation_pvalue_is_one_for_all_zero_differences():
    diffs = np.zeros(20)
    p = permutation_test(diffs, n_resamples=2000, seed=0)
    assert p == pytest.approx(1.0)


def test_permutation_pvalue_is_small_for_strong_separation():
    diffs = np.ones(30)  # every example favours B, maximally
    p = permutation_test(diffs, n_resamples=5000, seed=0)
    assert p < 0.01


def test_permutation_pvalue_is_large_for_symmetric_noise():
    rng = np.random.default_rng(5)
    diffs = rng.normal(0.0, 1.0, size=200)  # no real effect
    p = permutation_test(diffs, n_resamples=5000, seed=0)
    assert p > 0.05


def test_permutation_pvalue_in_unit_interval():
    rng = np.random.default_rng(6)
    diffs = rng.normal(0.15, 1.0, size=120)
    p = permutation_test(diffs, n_resamples=4000, seed=0)
    assert 0.0 <= p <= 1.0


def test_permutation_is_deterministic_for_fixed_seed():
    rng = np.random.default_rng(7)
    diffs = rng.normal(0.2, 1.0, size=80)
    assert permutation_test(diffs, seed=1) == permutation_test(diffs, seed=1)


def test_permutation_matches_scipy_reference():
    rng = np.random.default_rng(8)
    diffs = rng.normal(0.25, 1.0, size=150)
    p = permutation_test(diffs, n_resamples=9000, seed=3)

    def mean_stat(x):
        return np.mean(x)

    ref = sp.permutation_test(
        (diffs,), mean_stat, permutation_type="samples",
        n_resamples=9000, random_state=3, alternative="two-sided",
    )
    assert p == pytest.approx(ref.pvalue, abs=0.03)


# ---------------------------------------------------------------------------
# Memory-bounded resampling (issue #79): large evaluations must not allocate an
# (n_resamples, n) matrix. Chunking the resamples must not change the result.
# ---------------------------------------------------------------------------

from evaltrust.stats import resampling as _rs


def test_chunk_rows_is_a_single_block_for_small_n():
    # Small n: the whole resample set fits the budget in one block, so the path
    # is identical (and equal cost) to the original unchunked implementation.
    assert _rs._chunk_rows(n=100, n_resamples=10_000) == 10_000


def test_chunk_rows_bounds_the_working_set_for_large_n():
    n = 1_000_000
    rows = _rs._chunk_rows(n=n, n_resamples=10_000)
    assert rows >= 1
    assert rows * n <= _rs._MAX_RESAMPLE_CELLS   # peak block is bounded by n
    assert rows < 10_000                          # and it actually chunked


def test_bootstrap_ci_is_bitwise_invariant_to_chunk_size(monkeypatch):
    # The whole correctness argument: drawing the resamples in blocks from the
    # same generator reproduces one big block exactly, so a tiny memory budget
    # (many blocks) gives byte-identical output to one giant block.
    diffs = np.random.default_rng(11).normal(0.3, 1.0, size=5000)
    monkeypatch.setattr(_rs, "_MAX_RESAMPLE_CELLS", 10**12)   # one block
    big = bootstrap_ci(diffs, n_resamples=3000, seed=7)
    big_bca = bootstrap_ci(diffs, n_resamples=3000, seed=7, method="bca")
    monkeypatch.setattr(_rs, "_MAX_RESAMPLE_CELLS", 4096)     # one row per block
    small = bootstrap_ci(diffs, n_resamples=3000, seed=7)
    small_bca = bootstrap_ci(diffs, n_resamples=3000, seed=7, method="bca")
    assert big == small          # exact equality, not approx
    assert big_bca == small_bca


def test_bootstrap_statistic_ci_bounds_each_statistic_block(monkeypatch):
    values = np.arange(8, dtype=float)
    cap = 16
    monkeypatch.setattr(_rs, "_MAX_RESAMPLE_CELLS", cap)

    def recording_mean(matrix):
        assert matrix.shape[0] * values.size <= cap
        return matrix.mean(axis=-1)

    bootstrap_statistic_ci(values, recording_mean, n_resamples=7, seed=0)


def test_bootstrap_statistic_ci_is_bitwise_invariant_to_chunk_size(monkeypatch):
    from evaltrust.stats.effect import cohens_d_paired_along_rows

    finite = np.random.default_rng(13).normal(0.3, 1.0, size=17)
    non_finite_capable = np.array([1.0, 1.0, -1.0, -1.0])
    monkeypatch.setattr(_rs, "_MAX_RESAMPLE_CELLS", 10**12)
    big_mean = bootstrap_statistic_ci(
        finite, _mean_axis, n_resamples=301, seed=7,
    )
    big_cohens_d = bootstrap_statistic_ci(
        non_finite_capable, cohens_d_paired_along_rows,
        n_resamples=401, seed=0,
    )
    monkeypatch.setattr(_rs, "_MAX_RESAMPLE_CELLS", 1)
    small_mean = bootstrap_statistic_ci(
        finite, _mean_axis, n_resamples=301, seed=7,
    )
    small_cohens_d = bootstrap_statistic_ci(
        non_finite_capable, cohens_d_paired_along_rows,
        n_resamples=401, seed=0,
    )
    assert big_mean == small_mean
    assert big_cohens_d == small_cohens_d


def test_permutation_is_bitwise_invariant_to_chunk_size(monkeypatch):
    diffs = np.random.default_rng(12).normal(0.2, 1.0, size=4000)
    monkeypatch.setattr(_rs, "_MAX_RESAMPLE_CELLS", 10**12)
    big = permutation_test(diffs, n_resamples=3000, seed=5)
    monkeypatch.setattr(_rs, "_MAX_RESAMPLE_CELLS", 4096)
    small = permutation_test(diffs, n_resamples=3000, seed=5)
    assert big == small


def test_large_n_bootstrap_completes_within_bounded_memory():
    # Unchunked this would allocate a 3000 x 200_000 matrix (~4.8 GB) and OOM.
    # The chunked path caps the block, so it must complete with a valid interval.
    n = 200_000
    diffs = np.random.default_rng(0).normal(0.1, 1.0, size=n)
    assert _rs._chunk_rows(n, 3000) * n <= _rs._MAX_RESAMPLE_CELLS
    lo, hi = bootstrap_ci(diffs, n_resamples=3000, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < diffs.mean() < hi


def test_large_n_permutation_completes_within_bounded_memory():
    n = 200_000
    diffs = np.random.default_rng(1).normal(0.0, 1.0, size=n)
    p = permutation_test(diffs, n_resamples=3000, seed=0)
    assert 0.0 <= p <= 1.0

# ---------------------------------------------------------------------------
# cluster-aware resampling
# ---------------------------------------------------------------------------
from evaltrust.stats.resampling import (
    bootstrap_ci_clustered,
    permutation_test_clustered,
)


def test_bootstrap_ci_clustered_all_zero_is_zero():
    clusters = [np.zeros(5) for _ in range(10)]
    lo, hi = bootstrap_ci_clustered(clusters, confidence=0.95, n_resamples=2000, seed=0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


def test_bootstrap_ci_clustered_constant_collapses():
    clusters = [np.full(4, 2.0) for _ in range(8)]
    lo, hi = bootstrap_ci_clustered(clusters, confidence=0.95, n_resamples=2000, seed=0)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(2.0)


def test_bootstrap_ci_clustered_brackets_mean():
    rng = np.random.default_rng(42)
    clusters = [rng.normal(0.3, 0.5, size=10) for _ in range(20)]
    all_diffs = np.concatenate(clusters)
    lo, hi = bootstrap_ci_clustered(clusters, confidence=0.95, n_resamples=5000, seed=0)
    assert lo < all_diffs.mean() < hi


def test_bootstrap_ci_clustered_excludes_zero_for_clear_signal():
    rng = np.random.default_rng(7)
    clusters = [rng.normal(1.0, 0.2, size=5) for _ in range(30)]
    lo, _hi = bootstrap_ci_clustered(clusters, confidence=0.95, n_resamples=5000, seed=0)
    assert lo > 0.0


def test_bootstrap_ci_clustered_wider_than_unclustered():
    """Cluster CI must be at least as wide as the row-level CI when there is
    meaningful between-cluster variance (different cluster means)."""
    rng = np.random.default_rng(99)
    # 10 clusters, each with a different mean — lots of between-cluster variance.
    clusters = [np.full(10, float(i)) for i in range(10)]
    all_diffs = np.concatenate(clusters)
    lo_c, hi_c = bootstrap_ci_clustered(clusters, confidence=0.95, n_resamples=5000, seed=0)
    lo_r, hi_r = bootstrap_ci(all_diffs, confidence=0.95, n_resamples=5000, seed=0)
    assert (hi_c - lo_c) >= (hi_r - lo_r)


def test_permutation_test_clustered_null_is_not_significant():
    rng = np.random.default_rng(0)
    clusters = [rng.normal(0.0, 1.0, size=5) for _ in range(20)]
    p = permutation_test_clustered(clusters, n_resamples=5000, seed=0)
    assert p > 0.05


def test_permutation_test_clustered_strong_signal_is_significant():
    clusters = [np.full(5, 1.0) for _ in range(20)]
    p = permutation_test_clustered(clusters, n_resamples=5000, seed=0)
    assert p < 0.01


def test_permutation_test_clustered_pvalue_in_range():
    rng = np.random.default_rng(3)
    clusters = [rng.normal(0.2, 0.5, size=8) for _ in range(15)]
    p = permutation_test_clustered(clusters, n_resamples=2000, seed=0)
    assert 0.0 < p <= 1.0


def test_permutation_test_clustered_never_zero():
    """The (count+1)/(N+1) correction guarantees p > 0."""
    clusters = [np.full(10, 100.0) for _ in range(5)]
    p = permutation_test_clustered(clusters, n_resamples=1000, seed=0)
    assert p > 0.0
