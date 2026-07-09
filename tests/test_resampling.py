"""Tests for the resampling core: paired bootstrap CI and permutation test.

These functions are the heart of EvalTrust's statistical claims, so they are
validated against known-correct analytic values AND cross-checked against
scipy's own implementations.
"""

import numpy as np
import pytest
from scipy import stats as sp

from evaltrust.stats.resampling import bootstrap_ci, permutation_test


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
