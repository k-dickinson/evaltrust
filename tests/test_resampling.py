"""Tests for the resampling core: paired bootstrap CI and permutation test.

These functions are the heart of EvalLab's statistical claims, so they are
validated against known-correct analytic values AND cross-checked against
scipy's own implementations.
"""

import numpy as np
import pytest
from scipy import stats as sp

from evallab.stats.resampling import bootstrap_ci, permutation_test


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
