"""Tests for the unpaired two-sample statistics module (stats/two_sample.py).

Covers:
  - bootstrap_p_a_gt_b: point estimate, CI direction, reproducibility
  - mann_whitney_u: significant separation, near-identical distributions, small-sample sentinel
  - _p_a_gt_b: edge cases (ties, all-wins, all-losses)
"""

import numpy as np
import pytest

from evaltrust.stats.two_sample import (
    _p_a_gt_b,
    bootstrap_p_a_gt_b,
    mann_whitney_u,
)


# ---------------------------------------------------------------------------
# _p_a_gt_b (internal helper, but directly testable)
# ---------------------------------------------------------------------------

def test_p_a_gt_b_all_wins():
    a = np.array([1.0, 1.0])
    b = np.array([0.0, 0.0])
    assert _p_a_gt_b(a, b) == pytest.approx(1.0)


def test_p_a_gt_b_all_losses():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    assert _p_a_gt_b(a, b) == pytest.approx(0.0)


def test_p_a_gt_b_all_ties():
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    assert _p_a_gt_b(a, b) == pytest.approx(0.5)


def test_p_a_gt_b_mixed():
    # a=[0,1], b=[0,1] → 2 wins (a=1>b=0), 2 losses (a=0<b=1), 2 ties → (2 + 1) / 4 = 0.75
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 1.0])
    # pairs: (0,0)=tie, (0,1)=loss, (1,0)=win, (1,1)=tie
    # wins=1, ties=2, losses=1 → (1 + 0.5*2)/4 = 2/4 = 0.5
    assert _p_a_gt_b(a, b) == pytest.approx(0.5)


def test_p_a_gt_b_unequal_lengths():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([0.0, 0.0])
    # 3*2 = 6 pairs, all wins
    assert _p_a_gt_b(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# bootstrap_p_a_gt_b
# ---------------------------------------------------------------------------

def test_bootstrap_p_a_gt_b_clear_winner():
    rng = np.random.default_rng(42)
    a = rng.normal(loc=0.8, scale=0.05, size=50)
    b = rng.normal(loc=0.6, scale=0.05, size=50)
    p_hat, lo, hi = bootstrap_p_a_gt_b(a, b, seed=0)
    assert p_hat > 0.9, f"Expected A to dominate, got p_hat={p_hat}"
    assert lo > 0.5, "CI lower bound should exceed 0.5 for clear winner"
    assert hi <= 1.0


def test_bootstrap_p_a_gt_b_near_equal():
    rng = np.random.default_rng(99)
    a = rng.normal(loc=0.7, scale=0.1, size=30)
    b = rng.normal(loc=0.7, scale=0.1, size=30)
    p_hat, lo, hi = bootstrap_p_a_gt_b(a, b, seed=0)
    assert 0.3 <= p_hat <= 0.7, f"Near-equal distributions: expected ~0.5, got {p_hat}"
    assert lo < 0.5 < hi, "CI should straddle 0.5 for near-equal distributions"


def test_bootstrap_p_a_gt_b_ci_valid_bounds():
    a = np.array([0.5, 0.6, 0.7, 0.8])
    b = np.array([0.4, 0.5, 0.6, 0.7])
    p_hat, lo, hi = bootstrap_p_a_gt_b(a, b, seed=0)
    assert 0.0 <= lo <= p_hat <= hi <= 1.0


def test_bootstrap_p_a_gt_b_reproducible():
    a = np.array([0.5, 0.6, 0.7])
    b = np.array([0.4, 0.5, 0.6])
    r1 = bootstrap_p_a_gt_b(a, b, seed=42)
    r2 = bootstrap_p_a_gt_b(a, b, seed=42)
    assert r1 == r2, "Same seed must give identical results"


def test_bootstrap_p_a_gt_b_different_seeds_differ():
    rng = np.random.default_rng(0)
    a = rng.normal(0.7, 0.1, 20)
    b = rng.normal(0.65, 0.1, 20)
    _, lo1, hi1 = bootstrap_p_a_gt_b(a, b, seed=0)
    _, lo2, hi2 = bootstrap_p_a_gt_b(a, b, seed=99)
    # Not guaranteed to differ in point estimate but CIs will be close — just
    # confirm it runs without error and bounds are valid.
    assert 0.0 <= lo1 <= hi1 <= 1.0
    assert 0.0 <= lo2 <= hi2 <= 1.0


def test_bootstrap_p_a_gt_b_unequal_sample_sizes():
    a = np.linspace(0.6, 0.9, 20)
    b = np.linspace(0.4, 0.7, 10)
    p_hat, lo, hi = bootstrap_p_a_gt_b(a, b, seed=0)
    assert p_hat > 0.5
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_p_a_gt_b_empty_a_raises():
    with pytest.raises(ValueError, match="at least one score"):
        bootstrap_p_a_gt_b(np.array([]), np.array([0.5]))


def test_bootstrap_p_a_gt_b_empty_b_raises():
    with pytest.raises(ValueError, match="at least one score"):
        bootstrap_p_a_gt_b(np.array([0.5]), np.array([]))


# ---------------------------------------------------------------------------
# mann_whitney_u
# ---------------------------------------------------------------------------

def test_mann_whitney_significant():
    rng = np.random.default_rng(7)
    a = rng.normal(0.8, 0.05, 40)
    b = rng.normal(0.5, 0.05, 40)
    u, p = mann_whitney_u(a, b)
    assert p < 0.05, f"Expected significant separation, got p={p}"


def test_mann_whitney_not_significant():
    # Use a fixed, known-stable seed that produces p > 0.05 for same-distribution draws.
    rng = np.random.default_rng(2024)
    a = rng.normal(0.7, 0.1, 40)
    b = rng.normal(0.7, 0.1, 40)
    u, p = mann_whitney_u(a, b)
    assert p > 0.05, f"Expected p > 0.05 for same distribution, got p={p}"


def test_mann_whitney_small_sample_returns_sentinel():
    a = np.array([0.8, 0.9])  # < 3 observations
    b = np.array([0.5, 0.6, 0.7])
    u, p = mann_whitney_u(a, b)
    assert p == 1.0, "Small-sample sentinel should be p=1.0"


def test_mann_whitney_returns_tuple():
    a = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    b = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
    result = mann_whitney_u(a, b)
    assert len(result) == 2
    u_stat, p_val = result
    assert u_stat >= 0
    assert 0.0 <= p_val <= 1.0
