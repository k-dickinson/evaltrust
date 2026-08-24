"""Tests for the pass@k estimator and paired comparison (issue #164)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evaltrust.stats.pass_at_k import (
    PassAtKResult,
    _pass_at_k_single,
    paired_pass_at_k,
)


def _comb_estimator(n: int, c: int, k: int) -> float:
    """Reference implementation straight from the definition, for cross-checking."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


class TestSingleEstimator:
    @pytest.mark.parametrize(
        "n, c, k, expected",
        [
            (5, 2, 1, 0.4),      # pass@1 == c/n
            (5, 2, 2, 0.7),      # 1 - C(3,2)/C(5,2) = 1 - 3/10
            (5, 0, 2, 0.0),      # no correct sample -> 0
            (5, 5, 2, 1.0),      # all correct -> 1
            (3, 1, 3, 1.0),      # n == k, at least one correct -> 1
            (10, 3, 5, _comb_estimator(10, 3, 5)),
        ],
    )
    def test_matches_hand_and_reference(self, n, c, k, expected):
        assert _pass_at_k_single(n, c, k) == pytest.approx(expected, abs=1e-12)

    def test_pass_at_1_is_pass_rate(self):
        for n, c in [(20, 7), (4, 1), (100, 63)]:
            assert _pass_at_k_single(n, c, 1) == pytest.approx(c / n)

    def test_agrees_with_comb_reference_across_grid(self):
        for n in range(1, 30):
            for c in range(0, n + 1):
                for k in range(1, n + 1):
                    assert _pass_at_k_single(n, c, k) == pytest.approx(
                        _comb_estimator(n, c, k), abs=1e-12
                    )

    def test_large_n_is_finite_and_bounded(self):
        val = _pass_at_k_single(200, 40, 100)
        assert math.isfinite(val)
        assert 0.0 <= val <= 1.0

    def test_monotonic_in_c(self):
        prev = -1.0
        for c in range(0, 21):
            val = _pass_at_k_single(20, c, 5)
            assert val >= prev
            prev = val


class TestPairedPassAtK:
    def _tasks(self):
        # (n_a, c_a, n_b, c_b): B is uniformly stronger.
        return [
            (10, 3, 10, 6),
            (10, 4, 10, 7),
            (10, 2, 10, 5),
            (10, 5, 10, 8),
            (10, 1, 10, 4),
        ]

    def test_point_estimates_are_means_of_per_task(self):
        tasks = self._tasks()
        k = 2
        res = paired_pass_at_k(tasks, k, n_resamples=2000, seed=0)
        exp_a = np.mean([_pass_at_k_single(n, c, k) for n, c, _, _ in tasks])
        exp_b = np.mean([_pass_at_k_single(n, c, k) for _, _, n, c in tasks])
        assert res.pass_at_k_a == pytest.approx(exp_a)
        assert res.pass_at_k_b == pytest.approx(exp_b)
        assert res.delta == pytest.approx(exp_b - exp_a)
        assert res.k == k
        assert res.n_tasks == len(tasks)

    def test_deterministic_under_seed(self):
        tasks = self._tasks()
        r1 = paired_pass_at_k(tasks, 2, seed=7, n_resamples=3000)
        r2 = paired_pass_at_k(tasks, 2, seed=7, n_resamples=3000)
        assert r1 == r2

    def test_interval_brackets_point_and_is_ordered(self):
        res = paired_pass_at_k(self._tasks(), 2, n_resamples=4000, seed=1)
        assert res.a_low <= res.pass_at_k_a <= res.a_high
        assert res.b_low <= res.pass_at_k_b <= res.b_high
        assert res.delta_low <= res.delta <= res.delta_high

    def test_clear_gap_excludes_zero(self):
        # B far ahead on every task -> delta interval should exclude 0.
        tasks = [(10, 1, 10, 9)] * 8
        res = paired_pass_at_k(tasks, 2, n_resamples=4000, seed=0)
        assert res.delta_low > 0.0

    def test_returns_result_type_and_method_tag(self):
        res = paired_pass_at_k(self._tasks(), 2, n_resamples=1000, seed=0)
        assert isinstance(res, PassAtKResult)
        assert res.method == "passk-paired-percentile-bootstrap-v1"
        assert res.confidence == pytest.approx(0.95)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_nonpositive_k(self, bad):
        with pytest.raises(ValueError):
            paired_pass_at_k(self._tasks(), bad)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            paired_pass_at_k([], 2)

    @pytest.mark.parametrize("conf", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_bad_confidence(self, conf):
        with pytest.raises(ValueError):
            paired_pass_at_k(self._tasks(), 2, confidence=conf)

    def test_rejects_c_gt_n(self):
        with pytest.raises(ValueError):
            paired_pass_at_k([(5, 6, 5, 2)], 2)
