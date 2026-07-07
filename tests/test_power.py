"""Tests for power analysis, validated against statsmodels' TTestPower.

Power answers the question that catches underpowered evals: "even if there were
a real effect this size, was my sample large enough to have detected it?"
"""

import pytest
from statsmodels.stats.power import TTestPower

from evallab.stats.power import achieved_power, required_n

_REF = TTestPower()


def test_power_at_zero_effect_equals_alpha():
    # With no true effect, a two-sided test rejects at exactly its alpha rate.
    assert achieved_power(0.0, n=50, alpha=0.05) == pytest.approx(0.05, abs=1e-6)


def test_power_increases_with_sample_size():
    assert achieved_power(0.3, n=20) < achieved_power(0.3, n=200)


def test_power_increases_with_effect_size():
    assert achieved_power(0.2, n=50) < achieved_power(0.6, n=50)


@pytest.mark.parametrize("d,n", [(0.2, 40), (0.5, 30), (0.8, 25), (0.35, 120)])
def test_achieved_power_matches_statsmodels(d, n):
    ref = _REF.power(effect_size=d, nobs=n, alpha=0.05, alternative="two-sided")
    assert achieved_power(d, n=n, alpha=0.05) == pytest.approx(ref, abs=1e-6)


def test_power_is_capped_in_unit_interval():
    assert achieved_power(5.0, n=100) <= 1.0
    assert achieved_power(5.0, n=100) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("d", [0.2, 0.35, 0.5, 0.8])
def test_required_n_matches_statsmodels_within_one(d):
    ref = _REF.solve_power(effect_size=d, power=0.8, alpha=0.05,
                           alternative="two-sided")
    import math
    ref_n = math.ceil(ref)
    assert abs(required_n(d, power=0.8, alpha=0.05) - ref_n) <= 1


def test_required_n_is_the_smallest_sufficient_sample():
    d, power = 0.4, 0.8
    n = required_n(d, power=power)
    assert achieved_power(d, n=n) >= power
    assert achieved_power(d, n=n - 1) < power
