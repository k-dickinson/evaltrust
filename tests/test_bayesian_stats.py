"""Tests for the closed-form Bayesian decisive-pair win probability."""

import numpy as np
import pytest
from scipy import stats as sp

from evaltrust.stats.bayesian import bayesian_win_probability


@pytest.mark.parametrize("wins, losses", [(1, 0), (5, 5), (10, 0), (0, 10)])
def test_matches_scipy_beta_reference(wins, losses):
    posterior_a = wins + 0.5
    posterior_b = losses + 0.5
    expected_probability = float(sp.beta.sf(0.5, posterior_a, posterior_b))
    expected_low, expected_high = (
        float(value)
        for value in sp.beta.ppf([0.025, 0.975], posterior_a, posterior_b)
    )

    probability, low, high = bayesian_win_probability(wins, losses)

    assert probability == pytest.approx(expected_probability, abs=1e-12)
    assert low == pytest.approx(expected_low, abs=1e-12)
    assert high == pytest.approx(expected_high, abs=1e-12)


def test_swapping_labels_complements_probability_and_interval():
    probability, low, high = bayesian_win_probability(7, 2)
    swapped_probability, swapped_low, swapped_high = bayesian_win_probability(2, 7)

    assert swapped_probability == pytest.approx(1.0 - probability, abs=1e-14)
    assert swapped_low == pytest.approx(1.0 - high, abs=1e-14)
    assert swapped_high == pytest.approx(1.0 - low, abs=1e-14)


def test_is_analytic_and_deterministic_without_a_seed():
    first = bayesian_win_probability(9, 1)
    second = bayesian_win_probability(9, 1)
    assert first == second


def test_returns_plain_python_floats():
    result = bayesian_win_probability(np.int64(3), np.int64(1))
    assert all(type(value) is float for value in result)


@pytest.mark.parametrize(
    "wins, losses",
    [(-1, 1), (1, -1), (0, 0), (1.5, 1), (1, 1.5), (True, 1)],
)
def test_rejects_invalid_counts(wins, losses):
    with pytest.raises(ValueError):
        bayesian_win_probability(wins, losses)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        bayesian_win_probability(1, 1, confidence=confidence)
