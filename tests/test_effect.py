"""Tests for effect size: paired Cohen's d and its plain-language magnitude."""

import numpy as np
import pytest

from evallab.stats.effect import cohens_d_paired, magnitude_label


def test_cohens_d_matches_hand_calculation():
    diffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # mean = 3.0, sample std (ddof=1) = sqrt(2.5) = 1.5811..., d = 1.897...
    assert cohens_d_paired(diffs) == pytest.approx(3.0 / np.sqrt(2.5))


def test_cohens_d_is_zero_when_no_difference():
    assert cohens_d_paired(np.zeros(10)) == 0.0


def test_cohens_d_is_infinite_for_perfectly_consistent_effect():
    # Nonzero mean, zero variance: an infinitely reliable effect.
    assert np.isinf(cohens_d_paired(np.full(8, 2.0)))


def test_cohens_d_is_negative_when_a_beats_b():
    diffs = np.array([-1.0, -2.0, -3.0])
    assert cohens_d_paired(diffs) < 0


@pytest.mark.parametrize(
    "d,label",
    [
        (0.0, "negligible"),
        (0.1, "negligible"),
        (0.2, "small"),
        (0.4, "small"),
        (0.5, "medium"),
        (0.7, "medium"),
        (0.8, "large"),
        (1.5, "large"),
        (-0.9, "large"),  # magnitude, sign-agnostic
        (float("inf"), "large"),
    ],
)
def test_magnitude_label_thresholds(d, label):
    assert magnitude_label(d) == label
