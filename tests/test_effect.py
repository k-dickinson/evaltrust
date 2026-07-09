"""Tests for effect size: paired Cohen's d and its plain-language magnitude."""

import numpy as np
import pytest

from evaltrust.stats.effect import (
    cohens_d_paired,
    cohens_d_paired_along_rows,
    magnitude_label,
)


def test_cohens_d_matches_hand_calculation():
    diffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # mean = 3.0, sample std (ddof=1) = sqrt(2.5) = 1.5811..., d = 1.897...
    assert cohens_d_paired(diffs) == pytest.approx(3.0 / np.sqrt(2.5))


def test_cohens_d_along_rows_matches_the_scalar_version():
    # The vectorized row-wise Cohen's d (used for the bootstrap CI) must agree
    # with the scalar cohens_d_paired for every row, including degenerate rows.
    rng = np.random.default_rng(0)
    for _ in range(30):
        x = rng.normal(0.3, 1.0, size=int(rng.integers(2, 60)))
        assert cohens_d_paired_along_rows(x[None, :])[0] == pytest.approx(
            cohens_d_paired(x))


def test_cohens_d_along_rows_degenerate_rows_are_never_nan():
    rows = np.array([[2.0, 2.0, 2.0],    # zero variance, nonzero mean -> +inf
                     [-3.0, -3.0, -3.0],  # zero variance, negative mean -> -inf
                     [0.0, 0.0, 0.0],     # all zero -> 0.0
                     [0.1, 0.2, 0.3]])    # ordinary -> finite
    d = cohens_d_paired_along_rows(rows)
    assert not np.isnan(d).any()
    assert d[0] == np.inf and d[1] == -np.inf and d[2] == 0.0
    assert np.isfinite(d[3])


def test_cohens_d_along_rows_single_column_matches_scalar():
    # n == 1 columns: sd is undefined, so d degrades exactly like the scalar
    # (+/-inf for a nonzero value, 0.0 for zero) -- never NaN.
    d = cohens_d_paired_along_rows(np.array([[5.0], [0.0]]))
    assert d[0] == np.inf and d[1] == 0.0


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
