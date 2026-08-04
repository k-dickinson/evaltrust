"""Tests for paired Cohen and rank effect sizes and their magnitude labels."""

import numpy as np
import pytest
from scipy import stats

from evaltrust.stats.effect import (
    cohens_d_paired,
    cohens_d_paired_along_rows,
    magnitude_label,
    magnitude_label_rank_r,
    probability_of_superiority_paired,
    rank_biserial_paired,
    rank_biserial_paired_along_rows,
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
        (0.19, "negligible"),
        (0.2, "small"),
        (0.49, "small"),
        (0.5, "medium"),
        (0.79, "medium"),
        (0.8, "large"),
    ],
)
def test_magnitude_label_thresholds(d, label):
    assert magnitude_label(d) == label


@pytest.mark.parametrize("d", [0.19, 0.2, 0.49, 0.5, 0.79, 0.8])
def test_magnitude_label_is_sign_agnostic(d):
    assert magnitude_label(-d) == magnitude_label(d)


def test_probability_of_superiority_matches_brute_force_enumeration():
    rng = np.random.default_rng(166)
    for size in range(1, 40):
        diffs = rng.integers(-4, 5, size=size).astype(float)
        expected = (
            sum(value > 0 for value in diffs)
            + 0.5 * sum(value == 0 for value in diffs)
        ) / size
        actual = probability_of_superiority_paired(diffs)
        assert isinstance(actual, float)
        assert actual == pytest.approx(expected)
        assert 0.0 <= actual <= 1.0


def test_probability_of_superiority_matches_mann_whitney_relationship():
    # Expanding two independent samples into every pairwise difference makes
    # the paired probability calculation equal U / (n_x * n_y), including
    # half-credit for ties.
    x = np.array([1.0, 2.0, 2.0, 5.0])
    y = np.array([0.0, 2.0, 4.0])
    all_pairwise_differences = (x[:, None] - y[None, :]).ravel()
    u = float(stats.mannwhitneyu(x, y, alternative="greater").statistic)
    expected = u / (x.size * y.size)
    assert probability_of_superiority_paired(
        all_pairwise_differences
    ) == pytest.approx(expected)


def test_rank_biserial_matches_scipy_wilcoxon_with_ties_and_zeros():
    rng = np.random.default_rng(166)
    for _ in range(100):
        diffs = rng.integers(-4, 5, size=int(rng.integers(2, 60))).astype(float)
        if not np.any(diffs):
            diffs[0] = 1.0
        nonzero = diffs[diffs != 0.0]
        w_positive = float(
            stats.wilcoxon(
                nonzero,
                zero_method="wilcox",
                alternative="greater",
                method="auto",
            ).statistic
        )
        total_rank = nonzero.size * (nonzero.size + 1) / 2
        expected = (2 * w_positive - total_rank) / total_rank
        actual = rank_biserial_paired(diffs)
        assert isinstance(actual, float)
        assert actual == pytest.approx(expected)
        assert -1.0 <= actual <= 1.0


def test_rank_biserial_midranks_ties_and_drops_zeros():
    # |1| ties get rank 1.5 each and |2| ties get rank 3.5 each.
    # W+ = 3, W- = 7, so r = (3 - 7) / 10 = -0.4.
    diffs = np.array([1.0, 1.0, -2.0, -2.0, 0.0])
    assert rank_biserial_paired(diffs) == pytest.approx(-0.4)


def test_rank_biserial_along_rows_matches_scalar_and_never_returns_nan():
    rng = np.random.default_rng(166)
    rows = rng.integers(-4, 5, size=(40, 50)).astype(float)
    rows[0] = 0.0
    rows[1] = 3.0
    rows[2] = -2.0

    actual = rank_biserial_paired_along_rows(rows)

    expected = np.array([rank_biserial_paired(row) for row in rows])
    assert not np.isnan(actual).any()
    assert actual == pytest.approx(expected)
    assert actual[0] == 0.0
    assert actual[1] == 1.0
    assert actual[2] == -1.0


def test_rank_biserial_along_rows_single_column_matches_scalar():
    rows = np.array([[5.0], [-5.0], [0.0]])
    actual = rank_biserial_paired_along_rows(rows)
    assert actual.tolist() == [1.0, -1.0, 0.0]


def test_rank_statistics_define_all_zero_differences():
    diffs = np.zeros(12)
    assert probability_of_superiority_paired(diffs) == 0.5
    assert rank_biserial_paired(diffs) == 0.0
    assert rank_biserial_paired_along_rows(diffs[None, :])[0] == 0.0


@pytest.mark.parametrize(
    "function",
    [
        probability_of_superiority_paired,
        rank_biserial_paired,
        rank_biserial_paired_along_rows,
    ],
)
def test_rank_statistics_reject_empty_input(function):
    values = np.empty((1, 0)) if function is rank_biserial_paired_along_rows else []
    with pytest.raises(ValueError, match="at least one"):
        function(values)


@pytest.mark.parametrize(
    "r,label",
    [
        (0.099, "negligible"),
        (0.1, "small"),
        (0.299, "small"),
        (0.3, "medium"),
        (0.499, "medium"),
        (0.5, "large"),
    ],
)
def test_rank_magnitude_label_uses_r_family_thresholds(r, label):
    assert magnitude_label_rank_r(r) == label
    assert magnitude_label_rank_r(-r) == label
