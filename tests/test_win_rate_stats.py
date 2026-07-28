"""External-reference tests for the all-example paired win-rate primitive."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats as sp

from evaltrust.stats.win_rate import paired_win_rate


@pytest.mark.parametrize(
    ("differences", "rate", "wins_a", "ties", "wins_b"),
    [
        ([-2.0, -1.0], 1.0, 2, 0, 0),
        ([2.0, 1.0], 0.0, 0, 0, 2),
        ([0.0, 0.0], 0.5, 0, 2, 0),
        ([-2.0, -1.0, 0.0, 4.0], 0.625, 2, 1, 1),
    ],
)
def test_point_estimator_and_counts_by_hand(
    differences, rate, wins_a, ties, wins_b
):
    result = paired_win_rate(differences, n_resamples=99, seed=7)

    assert result.win_rate_a == rate
    assert result.n_examples == len(differences)
    assert result.n_wins_a == wins_a
    assert result.n_ties == ties
    assert result.n_wins_b == wins_b
    assert result.method == "half-tie-percentile-bootstrap-v1"
    assert type(result.win_rate_a) is float
    assert type(result.interval_low) is float
    assert type(result.interval_high) is float
    assert type(result.confidence) is float
    assert type(result.n_examples) is int


def test_ties_use_exact_equality_without_a_tolerance():
    result = paired_win_rate(
        [0.0, 1e-300, -1e-300],
        n_resamples=99,
        seed=3,
    )

    assert result.win_rate_a == 0.5
    assert result.n_wins_a == 1
    assert result.n_ties == 1
    assert result.n_wins_b == 1


def _exact_bootstrap_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    """Enumerate all n**n example-index resamples for tiny hand fixtures."""
    n = int(values.size)
    estimates = []
    for draw in itertools.product(range(n), repeat=n):
        sample = values[list(draw)]
        estimates.append(
            float(np.mean(np.where(sample < 0.0, 1.0, np.where(sample == 0.0, 0.5, 0.0))))
        )
    alpha = 1.0 - confidence
    return (
        # Use the exact distribution quantile. Linear interpolation over the
        # short n**n expansion is a finite-list artifact (for n=2 it blends
        # adjacent support points); a large seeded bootstrap converges to these
        # inverse-CDF percentile bounds.
        float(np.percentile(
            estimates, 100.0 * alpha / 2.0, method="inverted_cdf"
        )),
        float(np.percentile(
            estimates,
            100.0 * (1.0 - alpha / 2.0),
            method="inverted_cdf",
        )),
    )


@pytest.mark.parametrize(
    "differences",
    [
        np.array([-1.0, 1.0]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([-1.0, -1.0, 0.0, 1.0]),
    ],
)
def test_seeded_percentile_bootstrap_matches_exhaustive_enumeration(differences):
    exact_low, exact_high = _exact_bootstrap_interval(differences, 0.95)
    result = paired_win_rate(
        differences,
        confidence=0.95,
        n_resamples=100_000,
        seed=17,
    )

    assert result.interval_low == pytest.approx(exact_low, abs=0.01)
    assert result.interval_high == pytest.approx(exact_high, abs=0.01)


def test_no_tie_point_estimate_matches_scipy_binomial_proportion():
    differences = np.array([-3.0, -2.0, -1.0, 1.0, 2.0])
    result = paired_win_rate(differences, n_resamples=99, seed=4)
    reference = sp.binomtest(k=3, n=5).proportion_estimate

    assert result.n_ties == 0
    assert result.win_rate_a == float(reference)


def test_seed_controls_only_the_interval_resamples():
    differences = [-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0]
    same_a = paired_win_rate(differences, n_resamples=101, seed=42)
    same_b = paired_win_rate(differences, n_resamples=101, seed=42)
    other = paired_win_rate(differences, n_resamples=101, seed=43)

    assert same_a == same_b
    assert other.win_rate_a == same_a.win_rate_a
    assert (other.interval_low, other.interval_high) != (
        same_a.interval_low,
        same_a.interval_high,
    )


def test_cluster_resampling_changes_only_interval_and_is_wider_when_rows_correlate():
    differences = np.array([-1.0] * 8 + [1.0] * 2)
    clusters = ["majority"] * 8 + ["minority"] * 2

    example = paired_win_rate(
        differences,
        n_resamples=20_000,
        seed=8,
    )
    clustered = paired_win_rate(
        differences,
        n_resamples=20_000,
        seed=8,
        clusters=clusters,
    )

    assert clustered.win_rate_a == example.win_rate_a == 0.8
    assert clustered.interval_low < example.interval_low
    assert clustered.interval_high >= example.interval_high
    assert (
        clustered.interval_high - clustered.interval_low
        > example.interval_high - example.interval_low
    )


def test_singleton_clusters_are_byte_identical_to_example_resampling():
    differences = np.array([-1.0, 0.0, 1.0, -2.0])
    plain = paired_win_rate(
        differences,
        n_resamples=2_000,
        seed=12,
    )
    singletons = paired_win_rate(
        differences,
        n_resamples=2_000,
        seed=12,
        clusters=["a", "b", "c", "d"],
    )

    assert singletons == plain


def test_clustered_percentile_bootstrap_matches_exhaustive_cluster_enumeration():
    differences = np.array([-1.0, -1.0, 0.0, 1.0])
    clusters = ["two-row-win", "two-row-win", "tie", "loss"]
    labels = list(dict.fromkeys(clusters))
    members = [
        np.array([index for index, label in enumerate(clusters) if label == target])
        for target in labels
    ]
    estimates = []
    for draw in itertools.product(range(len(members)), repeat=len(members)):
        pooled = np.concatenate([differences[members[index]] for index in draw])
        estimates.append(
            float(np.mean(np.where(
                pooled < 0.0,
                1.0,
                np.where(pooled == 0.0, 0.5, 0.0),
            )))
        )
    exact_low, exact_high = np.percentile(
        estimates,
        [25.0, 75.0],
        method="inverted_cdf",
    )

    result = paired_win_rate(
        differences,
        confidence=0.5,
        n_resamples=100_000,
        seed=23,
        clusters=clusters,
    )

    assert result.interval_low == pytest.approx(float(exact_low), abs=0.01)
    assert result.interval_high == pytest.approx(float(exact_high), abs=0.01)


@pytest.mark.parametrize(
    "differences",
    [
        [],
        [0.0, np.nan],
        [0.0, np.inf],
        [0.0, -np.inf],
    ],
)
def test_empty_or_nonfinite_differences_raise(differences):
    with pytest.raises(ValueError):
        paired_win_rate(differences)


def test_cluster_labels_must_align_with_examples():
    with pytest.raises(ValueError, match="same length"):
        paired_win_rate([-1.0, 1.0], clusters=["only-one"])
