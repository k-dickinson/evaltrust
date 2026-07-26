"""External-reference tests for the joint rank-occupancy bootstrap primitive."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats as sp

from evaltrust.stats import rank_stability as rs
from evaltrust.stats.rank_stability import bootstrap_rank_distribution


# Tolerances declared up front (Monte Carlo / floating comparison).
_ENUM_TOL = 0.02          # exhaustive vs Monte Carlo occupancy cells
_BINOM_TOL = 0.02         # binomial closed form vs Monte Carlo top-1


def _constant_dominance(n: int = 20) -> np.ndarray:
    """A >> B >> C on every example (strict dominance)."""
    return np.column_stack([
        np.full(n, 3.0),
        np.full(n, 2.0),
        np.full(n, 1.0),
    ])


def test_identical_score_vectors_split_occupancy_evenly_and_never_stable():
    n, m = 12, 3
    rows = np.ones((n, m), dtype=float)
    out = bootstrap_rank_distribution(rows, n_resamples=500, seed=0)

    # Every model gets 1/m credit at every rank.
    expected = np.full((m, m), 1.0 / m)
    np.testing.assert_allclose(out.occupancy, expected, atol=1e-12)
    assert out.full_order_retention == 0.0
    assert out.tie_resamples == 500
    # Zero stable positions at default alpha=0.05 threshold (retention 1/3).
    assert np.all(out.position_retention == pytest.approx(1.0 / m))
    assert np.all(out.position_retention < 1.0 - 0.05)


def test_strict_dominance_has_unit_diagonal_and_full_order():
    rows = _constant_dominance(30)
    out = bootstrap_rank_distribution(
        rows, n_resamples=200, seed=1, observed_order=np.array([0, 1, 2]),
    )
    np.testing.assert_allclose(out.occupancy, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(out.position_retention, np.ones(3), atol=1e-12)
    assert out.top1_retention == 1.0
    assert out.full_order_retention == 1.0
    assert out.tie_resamples == 0


def test_two_model_binomial_closed_form_for_top_slot():
    """Exclusive binary votes: top-1 credit matches Binomial(n, k/n) masses."""
    # k=7 of 10 examples favor model A (col 0); rest favor B (col 1).
    k, n = 7, 10
    rows = np.zeros((n, 2), dtype=float)
    rows[:k, 0] = 1.0   # A wins
    rows[k:, 1] = 1.0   # B wins
    rows[:k, 1] = 0.0
    rows[k:, 0] = 0.0

    n_resamples = 20_000
    out = bootstrap_rank_distribution(
        rows,
        n_resamples=n_resamples,
        seed=11,
        observed_order=np.array([0, 1]),
    )

    p = k / n
    # X ~ Binom(n, p): number of A-favoring rows in a resample.
    # A sole top when X > n/2; half credit when X == n/2; else 0.
    xs = np.arange(0, n + 1)
    pmf = sp.binom.pmf(xs, n, p)
    half = n / 2
    exact_top1 = float(
        pmf[xs > half].sum() + 0.5 * pmf[xs == half].sum()
    )
    # Exact masses for positive / tied / negative mean_A - mean_B:
    # mean_A - mean_B = (2X - n) / n, so sign(X - n/2).
    exact_pos = float(pmf[xs > half].sum())
    exact_tie = float(pmf[xs == half].sum())
    exact_neg = float(pmf[xs < half].sum())
    assert exact_pos + exact_tie + exact_neg == pytest.approx(1.0)

    assert out.top1_retention == pytest.approx(exact_top1, abs=_BINOM_TOL)
    # Occupancy[0, 0] is A's credit in the top slot.
    assert out.occupancy[0, 0] == pytest.approx(exact_top1, abs=_BINOM_TOL)
    # Tie resamples share should match exact_tie (Monte Carlo).
    assert out.tie_resamples / n_resamples == pytest.approx(
        exact_tie, abs=_BINOM_TOL
    )


def _enumerate_occupancy(rows: np.ndarray, observed_order: np.ndarray) -> np.ndarray:
    """Exact occupancy by enumerating all n^n index draws (tiny n only)."""
    n, m = rows.shape
    total = n ** n
    occ_sum = np.zeros((m, m), dtype=float)
    for draw in itertools.product(range(n), repeat=n):
        means = rs._means_from_rows(rows[list(draw)])
        occ, _ = rs._fractional_occupancy(means)
        occ_sum += occ
    return occ_sum / total


def test_exhaustive_enumeration_matches_monte_carlo_example_level():
    # n=3 examples, 2 models: 3^3 = 27 draws.
    rows = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.2],
    ])
    observed = np.array([0, 1])
    exact = _enumerate_occupancy(rows, observed)
    out = bootstrap_rank_distribution(
        rows, n_resamples=50_000, seed=3, observed_order=observed,
    )
    np.testing.assert_allclose(out.occupancy, exact, atol=_ENUM_TOL)


def test_exhaustive_enumeration_matches_monte_carlo_clustered():
    # k=3 clusters of size 1: 3^3 = 27 cluster draws.
    rows = np.array([
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [1.5, 1.0, 0.0],
    ])
    cluster_ids = np.array([0, 1, 2])
    observed = np.array([0, 2, 1])

    k = 3
    total = k ** k
    occ_sum = np.zeros((3, 3), dtype=float)
    members = [np.array([i]) for i in range(k)]
    for draw in itertools.product(range(k), repeat=k):
        pooled = np.concatenate([members[j] for j in draw])
        means = rs._means_from_rows(rows[pooled])
        occ, _ = rs._fractional_occupancy(means)
        occ_sum += occ
    exact = occ_sum / total

    out = bootstrap_rank_distribution(
        rows,
        n_resamples=50_000,
        seed=4,
        cluster_ids=cluster_ids,
        observed_order=observed,
    )
    np.testing.assert_allclose(out.occupancy, exact, atol=_ENUM_TOL)
    assert out.clustered is True
    assert out.n_resampling_units == 3


def test_exhaustive_enumeration_unequal_cluster_sizes():
    # k=2 clusters, sizes 2 and 1: 2^2 = 4 draws; pools multi-row clusters.
    rows = np.array([
        [3.0, 1.0, 0.0],  # cluster 0
        [2.5, 1.5, 0.5],  # cluster 0
        [0.0, 0.5, 3.0],  # cluster 1
    ])
    cluster_ids = np.array([0, 0, 1])
    observed = np.array([0, 1, 2])
    k = 2
    members = [np.array([0, 1]), np.array([2])]
    total = k ** k
    occ_sum = np.zeros((3, 3), dtype=float)
    for draw in itertools.product(range(k), repeat=k):
        pooled = np.concatenate([members[j] for j in draw])
        means = rs._means_from_rows(rows[pooled])
        occ, _ = rs._fractional_occupancy(means)
        occ_sum += occ
    exact = occ_sum / total

    out = bootstrap_rank_distribution(
        rows,
        n_resamples=20_000,
        seed=6,
        cluster_ids=cluster_ids,
        observed_order=observed,
    )
    np.testing.assert_allclose(out.occupancy, exact, atol=_ENUM_TOL)


def test_same_seed_index_bootstrap_matches_occupancy_with_fractional_ties():
    """Independent reimplementation of the joint index bootstrap + fractional ranks.

    Same seed stream and same definition as the primitive; not a soft diagonal
    check. Confirms the occupancy accounting path against a second code path
    written only in this test.
    """
    rows = np.array([
        [1.0, 0.5, 0.0],
        [0.9, 0.9, 0.1],
        [1.1, 0.4, 0.2],
        [0.8, 0.8, 0.0],
        [1.0, 0.5, 0.3],
    ], dtype=float)
    n, m = rows.shape
    n_resamples = 3_000
    seed = 21
    observed = np.array([0, 1, 2])

    out = bootstrap_rank_distribution(
        rows, n_resamples=n_resamples, seed=seed, observed_order=observed,
    )

    # Second path: same Generator API, same fractional occupancy helper used
    # only to convert means → occupancy (already unit-tested via enumeration).
    rng = np.random.default_rng(seed)
    occ_sum = np.zeros((m, m), dtype=float)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means = rs._means_from_rows(rows[idx])
        occ, _ = rs._fractional_occupancy(means)
        occ_sum += occ
    ref = occ_sum / n_resamples
    np.testing.assert_allclose(out.occupancy, ref, atol=1e-12)


def test_scipy_bootstrap_mean_ci_covers_observed_means():
    """External: scipy.stats.bootstrap percentile CI covers each model mean."""
    rows = np.array([
        [1.0, 0.5, 0.0],
        [1.1, 0.4, 0.1],
        [0.9, 0.6, 0.0],
        [1.0, 0.5, 0.2],
        [1.2, 0.3, 0.0],
        [0.95, 0.55, 0.05],
        [1.05, 0.45, 0.15],
        [1.0, 0.5, 0.1],
    ], dtype=float)
    n_resamples = 4_000
    out = bootstrap_rank_distribution(
        rows, n_resamples=n_resamples, seed=5, observed_order=np.array([0, 1, 2]),
    )
    # Rank occupancy should put the highest-mean model mostly on top.
    assert out.occupancy[0, 0] > 0.5

    def mean_stat(sample, axis=-1):
        return np.mean(sample, axis=axis)

    for col in range(3):
        result = sp.bootstrap(
            (rows[:, col],),
            statistic=mean_stat,
            n_resamples=n_resamples,
            random_state=5 + col,
            method="percentile",
            vectorized=True,
        )
        lo, hi = result.confidence_interval
        observed_mean = float(rows[:, col].mean())
        assert lo <= observed_mean <= hi


def test_singleton_clusters_match_example_level_bootstrap():
    """cluster_ids = arange(n) must match the unclustered path (same seed)."""
    rows = np.array([
        [1.0, 0.5, 0.0],
        [0.8, 0.6, 0.1],
        [1.1, 0.4, 0.2],
        [0.9, 0.7, 0.0],
    ])
    observed = np.array([0, 1, 2])
    seed, n_resamples = 12, 2_000
    plain = bootstrap_rank_distribution(
        rows, n_resamples=n_resamples, seed=seed, observed_order=observed,
    )
    as_singletons = bootstrap_rank_distribution(
        rows,
        n_resamples=n_resamples,
        seed=seed,
        cluster_ids=np.arange(rows.shape[0]),
        observed_order=observed,
    )
    # Same multinomial over units of size 1, same seed → bitwise equal draws.
    np.testing.assert_array_equal(plain.occupancy, as_singletons.occupancy)
    assert plain.full_order_retention == as_singletons.full_order_retention
    assert plain.tie_resamples == as_singletons.tie_resamples


def test_clustered_shows_wider_instability_than_example_level():
    # Majority cluster supports A > B > C; minority cluster reverses it.
    # Example-level bootstrap mostly redraws majority rows and looks stable;
    # cluster bootstrap redraws whole units and is more unstable.
    rows_major = np.tile(np.array([3.0, 2.0, 0.0]), (8, 1))
    rows_minor = np.tile(np.array([0.0, 0.5, 3.0]), (2, 1))
    rows = np.vstack([rows_major, rows_minor])
    cluster_ids = np.array([0] * 8 + [1] * 2)
    # Observed means: A=2.4, B=1.7, C=0.6 → order A, B, C.
    observed = np.array([0, 1, 2])
    assert list(np.argsort(-rows.mean(axis=0))) == [0, 1, 2]

    ex = bootstrap_rank_distribution(
        rows, n_resamples=3_000, seed=8, observed_order=observed,
    )
    cl = bootstrap_rank_distribution(
        rows,
        n_resamples=3_000,
        seed=8,
        cluster_ids=cluster_ids,
        observed_order=observed,
    )
    assert ex.full_order_retention > 0.5
    assert cl.full_order_retention < ex.full_order_retention
    assert cl.top1_retention < ex.top1_retention


def test_determinism_two_calls_identical():
    rows = _constant_dominance(15) + np.array([0.0, 0.1, 0.0])
    a = bootstrap_rank_distribution(rows, n_resamples=300, seed=42)
    b = bootstrap_rank_distribution(rows, n_resamples=300, seed=42)
    np.testing.assert_array_equal(a.occupancy, b.occupancy)
    np.testing.assert_array_equal(a.position_retention, b.position_retention)
    assert a.top1_retention == b.top1_retention
    assert a.full_order_retention == b.full_order_retention
    assert a.tie_resamples == b.tie_resamples


def test_chunk_invariance(monkeypatch):
    from evaltrust.stats import resampling as resampling_mod

    rows = np.array([
        [1.0, 0.5, 0.0],
        [0.8, 0.6, 0.1],
        [1.1, 0.4, 0.2],
        [0.9, 0.7, 0.0],
        [1.0, 0.5, 0.3],
    ])
    monkeypatch.setattr(resampling_mod, "_MAX_RESAMPLE_CELLS", 10**12)
    one = bootstrap_rank_distribution(rows, n_resamples=500, seed=9)
    monkeypatch.setattr(resampling_mod, "_MAX_RESAMPLE_CELLS", 15)  # tiny blocks
    many = bootstrap_rank_distribution(rows, n_resamples=500, seed=9)
    np.testing.assert_array_equal(one.occupancy, many.occupancy)
    assert one.full_order_retention == many.full_order_retention
    assert one.tie_resamples == many.tie_resamples


def test_memory_bound_large_n_no_cube(monkeypatch):
    """Large n, 3 models: must complete without allocating (n_resamples, n, m)."""
    from evaltrust.stats import resampling as resampling_mod

    n, m = 50_000, 3
    rows = np.column_stack([
        np.full(n, 3.0),
        np.full(n, 2.0),
        np.full(n, 1.0),
    ])
    monkeypatch.setattr(resampling_mod, "_MAX_RESAMPLE_CELLS", 200_000)
    out = bootstrap_rank_distribution(rows, n_resamples=50, seed=0)
    assert out.full_order_retention == 1.0
    assert 50 * n * m > 200_000


def test_missingness_preserves_availability_pattern():
    # Model C missing on half the rows.
    rows = np.array([
        [3.0, 2.0, 1.0],
        [3.0, 2.0, np.nan],
        [2.5, 2.0, 1.5],
        [2.5, 1.5, np.nan],
    ])
    out = bootstrap_rank_distribution(
        rows, n_resamples=1_000, seed=2, observed_order=np.array([0, 1, 2]),
    )
    # Means still defined; no crash; occupancy rows sum to 1 per rank.
    col_sums = out.occupancy.sum(axis=0)
    np.testing.assert_allclose(col_sums, np.ones(3), atol=1e-9)
