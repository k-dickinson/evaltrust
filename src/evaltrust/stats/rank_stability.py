"""Joint bootstrap of multi-model mean-score rankings.

Resamples example rows (or whole clusters) with replacement, recomputes each
model's mean from available scores, and accumulates a model-by-rank occupancy
matrix. Exact mean ties split occupancy evenly across the tied slots so that
lexical model-id order is never counted as stability.

This is an advisory diagnostic (bootstrap retention / rank occupancy). It is
not a calibrated confidence interval and does not claim coverage near ties.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import resampling as _resampling


@dataclass(frozen=True)
class RankBootstrapResult:
    """Plain-scalar views of one joint bootstrap distribution."""

    occupancy: np.ndarray          # (n_models, n_ranks) mean fractional credit
    position_retention: np.ndarray  # (n_models,) retention of each observed slot
    top1_retention: float
    full_order_retention: float
    tie_resamples: int
    n_resamples: int
    n_resampling_units: int
    clustered: bool


def _means_from_rows(score_rows: np.ndarray) -> np.ndarray:
    """Per-model mean over non-NaN scores; -inf when a model has no scores."""
    with np.errstate(all="ignore"):
        counts = np.sum(~np.isnan(score_rows), axis=0)
        totals = np.nansum(score_rows, axis=0)
    means = np.full(score_rows.shape[1], -np.inf, dtype=float)
    ok = counts > 0
    means[ok] = totals[ok] / counts[ok]
    return means


def _fractional_occupancy(means: np.ndarray) -> tuple[np.ndarray, bool]:
    """One-resample occupancy matrix and whether any exact mean tie occurred.

    Tied models that share ``g`` consecutive slots each get ``1/g`` credit in
    every tied slot. Ranks are 0-based from best (highest mean) to worst.
    """
    m = int(means.size)
    occ = np.zeros((m, m), dtype=float)
    order = np.argsort(-means, kind="mergesort")
    sorted_means = means[order]
    has_tie = False
    i = 0
    while i < m:
        j = i + 1
        while j < m and sorted_means[j] == sorted_means[i]:
            j += 1
        group = order[i:j]
        size = j - i
        if size > 1:
            has_tie = True
        credit = 1.0 / size
        for slot in range(i, j):
            for model_idx in group:
                occ[int(model_idx), slot] += credit
        i = j
    return occ, has_tie


def _strict_order_matches(means: np.ndarray, observed_order: np.ndarray) -> bool:
    """True when means realise the observed strict ranking with no ties."""
    ordered_means = means[observed_order]
    if not np.isfinite(ordered_means).all():
        return False
    return bool(np.all(np.diff(ordered_means) < 0))


def bootstrap_rank_distribution(
    score_rows: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    cluster_ids: np.ndarray | None = None,
    observed_order: np.ndarray | None = None,
) -> RankBootstrapResult:
    """Bootstrap rank occupancy for models scored on shared example rows.

    Parameters
    ----------
    score_rows:
        Array of shape ``(n_examples, n_models)``. Missing scores are NaN.
        Means use available scores only; a model with no scores in a resample
        gets mean ``-inf`` and ranks last among such models.
    n_resamples:
        Number of bootstrap resamples.
    seed:
        Single RNG seed for the whole joint procedure.
    cluster_ids:
        Optional length-``n_examples`` labels. When provided, whole clusters are
        drawn with replacement and their rows pooled (ungrouped rows should
        already be encoded as singleton labels).
    observed_order:
        Optional permutation of model indices giving the observed ranking
        (best first). When omitted, observed order is the argsort of observed
        means with index order as the sole slot tie-break (not occupancy credit).

    Returns
    -------
    RankBootstrapResult
        Occupancy matrix, per-position retention against ``observed_order``,
        top-1 and full-order retention, and the tie-resample count.
    """
    rows = np.asarray(score_rows, dtype=float)
    if rows.ndim != 2:
        raise ValueError("score_rows must be 2-D (n_examples, n_models)")
    n, m = rows.shape
    if n < 1 or m < 1:
        raise ValueError("score_rows must have at least one example and one model")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    if observed_order is None:
        obs_means = _means_from_rows(rows)
        observed_order = np.argsort(-obs_means, kind="mergesort")
    else:
        observed_order = np.asarray(observed_order, dtype=int)
        if (
            observed_order.shape != (m,)
            or set(observed_order.tolist()) != set(range(m))
        ):
            raise ValueError("observed_order must be a permutation of 0..m-1")

    rng = np.random.default_rng(seed)
    occupancy_sum = np.zeros((m, m), dtype=float)
    retention_sum = np.zeros(m, dtype=float)
    full_order_count = 0
    tie_resamples = 0

    if cluster_ids is None:
        n_units = n
        clustered = False
        # Bound (block * n * m) working cells via the shared resampling cap so
        # monkeypatched tests and large-n runs share one memory policy.
        cells_per = max(n * m, 1)
        block_rows = max(
            1,
            min(n_resamples, _resampling._MAX_RESAMPLE_CELLS // cells_per),
        )

        pos = 0
        while pos < n_resamples:
            block = min(block_rows, n_resamples - pos)
            idx = rng.integers(0, n, size=(block, n))
            for b in range(block):
                means = _means_from_rows(rows[idx[b]])
                occ, has_tie = _fractional_occupancy(means)
                occupancy_sum += occ
                if has_tie:
                    tie_resamples += 1
                for slot, model_idx in enumerate(observed_order):
                    retention_sum[slot] += occ[int(model_idx), slot]
                if _strict_order_matches(means, observed_order):
                    full_order_count += 1
            pos += block
    else:
        labels = np.asarray(cluster_ids)
        if labels.shape != (n,):
            raise ValueError("cluster_ids must have length n_examples")
        unique, inverse = np.unique(labels, return_inverse=True)
        k = int(unique.size)
        # Group row indices by cluster in O(n log n). Avoids a per-cluster
        # flatnonzero scan that would be O(n * k) when many labels are singletons.
        row_order = np.argsort(inverse, kind="stable")
        boundaries = np.flatnonzero(np.diff(inverse[row_order])) + 1
        members = np.split(row_order, boundaries)
        n_units = k
        clustered = True
        # Cluster path mirrors bootstrap_ci_clustered: one draw of k units per
        # resample, then pool rows. Peak per draw is O(sum of chosen cluster
        # sizes + m^2), not an (n_resamples, n, m) cube. The example-level
        # cells cap does not apply; cluster counts are typically small.

        for _ in range(n_resamples):
            chosen = rng.integers(0, k, size=k)
            pooled_idx = np.concatenate([members[j] for j in chosen])
            means = _means_from_rows(rows[pooled_idx])
            occ, has_tie = _fractional_occupancy(means)
            occupancy_sum += occ
            if has_tie:
                tie_resamples += 1
            for slot, model_idx in enumerate(observed_order):
                retention_sum[slot] += occ[int(model_idx), slot]
            if _strict_order_matches(means, observed_order):
                full_order_count += 1

    inv = 1.0 / float(n_resamples)
    occupancy = occupancy_sum * inv
    position_retention = retention_sum * inv
    top1_retention = float(position_retention[0])
    full_order_retention = float(full_order_count) * inv

    return RankBootstrapResult(
        occupancy=occupancy,
        position_retention=position_retention,
        top1_retention=top1_retention,
        full_order_retention=full_order_retention,
        tie_resamples=int(tie_resamples),
        n_resamples=int(n_resamples),
        n_resampling_units=int(n_units),
        clustered=clustered,
    )
