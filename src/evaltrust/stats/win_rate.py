"""All-example paired win rate with a seeded percentile bootstrap interval."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .resampling import _chunk_rows


_METHOD = "half-tie-percentile-bootstrap-v1"


@dataclass(frozen=True)
class PairedWinRateResult:
    """Plain-scalar result for the all-example paired win-rate estimand."""

    win_rate_a: float
    interval_low: float
    interval_high: float
    confidence: float
    n_examples: int
    n_wins_a: int
    n_ties: int
    n_wins_b: int
    method: Literal["half-tie-percentile-bootstrap-v1"]


def paired_win_rate(
    differences: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
    clusters: Sequence[Hashable] | None = None,
) -> PairedWinRateResult:
    """Estimate how often A scores higher than B on paired examples.

    ``differences`` follows EvalTrust's ``score_B - score_A`` convention, so a
    negative value is an A win. Exact zero is a tie and receives half credit;
    no numerical tolerance is applied. The interval is a seeded percentile
    bootstrap over examples. When ``clusters`` is provided, whole clusters are
    sampled with replacement and all rows from each selected cluster are pooled.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired_win_rate requires at least one 1-D difference")
    if not bool(np.isfinite(values).all()):
        raise ValueError("paired_win_rate requires finite differences")
    if (
        isinstance(confidence, (bool, np.bool_))
        or not np.isscalar(confidence)
        or not np.isfinite(confidence)
        or confidence <= 0
        or confidence >= 1
    ):
        raise ValueError("confidence must be between 0 and 1")
    if (
        isinstance(n_resamples, (bool, np.bool_))
        or not isinstance(n_resamples, (int, np.integer))
        or n_resamples < 1
    ):
        raise ValueError("n_resamples must be a positive integer")

    confidence = float(confidence)
    n_resamples = int(n_resamples)
    n = int(values.size)
    n_wins_a = int(np.count_nonzero(values < 0.0))
    n_ties = int(np.count_nonzero(values == 0.0))
    n_wins_b = int(np.count_nonzero(values > 0.0))
    per_example_credit = np.where(
        values < 0.0,
        1.0,
        np.where(values == 0.0, 0.5, 0.0),
    )
    win_rate_a = float(per_example_credit.mean())

    members: list[list[int]]
    if clusters is None:
        members = [[index] for index in range(n)]
    else:
        try:
            labels = list(clusters)
        except TypeError as exc:
            raise ValueError("clusters must be a sequence of hashable labels") from exc
        if len(labels) != n:
            raise ValueError("clusters must have the same length as differences")
        grouped: dict[Hashable, list[int]] = {}
        for index, label in enumerate(labels):
            try:
                hash(label)
            except TypeError as exc:
                raise ValueError("cluster labels must be hashable") from exc
            grouped.setdefault(label, []).append(index)
        members = list(grouped.values())

    cluster_credits = np.array(
        [float(per_example_credit[indices].sum()) for indices in members],
        dtype=float,
    )
    cluster_sizes = np.array(
        [len(indices) for indices in members],
        dtype=float,
    )
    k = len(members)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)
    rows = _chunk_rows(k, n_resamples)
    position = 0
    while position < n_resamples:
        block = min(rows, n_resamples - position)
        chosen = rng.integers(0, k, size=(block, k))
        estimates[position:position + block] = (
            cluster_credits[chosen].sum(axis=1)
            / cluster_sizes[chosen].sum(axis=1)
        )
        position += block

    alpha = 1.0 - confidence
    interval_low = float(np.percentile(estimates, 100.0 * alpha / 2.0))
    interval_high = float(
        np.percentile(estimates, 100.0 * (1.0 - alpha / 2.0))
    )
    return PairedWinRateResult(
        win_rate_a=win_rate_a,
        interval_low=interval_low,
        interval_high=interval_high,
        confidence=confidence,
        n_examples=n,
        n_wins_a=n_wins_a,
        n_ties=n_ties,
        n_wins_b=n_wins_b,
        method=_METHOD,
    )
