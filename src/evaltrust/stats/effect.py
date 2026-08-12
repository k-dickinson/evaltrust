"""Effect size for a paired comparison.

A p-value says whether a difference is real; an effect size says whether it is
big enough to care about. EvalTrust reports both so nobody ships on significance
alone.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _sp


def cohens_d_paired(differences: np.ndarray) -> float:
    """Cohen's d for paired differences: mean(diff) / sd(diff).

    Zero difference gives 0.0; a consistent nonzero difference with zero spread
    gives +/-inf (sign preserved).
    """
    diffs = np.asarray(differences, dtype=float)
    if diffs.size == 0:
        raise ValueError("cohens_d_paired requires at least one difference")

    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0

    if sd == 0.0:
        if mean == 0.0:
            return 0.0
        return float(np.inf) if mean > 0 else float(-np.inf)
    return mean / sd


def cohens_d_paired_along_rows(matrix: np.ndarray) -> np.ndarray:
    """Cohen's d for each row of a 2-D array of paired differences.

    A vectorized companion to :func:`cohens_d_paired`, used to compute Cohen's d
    across many bootstrap resamples at once. Each row is reduced to
    ``mean / sd(ddof=1)`` with the *same* degenerate handling as the scalar
    version: zero spread yields ``0.0`` when the mean is 0, else ``+/-inf``
    (the sign preserved). It never returns ``NaN``.
    """
    m = np.asarray(matrix, dtype=float)
    mean = m.mean(axis=-1)
    n = m.shape[-1]
    sd = m.std(axis=-1, ddof=1) if n > 1 else np.zeros_like(mean)

    with np.errstate(divide="ignore", invalid="ignore"):
        d = mean / sd
        # Where the spread is zero, mirror cohens_d_paired: 0/0 -> 0.0, else
        # +/-inf. (The 0*inf here is masked out by the outer where.)
        degenerate = np.where(mean == 0.0, 0.0, np.sign(mean) * np.inf)
        return np.where(sd == 0.0, degenerate, d)


def probability_of_superiority_paired(differences: np.ndarray) -> float:
    """Dependent-groups probability of superiority for paired differences.

    Returns ``P(D > 0) + 0.5 * P(D = 0)``. A value of ``0.5`` means neither
    model has an advantage across the paired examples.
    """
    diffs = np.asarray(differences, dtype=float)
    if diffs.size == 0:
        raise ValueError(
            "probability_of_superiority_paired requires at least one difference"
        )

    wins = int(np.count_nonzero(diffs > 0.0))
    ties = int(np.count_nonzero(diffs == 0.0))
    return float((wins + 0.5 * ties) / diffs.size)


def rank_biserial_paired(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for paired differences.

    Zero differences are dropped. Absolute nonzero differences receive
    midranks, matching the Wilcoxon signed-rank convention. The result is the
    positive-rank sum minus the negative-rank sum, divided by their total.
    """
    diffs = np.asarray(differences, dtype=float)
    if diffs.size == 0:
        raise ValueError("rank_biserial_paired requires at least one difference")
    return float(rank_biserial_paired_along_rows(diffs[None, :])[0])


def rank_biserial_paired_along_rows(matrix: np.ndarray) -> np.ndarray:
    """Matched-pairs rank-biserial correlation for each row of differences.

    This vectorized companion to :func:`rank_biserial_paired` is used for
    bootstrap resamples. Zeros are dropped per row and ties receive midranks.
    An all-zero row is defined as ``0.0``. The result never contains ``NaN``.
    """
    m = np.asarray(matrix, dtype=float)
    if m.size == 0 or m.shape[-1] == 0:
        raise ValueError(
            "rank_biserial_paired_along_rows requires at least one difference"
        )

    absolute_nonzero = np.where(m != 0.0, np.abs(m), np.nan)
    ranks = _sp.rankdata(
        absolute_nonzero,
        method="average",
        axis=-1,
        nan_policy="omit",
    )
    positive = np.where(m > 0.0, ranks, 0.0).sum(axis=-1)
    negative = np.where(m < 0.0, ranks, 0.0).sum(axis=-1)
    total = positive + negative
    return np.divide(
        positive - negative,
        total,
        out=np.zeros_like(total, dtype=float),
        where=total != 0.0,
    )


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size between two proportions.

    The right effect size for pass-rate / accuracy comparisons, where Cohen's d
    (which assumes continuous data) does not fit. Uses the arcsine-square-root
    transform; positive when ``p1 > p2``.
    """
    phi1 = 2.0 * np.arcsin(np.sqrt(p1))
    phi2 = 2.0 * np.arcsin(np.sqrt(p2))
    return float(phi1 - phi2)


def magnitude_label(d: float) -> str:
    """Map an effect size to a plain-language magnitude (sign-agnostic).

    Uses Cohen's conventional thresholds: <0.2 negligible, <0.5 small,
    <0.8 medium, >=0.8 large.
    """
    m = abs(d)
    if m < 0.2:
        return "negligible"
    if m < 0.5:
        return "small"
    if m < 0.8:
        return "medium"
    return "large"


def magnitude_label_rank_r(r: float) -> str:
    """Map an r-family effect size to a sign-agnostic magnitude label.

    Uses Cohen's r-family conventions from *Statistical Power Analysis for the
    Behavioral Sciences*, 2nd ed. (1988): 0.1, 0.3, and 0.5.
    """
    magnitude = abs(r)
    if magnitude < 0.1:
        return "negligible"
    if magnitude < 0.3:
        return "small"
    if magnitude < 0.5:
        return "medium"
    return "large"
