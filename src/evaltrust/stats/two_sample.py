"""Unpaired two-sample statistics for run-level (aggregate) score comparison.

When a harness emits only one total score per run — rather than per-example
scores — pairing on examples is impossible.  The correct analysis is a proper
two-sample comparison of the two models' run-score distributions.

Two estimators are provided:

``bootstrap_p_a_gt_b``
    Bootstrap estimate of P(A > B) — the probability that a randomly drawn run
    of model A scores higher than a randomly drawn run of model B — with a
    percentile confidence interval.  This is the primary estimator: it gives a
    direction-aware probability on [0, 1] that a practitioner can interpret
    directly.

``mann_whitney_u``
    The Mann-Whitney U / Wilcoxon rank-sum test.  A nonparametric rank test
    of whether one sample tends to produce higher values than the other.
    Returns the two-sided p-value.  Used as a complementary significance check.

Neither estimator requires the two samples to have the same length (N runs per
model may differ across harnesses), and neither assumes normality.

Design note: P(A > B) is estimated by bootstrap rather than by the exact U-
statistic formula (U / (n_a * n_b)) because the bootstrap additionally gives a
CI without extra assumptions and scales identically to the rest of the
EvalTrust resampling infrastructure.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _sp


def bootstrap_p_a_gt_b(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap estimate of P(A > B) with a percentile CI.

    P(A > B) is the probability that a randomly drawn score from model A
    exceeds a randomly drawn score from model B.  Values above 0.5 favour A;
    below 0.5 favour B; 0.5 means the distributions are indistinguishable.

    The point estimate is the empirical fraction of all (a_i, b_j) pairs where
    a_i > b_j (ties split at 0.5, consistent with the Mann-Whitney U relation).
    The CI is a standard percentile bootstrap over resamples of the *pair*
    (one draw from A, one from B) — this resamples both margins independently
    and is valid for unpaired data.

    Parameters
    ----------
    scores_a, scores_b:
        1-D arrays of run-level scores (need not be the same length).
    confidence:
        Coverage for the CI (default 0.95 → 95 %).
    n_resamples:
        Number of bootstrap iterations (default 10 000).
    seed:
        RNG seed for reproducibility (default 0).

    Returns
    -------
    (p_hat, ci_low, ci_high)
        Point estimate and the lower/upper bounds of the ``confidence``-level
        percentile bootstrap CI.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError(
            "bootstrap_p_a_gt_b requires at least one score per model"
        )

    p_hat = _p_a_gt_b(a, b)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        a_resamp = a[rng.integers(0, a.size, size=a.size)]
        b_resamp = b[rng.integers(0, b.size, size=b.size)]
        boot[i] = _p_a_gt_b(a_resamp, b_resamp)

    alpha = 1.0 - confidence
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1.0 - alpha / 2)))
    return float(p_hat), lo, hi


def _p_a_gt_b(a: np.ndarray, b: np.ndarray) -> float:
    """Empirical P(A > B) with ties counted as 0.5.

    Computed in O(n_a * n_b) via broadcasting — appropriate for the typical
    sizes seen in run-level comparisons (tens to low hundreds of runs).
    """
    n_a, n_b = a.size, b.size
    pairs = float(n_a * n_b)
    wins = float(np.sum(a[:, None] > b[None, :]))
    ties = float(np.sum(a[:, None] == b[None, :]))
    return (wins + 0.5 * ties) / pairs


def mann_whitney_u(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test for unpaired two-sample comparison.

    Tests the null hypothesis that a randomly selected value from population A
    is equally likely to be greater or less than a randomly selected value from
    population B (equivalent to P(A > B) = 0.5).

    Parameters
    ----------
    scores_a, scores_b:
        1-D arrays of run-level scores (need not be the same length).

    Returns
    -------
    (u_stat, p_value)
        The U statistic and the two-sided p-value.  When either sample has
        fewer than 3 observations the p-value is set to 1.0 (the test is not
        reliable for very small samples; the bootstrap CI is reported instead).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.size < 3 or b.size < 3:
        # Not enough data for Mann-Whitney to be meaningful; return conservative
        # sentinel so the caller can report it as "insufficient data".
        u_fallback = float(np.sum(a[:, None] > b[None, :]))
        return u_fallback, 1.0
    result = _sp.mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)
