"""Paired hypothesis tests for binary (pass/fail) outcomes.

For paired binary data, McNemar's test looks only at the discordant pairs
(examples where the two models disagree) and asks whether the disagreements split
evenly. That is the right question for accuracy comparisons.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _sp


def mcnemar_exact(b_only: int, a_only: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant-pair counts.

    ``b_only`` counts examples the second model got right and the first wrong;
    ``a_only`` is the reverse. Concordant pairs are ignored. With no discordant
    pairs the p-value is 1. This is a two-sided binomial test of the discordant
    split against 50/50.
    """
    n = b_only + a_only
    if n == 0:
        return 1.0
    result = _sp.binomtest(min(b_only, a_only), n, 0.5, alternative="two-sided")
    return float(result.pvalue)


def paired_p_a_gt_b(differences: np.ndarray) -> float:
    """P(A > B) from paired differences (score_B - score_A).

    Returns the proportion of paired examples where A's score exceeds B's
    score, with ties counted as 0.5.  This is the common-language effect
    size (CLES) or probability of superiority: the probability that a
    randomly chosen example favours A.
    """
    d = np.asarray(differences, dtype=float)
    n = d.size
    if n == 0:
        raise ValueError("paired_p_a_gt_b requires at least one difference")
    a_wins = float(np.sum(d < 0))
    ties = float(np.sum(d == 0))
    return (a_wins + 0.5 * ties) / n
