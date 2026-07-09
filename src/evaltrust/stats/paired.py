"""Paired hypothesis tests for binary (pass/fail) outcomes.

For paired binary data, McNemar's test looks only at the discordant pairs
(examples where the two models disagree) and asks whether the disagreements split
evenly. That is the right question for accuracy comparisons.
"""

from __future__ import annotations

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
