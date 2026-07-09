"""Multiplicity corrections for testing several hypotheses at once.

Testing ``k`` metrics at ``alpha`` inflates the false-positive rate. Bonferroni
(``alpha / k``) is the blunt fix; Holm-Bonferroni is a more powerful step-down.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def holm_bonferroni(
    pvalues: Sequence[float], alpha: float
) -> tuple[list[bool], list[float]]:
    """Holm-Bonferroni step-down correction.

    Returns, in input order, the reject/keep flags and Holm-adjusted p-values.
    Matches ``statsmodels.stats.multitest.multipletests(..., method="holm")``.
    """
    p = np.asarray(pvalues, dtype=float)
    k = p.size
    if k == 0:
        return [], []

    order = np.argsort(p, kind="stable")           # indices of ascending p
    sorted_p = p[order]
    multipliers = k - np.arange(k)                 # k, k-1, ..., 1

    # Holm-adjusted p-values: (k - i) * p_(i), made monotone non-decreasing then
    # clipped into [0, 1]. Rejection is adjusted_p <= alpha.
    adjusted_sorted = np.maximum.accumulate(multipliers * sorted_p)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    rejected_sorted = adjusted_sorted <= alpha

    rejected = np.empty(k, dtype=bool)
    adjusted = np.empty(k, dtype=float)
    rejected[order] = rejected_sorted
    adjusted[order] = adjusted_sorted

    return [bool(r) for r in rejected], [float(a) for a in adjusted]
