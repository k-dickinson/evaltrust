"""Effect size for a paired comparison.

A p-value says whether a difference is real; an effect size says whether it is
big enough to care about. EvalTrust reports both so nobody ships on significance
alone.
"""

from __future__ import annotations

import numpy as np


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
