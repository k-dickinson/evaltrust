"""Effect size for a paired comparison.

A p-value tells you whether a difference is real; an effect size tells you
whether it is *big enough to care about*. A tiny, perfectly-measured difference
can be statistically significant yet practically meaningless — EvalLab reports
both so nobody ships on significance alone.
"""

from __future__ import annotations

import numpy as np


def cohens_d_paired(differences: np.ndarray) -> float:
    """Cohen's d for paired differences: mean(diff) / sd(diff).

    - No difference at all -> 0.0 (no effect).
    - A consistent nonzero difference with zero spread -> +/-inf (an infinitely
      reliable effect; the sign is preserved).
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
