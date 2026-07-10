"""Multiple-comparison corrections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HolmResult:
    reject: list[bool]
    adjusted_pvalues: list[float]
    thresholds: list[float]


def holm_bonferroni(p_values, alpha: float = 0.05) -> HolmResult:
    """Holm-Bonferroni step-down correction.

    Returns rejections, adjusted p-values, and the per-hypothesis alpha threshold
    in the original p-value order.

    Rejection is a *step-down* decision, not a per-hypothesis one: once a
    hypothesis fails its threshold, every larger p-value is retained even if it
    would clear its own (looser) threshold. Callers must therefore use ``reject``
    rather than re-deriving it from ``thresholds``.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(p)):
        raise ValueError("p_values must all be finite (no NaN or infinity)")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha!r}")

    m = p.size
    # Stable so tied p-values keep their input order: the reported per-hypothesis
    # thresholds are otherwise assigned arbitrarily within a tie group.
    order = np.argsort(p, kind="stable")
    thresholds = np.empty(m, dtype=float)
    reject = np.zeros(m, dtype=bool)
    adjusted_sorted = np.empty(m, dtype=float)

    still_rejecting = True
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        threshold = alpha / factor
        thresholds[idx] = threshold

        adjusted = min(1.0, p[idx] * factor)
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = min(1.0, running_max)

        if still_rejecting and p[idx] < threshold:
            reject[idx] = True
        else:
            still_rejecting = False

    adjusted_pvalues = np.empty(m, dtype=float)
    adjusted_pvalues[order] = adjusted_sorted
    return HolmResult(
        reject=[bool(x) for x in reject],
        adjusted_pvalues=[float(x) for x in adjusted_pvalues],
        thresholds=[float(x) for x in thresholds],
    )
