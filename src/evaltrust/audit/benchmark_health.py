"""Benchmark Health audit.

A comparison is worthless on a broken benchmark. Flags saturation (everyone near
the ceiling, no room to improve) and no discrimination (near-identical scores,
can't separate models).
"""

from __future__ import annotations

import numpy as np

from ..core.schema import EvalData, Finding, Status

PILLAR = "Benchmark Health"

SATURATION_FRACTION = 0.95   # mean >= 95% of the ceiling counts as saturated
MIN_SPREAD = 0.01            # pooled std below this = no discriminating signal


def audit_benchmark_health(
    data: EvalData,
    models: list[str] | None = None,
    saturation_fraction: float = SATURATION_FRACTION,
    min_spread: float = MIN_SPREAD,
) -> list[Finding]:
    models = models or data.models
    per_model = {
        m: np.array([ex.scores[m] for ex in data.examples if m in ex.scores],
                    dtype=float)
        for m in models
    }
    pooled = np.concatenate([v for v in per_model.values() if v.size])

    return [
        _saturation(per_model, pooled, saturation_fraction),
        _discrimination(pooled, min_spread),
    ]


def _saturation(per_model, pooled, saturation_fraction) -> Finding:
    ceiling = float(pooled.max())
    top_mean = max(float(v.mean()) for v in per_model.values() if v.size)
    frac = (top_mean / ceiling) if ceiling > 0 else 0.0
    saturated = ceiling > 0 and frac >= saturation_fraction

    return Finding(
        pillar=PILLAR,
        title="Benchmark is saturated" if saturated else "Benchmark has headroom",
        status=Status.WARN if saturated else Status.PASS,
        why=(
            "When the best model already scores near the maximum, there is almost "
            "no room left to show improvement, and small gaps near the ceiling are "
            "dominated by noise and label errors."
        ),
        how_detected=(
            f"The strongest model averaged {top_mean:.3f} against a ceiling of "
            f"{ceiling:.3f} ({frac:.0%} of maximum)."
        ),
        how_to_fix=(
            "Switch to a harder benchmark. Gains at the ceiling rarely transfer."
            if saturated else
            "There is room to distinguish models on this benchmark."
        ),
        details={"check": "saturation", "ceiling": ceiling,
                 "top_mean": top_mean, "fraction_of_ceiling": frac,
                 "saturated": saturated},
    )


def _discrimination(pooled, min_spread) -> Finding:
    spread = float(pooled.std())
    discriminating = spread >= min_spread

    return Finding(
        pillar=PILLAR,
        title=("Benchmark discriminates between examples" if discriminating
               else "Benchmark shows almost no variation"),
        status=Status.PASS if discriminating else Status.WARN,
        why=(
            "If a benchmark assigns nearly the same score to everything, it "
            "carries no signal to separate one model from another. Any ranking "
            "it produces is basically arbitrary."
        ),
        how_detected=(
            f"The pooled standard deviation of scores was {spread:.4f} "
            f"(threshold {min_spread})."
        ),
        how_to_fix=(
            "The benchmark produces a healthy spread of scores."
            if discriminating else
            "Add harder, more varied examples. A flat score spread can't rank models."
        ),
        details={"check": "discrimination", "pooled_std": spread,
                 "discriminating": discriminating},
    )
