"""Benchmark Health audit.

Even a flawless comparison is worthless on a broken benchmark. Two failure modes
matter most in practice:

  - Saturation: everyone already scores near the ceiling, so there is no room
    left to demonstrate an improvement.
  - No discrimination: the benchmark gives essentially the same score to
    everything, so it cannot separate any two models regardless of the stats.
"""

from __future__ import annotations

import numpy as np

from ..core.schema import EvalData, Finding, Status

PILLAR = "Benchmark Health"

SATURATION_FRACTION = 0.95   # mean >= 95% of the ceiling counts as saturated
MIN_SPREAD = 0.01            # pooled std below this = no discriminating signal


def audit_benchmark_health(
    data: EvalData, models: list[str] | None = None
) -> list[Finding]:
    models = models or data.models
    per_model = {
        m: np.array([ex.scores[m] for ex in data.examples if m in ex.scores],
                    dtype=float)
        for m in models
    }
    pooled = np.concatenate([v for v in per_model.values() if v.size])

    return [
        _saturation(per_model, pooled),
        _discrimination(pooled),
    ]


def _saturation(per_model, pooled) -> Finding:
    ceiling = float(pooled.max())
    top_mean = max(float(v.mean()) for v in per_model.values() if v.size)
    frac = (top_mean / ceiling) if ceiling > 0 else 0.0
    saturated = ceiling > 0 and frac >= SATURATION_FRACTION

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
            "Move to a harder or fresher benchmark with more headroom; gains "
            "measured at the ceiling rarely transfer."
            if saturated else
            "There is room to distinguish models on this benchmark."
        ),
        details={"check": "saturation", "ceiling": ceiling,
                 "top_mean": top_mean, "fraction_of_ceiling": frac,
                 "saturated": saturated},
    )


def _discrimination(pooled) -> Finding:
    spread = float(pooled.std())
    discriminating = spread >= MIN_SPREAD

    return Finding(
        pillar=PILLAR,
        title=("Benchmark discriminates between examples" if discriminating
               else "Benchmark shows almost no variation"),
        status=Status.PASS if discriminating else Status.WARN,
        why=(
            "If a benchmark assigns nearly the same score to everything, it "
            "carries no signal to separate one model from another — any ranking "
            "it produces is essentially arbitrary."
        ),
        how_detected=(
            f"The pooled standard deviation of scores was {spread:.4f} "
            f"(threshold {MIN_SPREAD})."
        ),
        how_to_fix=(
            "The benchmark produces a healthy spread of scores."
            if discriminating else
            "Add harder or more varied examples so the benchmark can actually "
            "separate models; a flat score distribution can't rank anything."
        ),
        details={"check": "discrimination", "pooled_std": spread,
                 "discriminating": discriminating},
    )
