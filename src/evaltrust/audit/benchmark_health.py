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
SCALE_MAX_RATIO = 20.0       # metric maxima this far apart suggest mixed scales
SCALE_MASS_FRACTION = 0.80   # dominant unit/percent-shaped mass for one metric
SCALE_UNIT_MAX = 1.0
SCALE_PERCENT_MIN = 1.5
SCALE_PERCENT_MAX = 100.0


def audit_benchmark_health(
    data: EvalData,
    models: list[str] | None = None,
    saturation_fraction: float = SATURATION_FRACTION,
    min_spread: float = MIN_SPREAD,
    score_ceiling: float | None = None,
    observed_ranges: dict[str, dict[str, float | int]] | None = None,
) -> list[Finding]:
    models = models or data.models
    per_model = {
        m: np.array([ex.scores[m] for ex in data.examples if m in ex.scores],
                    dtype=float)
        for m in models
    }
    pooled = np.concatenate([v for v in per_model.values() if v.size])

    return [
        _saturation(per_model, pooled, saturation_fraction, score_ceiling),
        _discrimination(pooled, min_spread),
        _scale_sanity(pooled, observed_ranges),
    ]


def _saturation(per_model, pooled, saturation_fraction, score_ceiling=None) -> Finding:
    observed_max = float(pooled.max())
    ceiling_is_configured = score_ceiling is not None
    ceiling = float(score_ceiling) if ceiling_is_configured else observed_max
    top_mean = max(float(v.mean()) for v in per_model.values() if v.size)
    frac = (top_mean / ceiling) if ceiling > 0 else 0.0
    display_frac = min(frac,1.0) if ceiling_is_configured else frac
    saturated = ceiling > 0 and frac >= saturation_fraction

    ceiling_source = "configured" if ceiling_is_configured else "observed"
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
            f"The strongest model averaged {top_mean:.3f} against a {ceiling_source} "
            f"ceiling of {ceiling:.3f} ({display_frac:.0%} of maximum)."
        ),
        how_to_fix=(
            "Switch to a harder benchmark. Gains at the ceiling rarely transfer."
            if saturated else
            "There is room to distinguish models on this benchmark."
        ),
        details={"check": "saturation", "ceiling": ceiling,
                 "ceiling_source": ceiling_source, "observed_max": observed_max,
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


def _observed_range(values) -> dict[str, float | int] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "n": int(finite.size),
    }


def _scale_sanity(pooled, observed_ranges=None) -> Finding:
    current_range = _observed_range(pooled)
    if observed_ranges is None:
        ranges = {"score": current_range} if current_range is not None else {}
    else:
        ranges = {
            metric: {
                "min": float(bounds["min"]),
                "max": float(bounds["max"]),
                "n": int(bounds["n"]),
            }
            for metric, bounds in observed_ranges.items()
            if int(bounds.get("n", 0)) > 0
            and np.isfinite(float(bounds["min"]))
            and np.isfinite(float(bounds["max"]))
        }

    finite = np.asarray(pooled, dtype=float)
    finite = finite[np.isfinite(finite)]
    trigger_reason = None
    ratio = None

    positive_maxima = [
        (metric, float(bounds["max"]))
        for metric, bounds in ranges.items()
        if float(bounds["max"]) > 0
    ]
    if len(positive_maxima) >= 2:
        smallest = min(value for _, value in positive_maxima)
        largest = max(value for _, value in positive_maxima)
        ratio = largest / smallest
        if ratio >= SCALE_MAX_RATIO:
            trigger_reason = "metric_maxima_ratio"

    if trigger_reason is None and finite.size >= 2:
        fractional_unit_count = int(np.count_nonzero(
            (finite > 0) & (finite < SCALE_UNIT_MAX)
        ))
        percent_count = int(np.count_nonzero(
            (finite > SCALE_PERCENT_MIN) & (finite <= SCALE_PERCENT_MAX)
        ))
        fractional_unit_fraction = fractional_unit_count / finite.size
        percent_fraction = percent_count / finite.size

        if percent_count and fractional_unit_fraction >= SCALE_MASS_FRACTION:
            trigger_reason = "mostly_unit_with_large_values"
        elif (
            fractional_unit_count >= 2
            and percent_fraction >= SCALE_MASS_FRACTION
            and float(finite.max()) >= SCALE_MAX_RATIO
        ):
            trigger_reason = "mostly_percent_with_unit_values"

    checkable = len(ranges) >= 2 or finite.size >= 2
    if not checkable:
        status = Status.SKIP
        trigger_reason = "insufficient_data"
    else:
        status = Status.WARN if trigger_reason is not None else Status.PASS

    range_text = ", ".join(
        f"{metric} [{bounds['min']:.3g}, {bounds['max']:.3g}]"
        for metric, bounds in ranges.items()
    ) or "none"
    rule = (
        "Warn when positive metric maxima differ by at least 20x, or when at "
        "least 80% of one metric is strictly between 0 and 1 or between 1.5 "
        "and 100, with values from the other range also present."
    )

    return Finding(
        pillar=PILLAR,
        title=(
            "Scores span an unexpected range"
            if status is Status.WARN else
            "Score scales look consistent"
            if status is Status.PASS else
            "Score scale not assessed"
        ),
        status=status,
        why=(
            "Mixed or unexpected score scales make saturation and comparisons "
            "misleading because the same numeric gap can mean different things."
        ),
        how_detected=f"{rule} Observed ranges: {range_text}.",
        how_to_fix=(
            "Normalize scores to a shared scale before comparing models. Set "
            "score_ceiling separately when saturation has a known upper bound."
            if status is Status.WARN else
            "Keep score scales consistent and set score_ceiling when a rubric "
            "has a fixed upper bound."
            if status is Status.PASS else
            "Provide at least two scores, or multiple scored metrics, to check "
            "whether scales are consistent."
        ),
        details={
            "check": "scale_sanity",
            "observed_ranges": ranges,
            "trigger_reason": trigger_reason,
            "maxima_ratio": ratio,
        },
    )
