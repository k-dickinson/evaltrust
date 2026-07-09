"""Audit a single model's evaluation.

With one model the question is "can I trust this score?": a bootstrap CI on the
mean (precision), an optional check of that CI against a target (threshold), plus
benchmark health.
"""

from __future__ import annotations

import numpy as np

from ..config import AuditConfig
from ..core.schema import EvalData, Finding, Status
from ..stats.resampling import bootstrap_ci
from .benchmark_health import audit_benchmark_health
from .repeatability import audit_single_repeatability

PILLAR = "Score Reliability"


def audit_single(
    data: EvalData,
    model: str,
    threshold: float | None = None,
    config: AuditConfig | None = None,
) -> list[Finding]:
    cfg = config or AuditConfig()
    scores = np.array([ex.scores[model] for ex in data.examples if model in ex.scores],
                      dtype=float)
    if scores.size == 0:
        raise ValueError(f"Model '{model}' has no scores to audit.")

    mean = float(scores.mean())
    lo, hi = bootstrap_ci(scores, confidence=0.95,
                          n_resamples=cfg.n_resamples, seed=cfg.seed)

    findings = [_precision(mean, lo, hi, scores.size, cfg.precision_margin)]
    if threshold is not None:
        findings.append(_threshold(mean, lo, hi, threshold, model))
    findings += audit_benchmark_health(
        data, [model], saturation_fraction=cfg.saturation_fraction,
        min_spread=cfg.min_spread)
    findings += audit_single_repeatability(data, model)
    return findings


def _precision(mean, lo, hi, n, margin) -> Finding:
    half_width = (hi - lo) / 2.0
    precise = half_width <= margin
    # CI half-width shrinks like 1/sqrt(n); estimate the n needed for the margin.
    need_n = int(np.ceil(n * (half_width / margin) ** 2)) if margin > 0 and half_width > 0 else n
    extra = max(0, need_n - n)

    return Finding(
        pillar=PILLAR,
        title=("Score is measured precisely" if precise
               else "Score is imprecise (too few examples)"),
        status=Status.PASS if precise else Status.WARN,
        why=(
            "A single score is only as trustworthy as it is precise. With few "
            "examples the true score could be far from the number you see, so "
            "decisions made on it are really decisions made on noise."
        ),
        how_detected=(
            f"Over {n} examples the score was {mean:.1%}, 95% CI "
            f"[{lo:.1%}, {hi:.1%}] (+/-{half_width:.1%})."
        ),
        how_to_fix=(
            "The score is pinned down tightly enough to rely on."
            if precise else
            f"Collect about {extra} more examples (~{need_n} total) to tighten "
            f"the interval to +/-{margin:.0%}."
        ),
        details={"check": "single_precision", "mean": mean, "ci_low": lo,
                 "ci_high": hi, "half_width": half_width, "n": n,
                 "precise": precise},
    )


def _threshold(mean, lo, hi, threshold, model) -> Finding:
    if lo > threshold:
        outcome, status = "above", Status.PASS
    elif hi < threshold:
        outcome, status = "below", Status.FAIL
    else:
        outcome, status = "inconclusive", Status.WARN

    verdict_text = {
        "above": f"clears the {threshold:.0%} target",
        "below": f"is below the {threshold:.0%} target",
        "inconclusive": f"can't be confirmed above the {threshold:.0%} target",
    }[outcome]

    return Finding(
        pillar=PILLAR,
        title=f"{model} {verdict_text}",
        status=status,
        why=(
            "Hitting a target on average isn't the same as clearing it reliably. "
            "If the confidence interval straddles the bar, the model might really "
            "be below it."
        ),
        how_detected=(
            f"Score {mean:.1%}, 95% CI [{lo:.1%}, {hi:.1%}], against a "
            f"{threshold:.0%} target."
        ),
        how_to_fix={
            "above": "The model clears the bar with room to spare.",
            "below": "The model is below the bar; it isn't ready against this target.",
            "inconclusive": ("Too close to call. Collect more examples to move the "
                             "interval fully above or below the target."),
        }[outcome],
        details={"check": "threshold", "outcome": outcome, "threshold": threshold,
                 "mean": mean, "ci_low": lo, "ci_high": hi},
    )
