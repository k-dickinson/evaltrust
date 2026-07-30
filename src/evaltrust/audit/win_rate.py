"""Interpret the all-example paired win rate as an advisory finding."""

from __future__ import annotations

from ..core.schema import EvalData, Finding, Status
from ..stats.win_rate import paired_win_rate

PILLAR = "Statistical Validity"
CHECK = "paired_win_rate"
METHOD = "half-tie-percentile-bootstrap-v1"
CONFIDENCE = 0.95
TIE_POLICY = "half_credit"
ESTIMAND = "all_example_paired_win_rate_a"


def _details(
    *,
    model_a: str,
    model_b: str,
    seed: int,
    clustered: bool,
    n_resampling_units: int,
    assessed: bool,
    paired_win_rate_a: float | None,
    interval_low: float | None,
    interval_high: float | None,
    n_examples: int,
    n_wins_a: int | None,
    n_ties: int | None,
    n_wins_b: int | None,
) -> dict:
    return {
        "check": CHECK,
        "assessed": assessed,
        "model_a": model_a,
        "model_b": model_b,
        "paired_win_rate_a": paired_win_rate_a,
        "win_rate_interval_low": interval_low,
        "win_rate_interval_high": interval_high,
        "confidence": CONFIDENCE,
        "n_examples": n_examples,
        "n_wins_a": n_wins_a,
        "n_ties": n_ties,
        "n_wins_b": n_wins_b,
        "tie_policy": TIE_POLICY,
        "estimand": ESTIMAND,
        "method": METHOD,
        "seed": int(seed),
        "clustered": clustered,
        "n_resampling_units": n_resampling_units,
    }


def _skip(
    *,
    model_a: str,
    model_b: str,
    seed: int,
    reason: str,
    detected: str,
    n_examples: int,
    clustered: bool = False,
    n_resampling_units: int = 0,
    calculation_error: str | None = None,
) -> Finding:
    details = _details(
        model_a=model_a,
        model_b=model_b,
        seed=seed,
        clustered=clustered,
        n_resampling_units=n_resampling_units,
        assessed=False,
        paired_win_rate_a=None,
        interval_low=None,
        interval_high=None,
        n_examples=n_examples,
        n_wins_a=None,
        n_ties=None,
        n_wins_b=None,
    )
    details["reason"] = reason
    if calculation_error is not None:
        details["calculation_error"] = calculation_error
    return Finding(
        pillar=PILLAR,
        title=(
            f"Win rate: {model_a} scores higher than {model_b} not assessed "
            f"(ties half credit; {n_examples} examples; 95% bootstrap interval "
            "unavailable)"
        ),
        status=Status.SKIP,
        why=(
            "The mean score gap does not show how often the first model scores "
            "higher on individual examples."
        ),
        how_detected=detected,
        how_to_fix=(
            "Add finite paired scores for both models on the same examples, then "
            "read this all-example rate and interval as an advisory view."
        ),
        details=details,
    )


def audit_win_rate(
    data: EvalData,
    model_a: str,
    model_b: str,
    *,
    seed: int = 0,
) -> list[Finding]:
    """Report one advisory all-example paired win-rate finding."""
    differences = data.differences(model_a, model_b)
    paired_examples = [
        (index, example)
        for index, example in enumerate(data.examples)
        if model_a in example.scores and model_b in example.scores
    ]
    if differences.size == 0:
        preference_only = data.has_preferences and not any(
            example.scores for example in data.examples
        )
        reason = "preference_only" if preference_only else "no_paired_scores"
        detected = (
            "Preference-only data contains no paired per-model scores; 0 examples "
            "were eligible for the all-example paired win rate."
            if preference_only
            else "No examples contain scores for both selected models; 0 paired "
            "examples were eligible."
        )
        return [_skip(
            model_a=model_a,
            model_b=model_b,
            seed=seed,
            reason=reason,
            detected=detected,
            n_examples=0,
        )]

    clustered = any(
        example.group_id is not None for _, example in paired_examples
    )
    clusters = None
    n_resampling_units = 0 if clustered else int(differences.size)
    if clustered:
        clusters = [
            ("group", example.group_id)
            if example.group_id is not None
            else ("example", index)
            for index, example in paired_examples
        ]

    try:
        result = paired_win_rate(
            differences,
            seed=seed,
            clusters=clusters,
        )
        if clustered:
            n_resampling_units = len(dict.fromkeys(clusters))
    except (FloatingPointError, OverflowError, TypeError, ValueError) as exc:
        return [_skip(
            model_a=model_a,
            model_b=model_b,
            seed=seed,
            reason="calculation_error",
            detected=(
                f"The {differences.size} paired score differences could not be "
                f"evaluated: {exc}."
            ),
            n_examples=int(differences.size),
            clustered=clustered,
            n_resampling_units=n_resampling_units,
            calculation_error=str(exc),
        )]

    confidence_label = f"{result.confidence:.0%}"
    resampling = (
        f"The interval resampled {n_resampling_units} whole group_id clusters "
        "with replacement and pooled the selected examples."
        if clustered
        else f"The interval resampled {n_resampling_units} examples with replacement."
    )
    return [Finding(
        pillar=PILLAR,
        title=(
            f"Win rate: {model_a} scores higher than {model_b} on "
            f"{result.win_rate_a:.1%} of examples (ties half credit; "
            f"{result.n_examples} examples; {confidence_label} bootstrap interval "
            f"[{result.interval_low:.1%}, {result.interval_high:.1%}])"
        ),
        status=Status.PASS,
        why=(
            "The mean score gap does not show how often the first model scores "
            "higher on individual examples."
        ),
        how_detected=(
            f"Across {result.n_examples} paired examples, {model_a} scored higher "
            f"on {result.n_wins_a}, tied exactly on {result.n_ties}, and scored "
            f"lower on {result.n_wins_b}. Exact ties received half credit. "
            f"{resampling}"
        ),
        how_to_fix=(
            "Read this all-example paired win rate and bootstrap interval as an "
            "advisory view, not as the decisive-pair or future-rerun probability."
        ),
        details=_details(
            model_a=model_a,
            model_b=model_b,
            seed=seed,
            clustered=clustered,
            n_resampling_units=n_resampling_units,
            assessed=True,
            paired_win_rate_a=result.win_rate_a,
            interval_low=result.interval_low,
            interval_high=result.interval_high,
            n_examples=result.n_examples,
            n_wins_a=result.n_wins_a,
            n_ties=result.n_ties,
            n_wins_b=result.n_wins_b,
        ),
    )]
