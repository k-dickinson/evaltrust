"""Eligibility and finding text for fixed-example predictive reruns."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..core.schema import EvalData, Finding, Status
from ..stats.predictive_rerun import predictive_rerun_normal_theory

PILLAR = "Statistical Validity"
CHECK = "predictive_rerun"
METHOD = "welch-satterthwaite-normal-theory-v1"
CENTRAL_MASS = 0.95
COVERAGE_LIMITATION = (
    "Measured approximate-range coverage fell to ~0.933 with 8 examples and "
    "3 observed runs, and ~0.937 with 25 examples and 6 observed runs, for "
    "skewed run streams. It returns toward nominal as examples and runs grow."
)
_REASONS = (
    "missing_runs_a",
    "missing_runs_b",
    "nonfinite_a",
    "nonfinite_b",
    "score_run_mean_disagreement",
    "singleton_in_mixed",
)


def _present_stream(
    data: Sequence[float] | None,
) -> bool:
    if data is None:
        return False
    try:
        return len(data) > 0
    except TypeError:
        return False


def _finite_stream(data: Sequence[float]) -> np.ndarray | None:
    try:
        if any(isinstance(value, (bool, np.bool_)) for value in data):
            return None
        values = np.asarray(data, dtype=float)
    except (TypeError, ValueError):
        return None
    if values.ndim != 1 or values.size == 0:
        return None
    if not bool(np.isfinite(values).all()):
        return None
    if not math.isfinite(float(values.mean())):
        return None
    if values.size > 1 and not math.isfinite(float(values.var(ddof=1))):
        return None
    return values


def _empty_details(
    *,
    model_a: str,
    model_b: str,
    n_total: int,
    n_predictive: int,
    reason_counts: dict[str, int],
    future_runs: int,
) -> dict:
    return {
        "check": CHECK,
        "assessed": False,
        "model_a": model_a,
        "model_b": model_b,
        "n_total": n_total,
        "n_predictive": n_predictive,
        "reason_counts": reason_counts,
        "future_runs": future_runs,
        "method": METHOD,
        "normal_theory_probability_a_better": None,
        "normal_theory_range_low": None,
        "normal_theory_range_high": None,
        "central_mass": CENTRAL_MASS,
        "degenerate_reason": None,
        "point_estimate_b_minus_a": None,
        "prediction_variance": None,
        "degrees_of_freedom": None,
        "coverage_limitation": COVERAGE_LIMITATION,
    }


def _prefix(
    *,
    future_runs: int,
    model_a: str,
    model_b: str,
) -> str:
    return (
        "P(A better on a rerun of these examples at "
        f"{future_runs} future runs) "
        f"(A={model_a}, B={model_b})"
    )


def _skip_finding(
    *,
    model_a: str,
    model_b: str,
    n_total: int,
    n_predictive: int,
    reason_counts: dict[str, int],
    future_runs: int,
    calculation_error: str | None = None,
) -> Finding:
    details = _empty_details(
        model_a=model_a,
        model_b=model_b,
        n_total=n_total,
        n_predictive=n_predictive,
        reason_counts=reason_counts,
        future_runs=future_runs,
    )
    if calculation_error is not None:
        details["calculation_error"] = calculation_error
    detected = (
        f"Eligibility retained {n_predictive} of {n_total} examples. "
        f"First-match SKIP counts were {reason_counts}."
    )
    if calculation_error is not None:
        detected += f" The admitted batch could not be evaluated: {calculation_error}."
    return Finding(
        pillar=PILLAR,
        title=(
            f"{_prefix(future_runs=future_runs, model_a=model_a, model_b=model_b)} "
            f"not assessed; n_predictive={n_predictive}/{n_total}; "
            "predictive range (normal-theory approximation) unavailable"
        ),
        status=Status.SKIP,
        why=(
            "This advisory prediction needs complete, finite, independent run "
            "streams for both models on a consistent example batch."
        ),
        how_detected=f"{detected} {COVERAGE_LIMITATION}",
        how_to_fix=(
            "Provide nonempty finite runs for both models, keep each stored "
            "score equal to its run mean, and do not mix singleton and "
            "multi-run examples."
        ),
        details=details,
    )


def audit_predictive_rerun(
    data: EvalData,
    model_a: str,
    model_b: str,
    *,
    future_runs: int,
) -> list[Finding]:
    """Add one advisory predictive finding after explicit eligibility checks."""
    reason_counts = {reason: 0 for reason in _REASONS}
    candidates: list[tuple[np.ndarray, np.ndarray]] = []

    for example in data.examples:
        runs = example.runs or {}
        raw_a = runs.get(model_a)
        raw_b = runs.get(model_b)
        if model_a not in example.scores or not _present_stream(raw_a):
            reason_counts["missing_runs_a"] += 1
            continue
        if model_b not in example.scores or not _present_stream(raw_b):
            reason_counts["missing_runs_b"] += 1
            continue

        values_a = _finite_stream(raw_a)
        if values_a is None:
            reason_counts["nonfinite_a"] += 1
            continue
        values_b = _finite_stream(raw_b)
        if values_b is None:
            reason_counts["nonfinite_b"] += 1
            continue

        agrees_a = np.isclose(
            example.scores[model_a],
            values_a.mean(),
            rtol=1e-10,
            atol=1e-12,
        )
        agrees_b = np.isclose(
            example.scores[model_b],
            values_b.mean(),
            rtol=1e-10,
            atol=1e-12,
        )
        if not bool(agrees_a and agrees_b):
            reason_counts["score_run_mean_disagreement"] += 1
            continue
        candidates.append((values_a, values_b))

    counts = [
        int(values.size)
        for pair in candidates
        for values in pair
    ]
    if any(count == 1 for count in counts) and any(count > 1 for count in counts):
        retained = []
        for values_a, values_b in candidates:
            if values_a.size == 1 or values_b.size == 1:
                reason_counts["singleton_in_mixed"] += 1
            else:
                retained.append((values_a, values_b))
        candidates = retained

    n_total = data.n_examples
    n_predictive = len(candidates)
    if n_predictive == 0:
        return [_skip_finding(
            model_a=model_a,
            model_b=model_b,
            n_total=n_total,
            n_predictive=n_predictive,
            reason_counts=reason_counts,
            future_runs=future_runs,
        )]

    runs_a = [values_a for values_a, _ in candidates]
    runs_b = [values_b for _, values_b in candidates]
    try:
        result = predictive_rerun_normal_theory(
            runs_a,
            runs_b,
            future_runs=future_runs,
        )
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return [_skip_finding(
            model_a=model_a,
            model_b=model_b,
            n_total=n_total,
            n_predictive=n_predictive,
            reason_counts=reason_counts,
            future_runs=future_runs,
            calculation_error=str(exc),
        )]

    range_text = (
        f"[{result.normal_theory_range_low:+.4g}, "
        f"{result.normal_theory_range_high:+.4g}]"
    )
    if result.normal_theory_probability_a_better is None:
        result_text = (
            f"not estimated; n_predictive={n_predictive}/{n_total}; "
            "predictive range (normal-theory approximation) for B - A "
            f"{range_text}; degenerate_reason={result.degenerate_reason}"
        )
    else:
        result_text = (
            f"= {result.normal_theory_probability_a_better:.1%}; "
            f"n_predictive={n_predictive}/{n_total}; "
            "predictive range (normal-theory approximation) for B - A "
            f"{range_text}"
        )

    details = {
        "check": CHECK,
        "assessed": True,
        "model_a": model_a,
        "model_b": model_b,
        "n_total": n_total,
        "n_predictive": n_predictive,
        "reason_counts": reason_counts,
        "future_runs": result.future_runs,
        "method": result.method,
        "normal_theory_probability_a_better": (
            result.normal_theory_probability_a_better
        ),
        "normal_theory_range_low": result.normal_theory_range_low,
        "normal_theory_range_high": result.normal_theory_range_high,
        "central_mass": result.central_mass,
        "degenerate_reason": result.degenerate_reason,
        "point_estimate_b_minus_a": result.point_estimate_b_minus_a,
        "prediction_variance": result.prediction_variance,
        "degrees_of_freedom": result.degrees_of_freedom,
        "coverage_limitation": COVERAGE_LIMITATION,
    }
    return [Finding(
        pillar=PILLAR,
        title=(
            f"{_prefix(future_runs=future_runs, model_a=model_a, model_b=model_b)} "
            f"{result_text}"
        ),
        status=Status.PASS,
        why=(
            "The observed mean-gap inference does not answer what may happen "
            "when these fixed examples are rerun with fresh independent draws."
        ),
        how_detected=(
            f"Eligibility retained {n_predictive} of {n_total} examples. "
            f"First-match SKIP counts were {reason_counts}. Separate A and B "
            f"streams were evaluated with {result.method}. "
            f"{COVERAGE_LIMITATION}"
        ),
        how_to_fix=(
            "Treat this as an advisory working-normal prediction. Do not use it "
            "as a calibrated coverage guarantee or as a replacement for the "
            "mean-gap inference and verdict."
        ),
        details=details,
    )]
