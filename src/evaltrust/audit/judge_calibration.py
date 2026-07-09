"""Judge calibration: does the AI judge agree with the humans?

Given a human/gold judge, measures how well each AI judge matches it: exact-match
agreement for pass/fail scores, Spearman rank correlation for continuous ones.
Include human labels as a judge named ``gold``/``human``/``reference`` (etc.).
"""

from __future__ import annotations

import math
import warnings

from scipy.stats import spearmanr

from ..core.schema import EvalData, Finding, Status
from .judge_reliability import _judge_names

PILLAR = "Judge Reliability"
REFERENCE_NAMES = {"gold", "human", "humans", "reference", "ground_truth",
                   "groundtruth", "expert", "label", "truth"}


def _use_exact_match(data: EvalData, judges, models) -> bool:
    """True when calibration should use exact-match agreement, not correlation.

    Exact match for binary 0/1 or non-numeric scores; correlation only when
    scores are numeric and range beyond {0, 1}.
    """
    numeric: set[float] = set()
    for ex in data.examples:
        if not ex.judges:
            continue
        for j in judges:
            js = ex.judges.get(j)
            if not js:
                continue
            for m in models:
                if m in js:
                    v = js[m]
                    if not isinstance(v, (int, float)):   # bool is an int subclass
                        return True                        # non-numeric -> agreement
                    numeric.add(float(v))
    return not numeric or numeric <= {0.0, 1.0}


def _spearman(judge_vals: list[float], ref_vals: list[float]) -> float | None:
    """Spearman rank correlation, or ``None`` when it isn't defined.

    Undefined when there are fewer than two paired points, or when either side is
    constant (no ranks to correlate). Callers treat ``None`` as "not measurable"
    rather than reporting a spurious number.
    """
    if len(judge_vals) < 2:
        return None
    with warnings.catch_warnings():        # a constant input warns, then returns nan
        warnings.simplefilter("ignore")
        rho = float(spearmanr(judge_vals, ref_vals).statistic)
    return None if math.isnan(rho) else rho


def audit_judge_calibration(
    data: EvalData,
    model_a: str,
    model_b: str,
    threshold: float = 0.8,
    reference_judge: str | None = None,
) -> list[Finding]:
    """Return a calibration finding, or [] when there's nothing to calibrate.

    ``threshold`` is the pass floor (a fraction agreed, or a correlation).
    """
    if not data.has_judges:
        return []
    judges = _judge_names(data)
    ref = reference_judge or next(
        (j for j in judges if j.lower() in REFERENCE_NAMES), None)
    if ref is None or ref not in judges:
        return []

    others = [j for j in judges if j != ref]
    if not others:                          # no AI judge to calibrate
        return []
    if _use_exact_match(data, [ref, *others], (model_a, model_b)):
        return _calibration_exact_match(data, model_a, model_b, threshold, ref, others)
    return _calibration_correlation(data, model_a, model_b, threshold, ref, others)


def _calibration_exact_match(
    data: EvalData, model_a: str, model_b: str, threshold: float,
    ref: str, others: list[str],
) -> list[Finding]:
    """Exact-match agreement for binary (pass/fail) judge scores."""
    accuracies: dict[str, float] = {}
    for j in others:
        matches = total = 0
        for ex in data.examples:
            if not ex.judges:
                continue
            j_scores, ref_scores = ex.judges.get(j), ex.judges.get(ref)
            if not j_scores or not ref_scores:
                continue
            for m in (model_a, model_b):
                if m in j_scores and m in ref_scores:
                    total += 1
                    matches += int(j_scores[m] == ref_scores[m])
        if total:
            accuracies[j] = matches / total

    if not accuracies:
        return []

    worst_judge = min(accuracies, key=accuracies.get)
    worst = accuracies[worst_judge]
    good = worst >= threshold
    per_judge = ", ".join(f"{j} {a:.0%}" for j, a in accuracies.items())

    return [Finding(
        pillar=PILLAR,
        title=(f"Judges track {ref}" if good
               else f"A judge disagrees with {ref}"),
        status=Status.PASS if good else Status.WARN,
        why=(
            "A judge is only trustworthy if it agrees with people. If it often "
            "disagrees with the human labels, its scores — and any ranking built "
            "on them — carry that error."
        ),
        how_detected=(
            f"Agreement with {ref} (treated as ground truth): {per_judge}."),
        how_to_fix=(
            f"The judge(s) track {ref} closely." if good else
            f"Recalibrate or replace {worst_judge}; it matches {ref} only "
            f"{worst:.0%} of the time."
        ),
        details={"check": "judge_calibration", "reference": ref,
                 "accuracies": accuracies, "worst_judge": worst_judge,
                 "worst_accuracy": worst},
    )]


def _calibration_correlation(
    data: EvalData, model_a: str, model_b: str, threshold: float,
    ref: str, others: list[str],
) -> list[Finding]:
    """Spearman rank correlation for continuous judge scores."""
    correlations: dict[str, float] = {}
    max_points = 0
    for j in others:
        judge_vals: list[float] = []
        ref_vals: list[float] = []
        for ex in data.examples:
            if not ex.judges:
                continue
            j_scores, ref_scores = ex.judges.get(j), ex.judges.get(ref)
            if not j_scores or not ref_scores:
                continue
            for m in (model_a, model_b):
                if m in j_scores and m in ref_scores:
                    judge_vals.append(j_scores[m])
                    ref_vals.append(ref_scores[m])
        max_points = max(max_points, len(judge_vals))
        rho = _spearman(judge_vals, ref_vals)
        if rho is not None:
            correlations[j] = rho

    if not correlations:
        if max_points == 0:
            # No comparable scores at all: nothing to calibrate, so stay silent.
            return []
        # Some comparable data, but no judge has two or more paired points whose
        # values vary (both needed for a rank correlation).
        return [Finding(
            pillar=PILLAR,
            title=f"Judge calibration vs {ref} not measurable",
            status=Status.SKIP,
            why=(
                "Judge scores here are on a continuous scale, so calibration is a "
                "rank correlation against the reference — but no judge has enough "
                "varied, paired data to compute one."
            ),
            how_detected=(
                f"No AI judge had two or more comparable {ref}-vs-judge score "
                "pairs whose values vary across examples (a rank correlation "
                "needs both)."
            ),
            how_to_fix=(
                f"Add more examples scored by both {ref} and the AI judge(s), on a "
                "scale whose values vary across examples, so a Spearman rank "
                "correlation can be computed."
            ),
            details={"check": "judge_calibration", "metric": "spearman",
                     "reference": ref},
        )]

    worst_judge = min(correlations, key=correlations.get)
    worst = correlations[worst_judge]
    good = worst >= threshold
    per_judge = ", ".join(f"{j} rho={r:.2f}" for j, r in correlations.items())

    return [Finding(
        pillar=PILLAR,
        title=(f"Judges rank like {ref}" if good
               else f"A judge's ranking diverges from {ref}"),
        status=Status.PASS if good else Status.WARN,
        why=(
            "A judge is only trustworthy if it ranks answers like people do. On a "
            "continuous scale, what matters for comparing two models is that the "
            "judge orders examples the way the human labels do — a rank "
            "correlation, not exact agreement — so any ranking built on it holds."
        ),
        how_detected=(
            f"Spearman rank correlation with {ref} (continuous scores, treated as "
            f"ground truth): {per_judge}."),
        how_to_fix=(
            f"The judge(s) rank examples like {ref} does." if good else
            f"Recalibrate or replace {worst_judge}; its scores correlate with "
            f"{ref} only rho={worst:.2f} (a rank correlation, not an agreement rate)."
        ),
        details={"check": "judge_calibration", "metric": "spearman",
                 "reference": ref, "correlations": correlations,
                 "worst_judge": worst_judge, "worst_correlation": worst},
    )]
