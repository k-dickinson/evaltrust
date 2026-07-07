"""Runs every applicable audit check and assembles the final report."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.schema import EvalData, Finding
from .benchmark_health import audit_benchmark_health
from .judge_reliability import audit_judge_reliability
from .repeatability import audit_repeatability
from .statistical import audit_statistical_validity
from .verdict import Verdict, compute_verdict


@dataclass(frozen=True)
class AuditReport:
    model_a: str
    model_b: str
    n_examples: int
    source_format: str
    findings: list[Finding]
    verdict: Verdict


def _mean_score(data: EvalData, model: str) -> float:
    vals = [ex.scores[model] for ex in data.examples if model in ex.scores]
    return float(np.mean(vals)) if vals else float("-inf")


def _pick_models(data: EvalData) -> tuple[str, str]:
    """Compare the two strongest models by mean score (stable, documented)."""
    if len(data.models) < 2:
        raise ValueError("EvalLab needs at least two models to compare.")
    ranked = sorted(data.models, key=lambda m: _mean_score(data, m), reverse=True)
    return ranked[0], ranked[1]


def run_audit(
    data: EvalData,
    model_a: str | None = None,
    model_b: str | None = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> AuditReport:
    if model_a is None or model_b is None:
        model_a, model_b = _pick_models(data)

    findings: list[Finding] = []
    findings += audit_statistical_validity(data, model_a, model_b,
                                           alpha=alpha, seed=seed)
    findings += audit_benchmark_health(data, [model_a, model_b])
    findings += audit_repeatability(data, model_a, model_b)
    findings += audit_judge_reliability(data, model_a, model_b)

    return AuditReport(
        model_a=model_a,
        model_b=model_b,
        n_examples=data.n_examples,
        source_format=data.source_format,
        findings=findings,
        verdict=compute_verdict(findings),
    )
