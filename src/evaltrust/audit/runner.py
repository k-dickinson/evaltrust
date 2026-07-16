"""Runs every applicable audit check and assembles the final report."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import AuditConfig
from ..core.schema import EvalData, Finding, Status
from ..versions import METHODOLOGY_VERSION, SCHEMA_VERSION
from .allpairs import audit_all_pairs
from .benchmark_health import audit_benchmark_health
from .judge_calibration import audit_judge_calibration
from .judge_reliability import audit_judge_reliability
from .preference import audit_preferences
from .single import audit_single
from .repeatability import audit_repeatability
from .slice import audit_slices
from .statistical import audit_statistical_validity
from .verdict import Verdict, VerdictLevel, compute_verdict, enforce_level


@dataclass(frozen=True)
class AuditReport:
    model_a: str
    model_b: str | None          # None for a single-model audit
    n_examples: int
    source_format: str
    findings: list[Finding]
    verdict: Verdict
    models_available: list[str] = field(default_factory=list)

    @property
    def is_single(self) -> bool:
        return self.model_b is None

    def raise_if_below(self, minimum: "str | VerdictLevel" = "moderate") -> "AuditReport":
        """Raise UntrustworthyError if confidence is below ``minimum``.

        Drop this into a script or test to fail when the evaluation isn't
        trustworthy enough:  ``evaltrust.audit(results).raise_if_below("moderate")``.
        Returns self on success so it can be chained.
        """
        subject = self.model_a if self.is_single else f"{self.model_a} vs {self.model_b}"
        enforce_level(self.verdict.level, minimum, context=subject)
        return self

    def to_dict(self) -> dict:
        """A JSON-serializable representation of the whole audit."""
        models = [self.model_a] if self.is_single else [self.model_a, self.model_b]
        return {
            "schema_version": SCHEMA_VERSION,
            "methodology_version": METHODOLOGY_VERSION,
            "models": models,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "models_available": self.models_available,
            "n_examples": self.n_examples,
            "source_format": self.source_format,
            "verdict": self.verdict.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
        }


def _mean_score(data: EvalData, model: str) -> float:
    vals = [ex.scores[model] for ex in data.examples if model in ex.scores]
    return float(np.mean(vals)) if vals else float("-inf")


def _data_quality(data: EvalData) -> Finding | None:
    """Flag rows dropped during loading for missing or unreadable scores."""
    skipped = int(data.metadata.get("skipped_rows", 0))
    if skipped <= 0:
        return None
    kept = data.n_examples
    return Finding(
        pillar="Data Quality",
        title=f"{skipped} rows skipped while loading",
        status=Status.WARN,
        why=("Rows with a missing or unreadable score were dropped. If many were "
             "dropped, or they weren't random, the audit sees a biased slice of "
             "your data."),
        how_detected=(f"Loaded {kept} usable examples; skipped {skipped} rows "
                      "whose score couldn't be read."),
        how_to_fix="Check those rows in your results file and re-export if needed.",
        details={"check": "data_quality", "skipped_rows": skipped, "kept": kept},
    )


def _pairing_coverage(data: EvalData) -> Finding | None:
    """Flag examples dropped while pairing two single-model files."""
    unmatched = int(data.metadata.get("unmatched_examples", 0))
    if unmatched <= 0:
        return None
    kept = data.n_examples
    return Finding(
        pillar="Data Quality",
        title=f"{unmatched} examples dropped during pairing",
        status=Status.WARN,
        why=("Only examples present in both files are compared. If the overlap "
             "isn't random — one run stopped early, or covers different cases — "
             "the audit compares the models on a biased slice."),
        how_detected=(f"Paired {kept} shared examples; {unmatched} appeared in "
                      "only one file (or lacked a score) and were dropped."),
        how_to_fix=("Re-run both evaluations on the same example set, or check "
                    "that ids match between the files."),
        details={"check": "pairing_coverage", "unmatched_examples": unmatched,
                 "kept": kept},
    )


def _pick_models(data: EvalData) -> tuple[str, str]:
    """Compare the two strongest models by mean score (stable, documented)."""
    preference_only = data.has_preferences and not any(
        ex.scores for ex in data.examples)
    if preference_only:
        if len(data.models) < 2:
            raise ValueError(
                "Pairwise preferences need two models to compare. Include both "
                "model names in the input or pass model_a and model_b.")
        if len(data.models) > 2:
            raise ValueError(
                "Preference-only data names more than two models; name the two "
                "models to compare with model_a and model_b.")
        return data.models[0], data.models[1]
    if len(data.models) < 2:
        raise ValueError("EvalTrust needs at least two models to compare.")
    ranked = sorted(data.models, key=lambda m: _mean_score(data, m), reverse=True)
    return ranked[0], ranked[1]


def run_audit(
    data: EvalData,
    model_a: str | None = None,
    model_b: str | None = None,
    alpha: float = 0.05,
    equivalence_margin: float = 0.05,
    seed: int = 0,
    threshold: float | None = None,
    config: "AuditConfig | None" = None,
    slice_by: str | None = None,
    *,
    significant: bool | None = None,
) -> AuditReport:
    # When no config is given, build one from the loose kwargs.
    cfg = config or AuditConfig(alpha=alpha, equivalence_margin=equivalence_margin,
                                seed=seed)

    # `significant` is a per-run override for the comparison path (used by Holm);
    # single-model audits ignore it. Dispatch: two models -> comparison; a
    # threshold or a lone model -> single.
    if model_a is not None and model_b is not None:
        return _comparison(data, model_a, model_b, cfg, significant=significant,
                           slice_by=slice_by)
    if threshold is not None:
        if data.has_preferences and not any(ex.scores for ex in data.examples):
            raise ValueError(
                "A threshold audit needs scores. Pairwise preferences need two "
                "models and cannot be audited against a score threshold.")
        return _single(data, model_a or _strongest(data), threshold, cfg)
    if len(data.models) == 1:
        if data.has_preferences and not any(ex.scores for ex in data.examples):
            raise ValueError(
                "Pairwise preferences need two models to compare. Include both "
                "model names in the input or pass model_a and model_b.")
        return _single(data, model_a or data.models[0], None, cfg)
    model_a, model_b = _pick_models(data)
    return _comparison(data, model_a, model_b, cfg, significant=significant,
                       slice_by=slice_by)


def _strongest(data: EvalData) -> str:
    if not data.models:
        raise ValueError("EvalTrust needs at least one model to audit.")
    return max(data.models, key=lambda m: _mean_score(data, m))


def _comparison(data, model_a, model_b, cfg, significant=None,
                slice_by=None) -> AuditReport:
    differences = data.differences(model_a, model_b)
    has_pair_scores = any(
        model_a in ex.scores or model_b in ex.scores for ex in data.examples)
    preference_only = not has_pair_scores and data.has_preferences
    if differences.size == 0 and not data.has_preferences:
        available = ", ".join(repr(model) for model in data.models) or "none"
        raise ValueError(
            f"No examples have scores for both '{model_a}' and '{model_b}', so "
            "there's nothing to compare. Check the models and provide scores or "
            f"preferences. Available models: {available}.")

    findings: list[Finding] = []
    for quality in (_data_quality(data), _pairing_coverage(data)):
        if quality is not None:
            findings.append(quality)
    if differences.size:
        findings += audit_statistical_validity(
            data, model_a, model_b, alpha=cfg.alpha,
            equivalence_margin=cfg.equivalence_margin, power_target=cfg.power_target,
            smallest_meaningful_effect=cfg.smallest_meaningful_effect,
            n_resamples=cfg.n_resamples, seed=cfg.seed, significant=significant)
    else:
        findings.append(_score_skip(
            "Statistical Validity",
            "score_statistical_validity",
            (
                "Preference-only data has no paired model scores."
                if preference_only else
                "No examples contain scores for both selected models."
            ),
            (
                "Add paired per-model scores to run the score test. Preference "
                "significance is assessed separately."
            ),
            "preference_only" if preference_only else "no_paired_scores",
        ))

    if cfg.all_pairs:
        findings += audit_all_pairs(data, cfg)

    if has_pair_scores:
        findings += audit_benchmark_health(
            data, [model_a, model_b],
            saturation_fraction=cfg.saturation_fraction, min_spread=cfg.min_spread,
            score_ceiling=cfg.score_ceiling)
    else:
        findings.append(_score_skip(
            "Benchmark Health",
            "benchmark_health",
            "Preference-only data has no scores for benchmark saturation or spread.",
            "Add per-model scores to assess benchmark headroom and discrimination.",
            "preference_only",
        ))

    if preference_only:
        findings += [
            _score_skip(
                "Repeatability",
                "repeatability",
                "Preference-only data has no repeated score runs to compare.",
                "Add repeated per-model score runs to assess score repeatability.",
                "preference_only",
            ),
            _score_skip(
                "Judge Reliability",
                "judge_reliability",
                "Preference votes are present, but per-judge model scores are absent.",
                "Add per-judge model scores to run score reliability and calibration checks.",
                "preference_only",
            ),
        ]
    else:
        findings += audit_repeatability(data, model_a, model_b)
        findings += audit_judge_reliability(
            data, model_a, model_b,
            agreement_threshold=cfg.judge_agreement_threshold)
        findings += audit_judge_calibration(
            data, model_a, model_b,
            threshold=cfg.judge_agreement_threshold,
            correlation_threshold=cfg.judge_correlation_threshold,
            reference_judge=cfg.reference_judge)

    if data.has_preferences:
        findings += audit_preferences(
            data, model_a, model_b, alpha=cfg.alpha,
            n_resamples=cfg.n_resamples, seed=cfg.seed,
            significant=significant if preference_only else None)

    if slice_by is not None:
        if differences.size:
            findings += audit_slices(
                data, model_a, model_b, slice_by, alpha=cfg.alpha,
                n_resamples=cfg.n_resamples, seed=cfg.seed,
                overall_mean_diff=float(differences.mean()))
        else:
            findings.append(_score_skip(
                "Per-slice Comparison",
                "slice_comparison",
                (
                    "Preference-only data has no paired model scores to "
                    "compare per slice."
                    if preference_only else
                    "No examples contain scores for both selected models, so "
                    "there is nothing to slice."
                ),
                (
                    "Add paired per-model scores to run the per-slice "
                    "comparison."
                ),
                "preference_only" if preference_only else "no_paired_scores",
            ))

    return AuditReport(
        model_a=model_a, model_b=model_b, n_examples=data.n_examples,
        source_format=data.source_format, findings=findings,
        verdict=compute_verdict(findings), models_available=list(data.models))


def _score_skip(pillar, check, detected, fix, reason) -> Finding:
    return Finding(
        pillar=pillar,
        title="Not assessed",
        status=Status.SKIP,
        why=(
            "This score-based check needs paired per-model scores, which "
            "preference-only data does not provide."
        ),
        how_detected=detected,
        how_to_fix=fix,
        details={
            "check": check,
            "assessed": False,
            "reason": reason,
        },
    )


def _single(data, model, threshold, cfg) -> AuditReport:
    findings: list[Finding] = []
    dq = _data_quality(data)
    if dq is not None:
        findings.append(dq)
    findings += audit_single(data, model, threshold=threshold, config=cfg)

    return AuditReport(
        model_a=model, model_b=None, n_examples=data.n_examples,
        source_format=data.source_format, findings=findings,
        verdict=compute_verdict(findings), models_available=list(data.models))
