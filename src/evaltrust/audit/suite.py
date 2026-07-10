"""Audit a multi-metric evaluation suite.

Real evals score several metrics per example (correctness, safety, helpfulness).
A suite is just a set of named single-metric datasets, so we audit each one with
the existing engine, comparing the *same* pair of models throughout, and correct
the significance threshold for the number of metrics tested.

Testing many metrics at the same alpha inflates false positives (test 20 metrics
at 0.05 and one looks "significant" by luck). Bonferroni is the default; Holm-
Bonferroni is available when a less conservative step-down correction is useful.

Whichever correction runs, every metric is audited *once*, at the alpha that
correction assigns it. Nothing about a finished report is rewritten afterwards:
a corrected alpha has to reach the checks before they run, or the equivalence
test and the minimum-detectable-effect would still be answering the uncorrected
question.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import numpy as np

from ..config import AuditConfig
from ..core.schema import EvalData
from ..stats.multiple import holm_bonferroni
from .runner import AuditReport, run_audit
from .statistical import decision_p_value
from .verdict import VerdictLevel, enforce_level

# Worst-to-best ordering for rolling metric verdicts up into one.
_RANK = {VerdictLevel.LOW: 0, VerdictLevel.MODERATE: 1, VerdictLevel.HIGH: 2}

# Spellings accepted for each correction, normalised to one identifier.
_METHODS = {
    "bonferroni": "bonferroni",
    "holm": "holm",
    "holm-bonferroni": "holm",
    "none": "none",
}


@dataclass(frozen=True)
class SuiteReport:
    reports: "OrderedDict[str, AuditReport]"
    alpha: float
    corrected_alpha: float          # the strictest per-metric threshold
    correction: str                 # human-readable description
    correction_method: str = "bonferroni"   # stable id: bonferroni | holm | none
    corrected_alpha_by_metric: dict[str, float] = field(default_factory=dict)

    @property
    def overall_level(self) -> VerdictLevel:
        """The worst verdict across metrics — the suite is only as trustworthy as
        its weakest metric."""
        return min((r.verdict.level for r in self.reports.values()),
                   key=lambda lvl: _RANK[lvl])

    def raise_if_below(self, minimum: "str | VerdictLevel" = "moderate") -> "SuiteReport":
        """Raise UntrustworthyError if the suite's overall (weakest) confidence is
        below ``minimum``. Returns self so it can be chained."""
        enforce_level(self.overall_level, minimum, context="the metric suite")
        return self

    def to_dict(self) -> dict:
        return {
            "overall_level": self.overall_level.name,
            "alpha": self.alpha,
            "corrected_alpha": self.corrected_alpha,
            "correction": self.correction,
            "correction_method": self.correction_method,
            "corrected_alpha_by_metric": self.corrected_alpha_by_metric,
            "metrics": {m: r.to_dict() for m, r in self.reports.items()},
        }


def _suite_models(suite: dict[str, EvalData], model_a, model_b) -> tuple[str, str]:
    """Pick one model pair to compare across every metric.

    Ranks models by their mean score averaged over all metrics, so the same two
    models are compared consistently rather than a different pair per metric.
    """
    if model_a is not None and model_b is not None:
        return model_a, model_b

    totals: "OrderedDict[str, list[float]]" = OrderedDict()
    for data in suite.values():
        for m in data.models:
            vals = [ex.scores[m] for ex in data.examples if m in ex.scores]
            if vals:
                totals.setdefault(m, []).append(float(np.mean(vals)))
    if len(totals) < 2:
        raise ValueError("A suite needs at least two models to compare.")
    ranked = sorted(totals, key=lambda m: np.mean(totals[m]), reverse=True)
    return ranked[0], ranked[1]



def _resolve_method(correction, cfg, correct) -> str:
    """Normalise the requested correction into one of the ``_METHODS`` ids."""
    requested = correction if correction is not None else cfg.suite_correction
    key = str(requested).strip().lower().replace("_", "-")
    if key not in _METHODS:
        raise ValueError(
            f"Unknown suite correction {requested!r}; "
            "use 'bonferroni', 'holm', or 'none'.")
    method = _METHODS[key]

    if not correct:
        warnings.warn(
            "audit_suite(correct=False) is deprecated; "
            "pass correction='none' instead.",
            DeprecationWarning, stacklevel=3)
        method = "none"
    return method


def _correction_label(method: str, alpha: float, k: int, by_metric: dict) -> str:
    """Describe what the correction did to alpha.

    The renderers already state how many metrics there are, so the label doesn't
    repeat the count.
    """
    if method == "none":
        return ("none (single metric)" if k == 1
                else f"none - no correction applied across {k} metrics")
    if method == "bonferroni":
        return f"Bonferroni: alpha {alpha} / {k} = {alpha / k:.4f}"
    lo, hi = min(by_metric.values()), max(by_metric.values())
    return f"Holm-Bonferroni step-down: per-metric alpha {lo:.4f} to {hi:.4f}"


def audit_suite(
    suite: dict[str, EvalData],
    model_a: str | None = None,
    model_b: str | None = None,
    alpha: float = 0.05,
    equivalence_margin: float = 0.05,
    seed: int = 0,
    correct: bool = True,
    correction: str | None = None,
    config: "AuditConfig | None" = None,
) -> SuiteReport:
    if not suite:
        raise ValueError("The suite is empty.")

    cfg = config or AuditConfig(alpha=alpha, equivalence_margin=equivalence_margin,
                                seed=seed)
    method = _resolve_method(correction, cfg, correct)
    model_a, model_b = _suite_models(suite, model_a, model_b)

    k = len(suite)
    if k == 1:
        method = "none"

    if method == "holm":
        reports, alpha_by_metric = _audit_suite_holm(suite, model_a, model_b, cfg)
    else:
        per_metric = cfg.alpha / k if method == "bonferroni" else cfg.alpha
        alpha_by_metric = {metric: per_metric for metric in suite}
        metric_cfg = replace(cfg, alpha=per_metric)
        reports = OrderedDict(
            (metric, run_audit(data, model_a=model_a, model_b=model_b,
                               config=metric_cfg))
            for metric, data in suite.items())

    return SuiteReport(
        reports=reports,
        alpha=cfg.alpha,
        corrected_alpha=min(alpha_by_metric.values()),
        correction=_correction_label(method, cfg.alpha, k, alpha_by_metric),
        correction_method=method,
        corrected_alpha_by_metric=alpha_by_metric)


def _audit_suite_holm(suite, model_a, model_b, cfg):
    """Audit every metric once, each at the alpha Holm assigns it.

    Holm needs all p-values before it can assign any threshold, so this runs the
    significance test alone first (cheap: one permutation test per metric, no
    bootstraps), derives the step-down thresholds, then runs the real audit at
    those thresholds. The step-down rejection is handed to the audit rather than
    recomputed from ``p < threshold``, because a metric can clear its own
    threshold and still be retained when an earlier one failed.
    """
    metrics = list(suite)
    p_values = [
        decision_p_value(suite[metric], model_a, model_b,
                         n_resamples=cfg.n_resamples, seed=cfg.seed)
        for metric in metrics
    ]
    holm = holm_bonferroni(p_values, alpha=cfg.alpha)
    alpha_by_metric = dict(zip(metrics, holm.thresholds))

    reports: "OrderedDict[str, AuditReport]" = OrderedDict()
    for i, metric in enumerate(metrics):
        metric_cfg = replace(cfg, alpha=holm.thresholds[i])
        report = run_audit(suite[metric], model_a=model_a, model_b=model_b,
                           config=metric_cfg,
                           significant_override=holm.reject[i])
        reports[metric] = _annotate_holm(
            report, rejected=holm.reject[i],
            adjusted_p=holm.adjusted_pvalues[i],
            threshold=holm.thresholds[i],
            alpha=cfg.alpha, k=len(metrics))
    return reports, alpha_by_metric


def _annotate_holm(report, *, rejected, adjusted_p, threshold, alpha, k):
    """Record how Holm judged this metric on its decision finding.

    Presentation and provenance only. The outcome, status and verdict were
    already decided by the audit running at ``threshold``, so nothing here
    changes what the report concluded.
    """
    verb = "cleared" if rejected else "did not clear"
    note = (f"Holm-Bonferroni over {k} metrics set this metric's alpha to "
            f"{threshold:.4f}; its adjusted p = {adjusted_p:.4f} {verb} the "
            f"suite alpha {alpha:.4f}.")

    findings = []
    for finding in report.findings:
        if finding.details.get("check") != "decision":
            findings.append(finding)
            continue
        findings.append(replace(
            finding,
            how_detected=f"{finding.how_detected} {note}",
            details={**finding.details,
                     "correction": "holm",
                     "holm_rejected": rejected,
                     "holm_adjusted_p": adjusted_p,
                     "holm_alpha": threshold}))
    return replace(report, findings=findings)
