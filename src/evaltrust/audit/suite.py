"""Audit a multi-metric evaluation suite.

Audits each metric's dataset for the same model pair, correcting the significance
threshold for the number of metrics via Bonferroni (default) or Holm-Bonferroni.
Holm runs in two passes because a metric's threshold depends on its p-value rank.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from ..config import AuditConfig
from ..core.schema import EvalData, Finding
from ..stats.multiplicity import holm_bonferroni
from ..versions import METHODOLOGY_VERSION, SCHEMA_VERSION
from .runner import AuditReport, run_audit
from .verdict import VerdictLevel, enforce_level

# Worst-to-best ordering for rolling metric verdicts up into one.
_RANK = {VerdictLevel.LOW: 0, VerdictLevel.MODERATE: 1, VerdictLevel.HIGH: 2}

_METHODS = ("bonferroni", "holm", "none")


@dataclass(frozen=True)
class SuiteReport:
    reports: "OrderedDict[str, AuditReport]"
    alpha: float
    corrected_alpha: float
    correction: str
    metric_alphas: "OrderedDict[str, float]" = field(default_factory=OrderedDict)
    adjusted_p: "OrderedDict[str, float]" = field(default_factory=OrderedDict)
    _config_gated: frozenset = field(default_factory=frozenset, compare=False, repr=False)
    _config_weights: "dict | MappingProxyType" = field(
        default_factory=dict, compare=False, repr=False)

    @property
    def overall_level(self) -> VerdictLevel:
        """Roll up per-metric verdicts into one suite-level verdict.

        Evaluation order (first matching rule wins):

        1. **Gate check** — if any gated metric (``_config_gated``) is below
           HIGH the whole suite is LOW, regardless of every other metric.
        2. **Fallback** — plain weakest-metric (original behaviour).

        Only named metrics present in the suite are considered; unknown gate
        names are silently ignored.

        Note: ``_config_weights`` is validated and stored but not yet used in
        rollup — weighting changes the ``--fail-under`` contract and will land
        in a follow-up PR once that contract is settled.
        """
        levels = {m: r.verdict.level for m, r in self.reports.items()}

        # 1. Gated metrics: any gate failure → whole suite is LOW.
        for metric, level in levels.items():
            if metric in self._config_gated and level is not VerdictLevel.HIGH:
                return VerdictLevel.LOW

        # 2. Default: weakest metric wins.
        return min(levels.values(), key=lambda lvl: _RANK[lvl])

    def raise_if_below(self, minimum: "str | VerdictLevel" = "moderate") -> "SuiteReport":
        """Raise UntrustworthyError if the suite's overall confidence is below
        ``minimum``. Returns self so it can be chained."""
        enforce_level(self.overall_level, minimum, context="the metric suite")
        return self

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "methodology_version": METHODOLOGY_VERSION,
            "overall_level": self.overall_level.name,
            "alpha": self.alpha,
            "corrected_alpha": self.corrected_alpha,
            "correction": self.correction,
            "metric_alphas": dict(self.metric_alphas),
            "adjusted_p": dict(self.adjusted_p),
            "metrics": {m: r.to_dict() for m, r in self.reports.items()},
            # Surface the applied policy so downstream JSON consumers know which
            # gates and weights produced this overall_level.
            "applied_gates": sorted(self._config_gated),
            "applied_weights": dict(self._config_weights),
        }


def _suite_models(suite: dict[str, EvalData], model_a, model_b) -> tuple[str, str]:
    """Pick one model pair to compare across every metric.

    Ranks models by their mean score averaged over all metrics, so the same two
    models are compared consistently rather than a different pair per metric.
    """
    if model_a is not None and model_b is not None:
        return model_a, model_b

    totals: "OrderedDict[str, list[float]]" = OrderedDict()
    preference_models: "OrderedDict[str, None]" = OrderedDict()
    for data in suite.values():
        if data.has_preferences:
            for model in data.models:
                preference_models.setdefault(model, None)
        for m in data.models:
            vals = [ex.scores[m] for ex in data.examples if m in ex.scores]
            if vals:
                totals.setdefault(m, []).append(float(np.mean(vals)))
    if len(totals) >= 2:
        ranked = sorted(totals, key=lambda m: np.mean(totals[m]), reverse=True)
        return ranked[0], ranked[1]
    if len(preference_models) == 2:
        pair = list(preference_models)
        return pair[0], pair[1]
    raise ValueError(
        "A suite needs one unambiguous pair of models to compare. Include two "
        "models in the data or pass model_a and model_b.")


def audit_suite(
    suite: dict[str, EvalData],
    model_a: str | None = None,
    model_b: str | None = None,
    alpha: float = 0.05,
    equivalence_margin: float = 0.05,
    seed: int = 0,
    correct: bool = True,
    config: "AuditConfig | None" = None,
    correction: str | None = None,
) -> SuiteReport:
    """Audit every metric in a suite for the same model pair.

    ``correction`` is ``{"bonferroni", "holm", "none"}``; ``None`` defers to the
    config. ``correct=False`` is a deprecated alias for ``correction="none"``.
    """
    if not suite:
        raise ValueError("The suite is empty.")

    cfg = config or AuditConfig(alpha=alpha, equivalence_margin=equivalence_margin,
                                seed=seed)

    # Resolve the correction: an explicit argument beats the config; the legacy
    # `correct=False` switch still forces no correction.
    method = correction if correction is not None else cfg.correction
    if not correct:
        method = "none"
    if method not in _METHODS:
        raise ValueError(
            f"correction must be one of {_METHODS}, got {method!r}")

    model_a, model_b = _suite_models(suite, model_a, model_b)
    k = len(suite)

    mixed_hypotheses = any(
        data.has_preferences and data.differences(model_a, model_b).size
        for data in suite.values()
    )
    if k > 1 and method != "none" and mixed_hypotheses:
        raise ValueError(
            "A corrected multi-metric suite cannot combine score and preference "
            "significance in the same metric yet. Split the hypothesis families "
            "or use correction='none'."
        )

    # A single metric can't inflate anything, so no correction ever applies.
    if k == 1 or method == "none":
        return _uncorrected_suite(suite, model_a, model_b, cfg, single=(k == 1))
    if method == "bonferroni":
        return _bonferroni_suite(suite, model_a, model_b, cfg, k)
    return _holm_suite(suite, model_a, model_b, cfg, k)


def _run_metrics(suite, model_a, model_b, cfg_for,
                 significant_for=lambda _m: None) -> "OrderedDict[str, AuditReport]":
    """Run one audit per metric.

    ``cfg_for(metric)`` supplies each metric's config. ``significant_for(metric)``
    optionally passes a pre-decided significance (used by Holm); the default
    ``None`` lets each metric's audit decide with its own ``p < alpha``.
    """
    reports: "OrderedDict[str, AuditReport]" = OrderedDict()
    for metric, data in suite.items():
        metric_cfg = cfg_for(metric)
        # All-pairs belongs to the single-file comparison. Do not build a nested
        # pair-by-metric family or correct pairs against a suite-adjusted alpha.
        if metric_cfg.all_pairs:
            metric_cfg = replace(metric_cfg, all_pairs=False)
        reports[metric] = run_audit(
            data, model_a=model_a, model_b=model_b, config=metric_cfg,
            significant=significant_for(metric))
    return reports


def _decision(report: AuditReport) -> "Finding | None":
    for f in report.findings:
        if f.details.get("check") in {"decision", "preference_significance"}:
            return f
    return None


def _pvalue(report: AuditReport) -> float:
    finding = _decision(report)
    return float(finding.details["p_value"]) if finding is not None else 1.0


def _n_significant(reports) -> int:
    return sum(1 for r in reports.values()
               if (f := _decision(r)) is not None
               and f.details.get("outcome") == "significant")


def _uncorrected_suite(suite, model_a, model_b, cfg, single: bool) -> SuiteReport:
    reports = _run_metrics(suite, model_a, model_b, lambda _m: cfg)
    metric_alphas = OrderedDict((m, cfg.alpha) for m in suite)
    adjusted_p = OrderedDict((m, _pvalue(reports[m])) for m in suite)
    description = "none (single metric)" if single else "none (uncorrected)"
    return SuiteReport(reports=reports, alpha=cfg.alpha, corrected_alpha=cfg.alpha,
                       correction=description, metric_alphas=metric_alphas,
                       adjusted_p=adjusted_p, _config_gated=cfg.gated_metrics,
                       _config_weights=cfg.metric_weights)


def _bonferroni_suite(suite, model_a, model_b, cfg, k: int) -> SuiteReport:
    corrected_alpha = cfg.alpha / k
    metric_cfg = replace(cfg, alpha=corrected_alpha)
    reports = _run_metrics(suite, model_a, model_b, lambda _m: metric_cfg)
    metric_alphas = OrderedDict((m, corrected_alpha) for m in suite)
    adjusted_p = OrderedDict(
        (m, min(k * _pvalue(reports[m]), 1.0)) for m in suite)
    description = (f"Bonferroni: alpha {cfg.alpha} / {k} metrics "
                  f"= {corrected_alpha:.4f}")
    return SuiteReport(reports=reports, alpha=cfg.alpha,
                       corrected_alpha=corrected_alpha, correction=description,
                       metric_alphas=metric_alphas, adjusted_p=adjusted_p,
                       _config_gated=cfg.gated_metrics,
                       _config_weights=cfg.metric_weights)


def _holm_suite(suite, model_a, model_b, cfg, k: int) -> SuiteReport:
    metrics = list(suite)

    # Pass 1: run each metric at the raw alpha only to read its p-value (the
    # p-value doesn't depend on alpha, so this pass exists purely to rank them).
    pass1 = _run_metrics(suite, model_a, model_b, lambda _m: cfg)
    pvalues = [_pvalue(pass1[m]) for m in metrics]

    rejected, adjusted = holm_bonferroni(pvalues, cfg.alpha)
    alpha_for = dict(zip(metrics, _holm_step_thresholds(pvalues, rejected, cfg.alpha)))
    rejected_for = dict(zip(metrics, rejected))

    # Pass 2: each metric's audit is told whether Holm rejected it, and re-runs at
    # its step threshold so the prose and equivalence CI quote that threshold.
    reports = _run_metrics(
        suite, model_a, model_b,
        lambda m: replace(cfg, alpha=alpha_for[m]),
        significant_for=lambda m: rejected_for[m])

    metric_alphas = OrderedDict((m, alpha_for[m]) for m in metrics)
    adjusted_p = OrderedDict((m, adjusted[i]) for i, m in enumerate(metrics))
    # corrected_alpha stays a scalar for back-compat: Holm's most conservative
    # (first) step is alpha / k, the same value Bonferroni uses everywhere.
    corrected_alpha = cfg.alpha / k
    description = (f"Holm-Bonferroni (step-down): {_n_significant(reports)} "
                  f"of {k} metrics significant")
    return SuiteReport(reports=reports, alpha=cfg.alpha,
                       corrected_alpha=corrected_alpha, correction=description,
                       metric_alphas=metric_alphas, adjusted_p=adjusted_p,
                       _config_gated=cfg.gated_metrics,
                       _config_weights=cfg.metric_weights)


def _holm_step_thresholds(pvalues, rejected, alpha: float) -> list[float]:
    """Per-metric Holm step threshold, for reporting only (not the decision).

    A rejected metric gets its own step ``alpha / (k - rank)``; a retained one
    gets the first failure's threshold. Quoted in prose and the TOST interval.
    """
    p = np.asarray(pvalues, dtype=float)
    k = p.size
    order = np.argsort(p, kind="stable")
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(k)
    n_rejected = int(sum(rejected))

    thresholds = []
    for i in range(k):
        if rejected[i]:
            thresholds.append(float(alpha / (k - rank[i])))
        else:
            thresholds.append(float(alpha / (k - n_rejected)))
    return thresholds
