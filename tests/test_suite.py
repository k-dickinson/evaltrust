"""Tests for auditing a multi-metric suite.

A suite is audited as one single-metric audit per metric, sharing the same model
pair, with the significance threshold corrected for the number of metrics tested
so testing many metrics doesn't manufacture false positives.
"""

import json

import pytest

from evaltrust.audit.suite import audit_suite
from evaltrust.audit.verdict import VerdictLevel
from evaltrust.core.schema import EvalData, Example
from evaltrust.stats.multiple import holm_bonferroni


def metric_data(a_scores, b_scores):
    examples = [
        Example(id=str(i), scores={"A": float(a), "B": float(b)})
        for i, (a, b) in enumerate(zip(a_scores, b_scores))
    ]
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


def _finding(report, check):
    return [f for f in report.findings if f.details.get("check") == check][0]


def _decision(report):
    return _finding(report, "decision")


def test_audits_every_metric():
    suite = {
        "correctness": metric_data([0] * 200, [1] * 180 + [0] * 20),
        "safety": metric_data([1] * 100, [1] * 100),
    }
    report = audit_suite(suite, seed=0)
    assert set(report.reports.keys()) == {"correctness", "safety"}


def test_shares_one_model_pair_across_metrics():
    suite = {"m1": metric_data([0] * 30, [1] * 30),
             "m2": metric_data([1] * 30, [0] * 30)}
    report = audit_suite(suite)
    pairs = {(r.model_a, r.model_b) for r in report.reports.values()}
    assert len(pairs) == 1  # same two models compared for every metric


def test_bonferroni_corrects_alpha_by_metric_count():
    suite = {f"m{i}": metric_data([0] * 40, [1] * 40) for i in range(5)}
    report = audit_suite(suite, alpha=0.05)
    assert report.corrected_alpha == 0.05 / 5
    assert "bonferroni" in report.correction.lower()


def test_holm_bonferroni_correction_matches_stats_helper():
    suite = {
        "strong": metric_data([0] * 80, [1] * 70 + [0] * 10),
        "borderline": metric_data([0, 1] * 50, [1] * 55 + [0] * 45),
        "noise": metric_data([0, 1] * 50, [1, 0] * 50),
    }
    report = audit_suite(suite, alpha=0.05, correction="holm", seed=0)

    p_values = []
    thresholds = []
    adjusted = []
    for metric_report in report.reports.values():
        decision = [f for f in metric_report.findings
                    if f.details.get("check") == "decision"][0]
        p_values.append(decision.details["p_value"])
        thresholds.append(decision.details["holm_alpha"])
        adjusted.append(decision.details["holm_adjusted_p"])

    expected = holm_bonferroni(p_values, alpha=0.05)
    assert "holm" in report.correction.lower()
    assert report.corrected_alpha_by_metric == pytest.approx(
        dict(zip(suite, expected.thresholds)))
    assert thresholds == pytest.approx(expected.thresholds)
    assert adjusted == pytest.approx(expected.adjusted_pvalues)


def binary_metric(n, b_only, a_only):
    """A metric where only B passes ``b_only`` examples, only A passes ``a_only``.

    Binary scores route the decision through McNemar's *exact* test, so the
    p-value is a closed form rather than a resampled estimate. That keeps these
    fixtures pinned to a known p regardless of seed or resample count.
    """
    concordant = n - b_only - a_only
    a = [0.0] * b_only + [1.0] * a_only + [1.0] * concordant
    b = [1.0] * b_only + [0.0] * a_only + [1.0] * concordant
    return metric_data(a, b)


# Two metrics with p = 0.0309 and 0.0414: both clear a raw alpha of 0.05, and
# neither clears alpha/2, so no correction rejects either one. The gap is 0.025
# against a margin of 0.15, which makes them genuinely *equivalent* rather than
# merely unproven.
def _borderline_equivalent_suite():
    return {"m_small": binary_metric(400, 14, 4),
            "m_large": binary_metric(400, 15, 5)}


def test_holm_keeps_the_equivalent_outcome_it_cannot_reject():
    """Losing significance must not silently become 'inconclusive'.

    The outcome is a cascade (significant -> equivalent -> inconclusive) and the
    equivalence arm is only reached when significance fails. A correction that
    strips significance after the fact would skip it and mislabel a real
    equivalence as missing data.
    """
    report = audit_suite(_borderline_equivalent_suite(), alpha=0.05,
                         equivalence_margin=0.15, correction="holm", seed=0)

    for metric_report in report.reports.values():
        decision = _decision(metric_report)
        assert decision.details["holm_rejected"] is False
        assert decision.details["outcome"] == "equivalent"


def test_holm_and_bonferroni_agree_when_neither_rejects():
    """Same data, same rejections — so the two corrections must agree on outcome."""
    outcomes = {}
    for method in ("bonferroni", "holm"):
        report = audit_suite(_borderline_equivalent_suite(), alpha=0.05,
                             equivalence_margin=0.15, correction=method, seed=0)
        outcomes[method] = {m: _decision(r).details["outcome"]
                            for m, r in report.reports.items()}

    assert outcomes["holm"] == outcomes["bonferroni"]
    assert set(outcomes["holm"].values()) == {"equivalent"}


def test_holm_step_down_retains_a_metric_that_clears_its_own_threshold():
    """Holm is step-down, not per-hypothesis.

    ``m_large`` has p = 0.0391 against its own threshold of 0.05. Comparing the
    two in isolation would reject it. Holm must not, because the smaller p-value
    already failed, and every larger p-value is retained after the first failure.
    """
    suite = {"m_small": binary_metric(20, 6, 0),   # p = 0.0312, threshold 0.025
             "m_large": binary_metric(20, 8, 1)}   # p = 0.0391, threshold 0.050
    report = audit_suite(suite, alpha=0.05, equivalence_margin=0.05,
                         correction="holm", seed=0)

    large = _decision(report.reports["m_large"])
    assert large.details["p_value"] < large.details["holm_alpha"]  # naively "significant"
    assert large.details["holm_rejected"] is False
    assert large.details["outcome"] == "inconclusive"


def test_holm_corrects_every_finding_not_just_the_decision():
    """The corrected alpha must reach the checks, not be patched on afterwards.

    ``precision`` reports whether the sample reached a conclusion. If the audit
    ran at the uncorrected alpha and only the decision was rewritten, the report
    would say 'inconclusive' and 'sample size was sufficient' side by side.
    """
    suite = {"m_small": binary_metric(20, 6, 0),
             "m_large": binary_metric(20, 8, 1)}
    report = audit_suite(suite, alpha=0.05, equivalence_margin=0.05,
                         correction="holm", seed=0)

    for metric, metric_report in report.reports.items():
        decision = _decision(metric_report)
        precision = _finding(metric_report, "precision")
        conclusive = decision.details["outcome"] in {"significant", "equivalent"}
        assert precision.details["conclusive"] is conclusive, metric
        # the decision was judged against the metric's own corrected alpha
        assert decision.details["alpha"] == pytest.approx(
            report.corrected_alpha_by_metric[metric])


def test_holm_rejections_are_a_superset_of_bonferroni():
    """Holm is uniformly more powerful; it can never reject less."""
    suite = {
        "strong": binary_metric(200, 40, 2),
        "medium": binary_metric(200, 15, 5),
        "weak": binary_metric(200, 8, 6),
    }
    significant = {}
    for method in ("bonferroni", "holm"):
        report = audit_suite(suite, alpha=0.05, correction=method, seed=0)
        significant[method] = {
            m for m, r in report.reports.items()
            if _decision(r).details["outcome"] == "significant"}

    assert significant["bonferroni"] <= significant["holm"]


def test_every_correction_reports_a_per_metric_alpha():
    suite = {f"m{i}": metric_data([0] * 40, [1] * 40) for i in range(4)}
    expected = {"bonferroni": 0.05 / 4, "none": 0.05}
    for method, per_metric in expected.items():
        report = audit_suite(suite, alpha=0.05, correction=method, seed=0)
        assert report.correction_method == method
        assert report.corrected_alpha_by_metric == {m: per_metric for m in suite}
        assert report.corrected_alpha == per_metric


def test_disabling_correction_is_stated_in_the_report():
    suite = {f"m{i}": metric_data([0] * 40, [1] * 40) for i in range(3)}
    report = audit_suite(suite, alpha=0.05, correction="none")
    assert report.correction_method == "none"
    assert "no correction applied" in report.correction


def test_correction_method_is_a_stable_identifier():
    suite = {"a": metric_data([0] * 40, [1] * 40),
             "b": metric_data([0] * 40, [1] * 40)}
    for spelling in ("holm", "HOLM", "holm-bonferroni", "holm_bonferroni"):
        report = audit_suite(suite, correction=spelling, seed=0)
        assert report.correction_method == "holm"


def test_unknown_suite_correction_errors():
    suite = {"a": metric_data([0] * 40, [1] * 40),
             "b": metric_data([0] * 40, [1] * 40)}
    with pytest.raises(ValueError, match="Unknown suite correction"):
        audit_suite(suite, correction="banana")


def test_correct_false_is_deprecated_but_disables_correction():
    suite = {"a": metric_data([0] * 40, [1] * 40),
             "b": metric_data([0] * 40, [1] * 40)}
    with pytest.warns(DeprecationWarning):
        report = audit_suite(suite, correct=False)
    assert report.correction_method == "none"


def test_no_correction_for_single_metric():
    report = audit_suite({"score": metric_data([0] * 40, [1] * 40)}, alpha=0.05)
    assert report.corrected_alpha == 0.05
    assert report.correction_method == "none"
    assert "single metric" in report.correction


def test_overall_level_is_the_worst_metric():
    suite = {
        "good": metric_data([0] * 200, [1] * 180 + [0] * 20),   # clear win -> HIGH
        "noise": metric_data([0, 1] * 60, [1, 0] * 60),         # noise -> LOW
    }
    report = audit_suite(suite, seed=0)
    assert report.overall_level is VerdictLevel.LOW


def test_to_dict_is_json_serializable():
    suite = {"correctness": metric_data([0] * 60, [1] * 55 + [0] * 5),
             "safety": metric_data([1] * 60, [1] * 58 + [0] * 2)}
    d = audit_suite(suite, seed=0).to_dict()
    text = json.dumps(d)
    parsed = json.loads(text)
    assert set(parsed["metrics"].keys()) == {"correctness", "safety"}
    assert parsed["overall_level"] in {"HIGH", "MODERATE", "LOW"}
    assert "corrected_alpha" in parsed
