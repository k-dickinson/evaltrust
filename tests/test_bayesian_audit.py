"""Tests for the optional Bayesian paired-comparison view."""

import numpy as np
import pytest

from evaltrust.audit.bayesian import audit_bayesian_win_probability
from evaltrust.audit.runner import run_audit
from evaltrust.config import AuditConfig
from evaltrust.core.schema import EvalData, Example, Preference, Status
from evaltrust.report.html import render_html
from evaltrust.report.terminal import render_markdown, render_plain, render_report


def make_data(differences):
    examples = [
        Example(id=str(index), scores={"A": 0.0, "B": float(difference)})
        for index, difference in enumerate(differences)
    ]
    return EvalData(models=["A", "B"], examples=examples, source_format="test")


def by_check(findings, check):
    (finding,) = [f for f in findings if f.details.get("check") == check]
    return finding


def _squash(s: str) -> str:
    """Collapse whitespace so wrap-induced newlines do not break substring asserts."""
    return " ".join(s.split())


def test_counts_negative_difference_as_a_win_and_zero_as_a_tie():
    finding = audit_bayesian_win_probability(
        make_data([-0.4, -0.1, 0.2, 0.0]), "A", "B"
    )[0]

    assert finding.status is Status.PASS
    assert finding.details["wins_a"] == 2
    assert finding.details["wins_b"] == 1
    assert finding.details["ties"] == 1
    assert finding.details["n_decisive"] == 3
    assert finding.details["assessed"] is True
    assert type(finding.to_dict()["details"]["probability_a_better"]) is float


def test_normalizes_confidence_to_a_plain_float_in_details():
    finding = audit_bayesian_win_probability(
        make_data([-1.0]), "A", "B", confidence=np.float32(0.95)
    )[0]
    assert type(finding.details["confidence"]) is float


def test_clean_win_agrees_with_frequentist_direction():
    data = make_data([-1.0] * 20)
    report = run_audit(
        data, model_a="A", model_b="B", config=AuditConfig(bayesian=True)
    )
    decision = by_check(report.findings, "decision")
    bayesian = by_check(report.findings, "bayesian_win_probability")

    assert decision.details["outcome"] == "significant"
    assert decision.title.startswith("A is significantly better")
    assert bayesian.details["probability_a_better"] > 0.5
    assert bayesian.details["ci_low"] > 0.5


def test_estimand_disagreement_is_reported_honestly():
    data = make_data([-0.01] * 9 + [1.0])
    report = run_audit(
        data, model_a="A", model_b="B", config=AuditConfig(bayesian=True)
    )
    decision = by_check(report.findings, "decision")
    bayesian = by_check(report.findings, "bayesian_win_probability")

    assert decision.title.startswith("Improvement of B over A")
    assert bayesian.details["wins_a"] == 9
    assert bayesian.details["wins_b"] == 1
    assert bayesian.details["probability_a_better"] > 0.5
    assert "P(A wins more often than B on decisive examples)" in bayesian.title


def test_all_ties_skip_without_a_prior_only_claim():
    finding = audit_bayesian_win_probability(make_data([0.0] * 5), "A", "B")[0]

    assert finding.status is Status.SKIP
    assert finding.details["assessed"] is False
    assert finding.details["n_decisive"] == 0
    assert finding.details["ties"] == 5
    assert "probability_a_better" not in finding.details
    assert finding.why.strip() and finding.how_detected.strip() and finding.how_to_fix.strip()


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1, np.nan, True])
def test_all_ties_still_validate_confidence(confidence):
    with pytest.raises(ValueError):
        audit_bayesian_win_probability(
            make_data([0.0]), "A", "B", confidence=confidence
        )


def test_preference_only_data_adds_a_score_based_skip():
    data = EvalData(
        models=["A", "B"],
        examples=[
            Example(
                id="1",
                scores={},
                preferences={"judge": Preference.TIE},
            )
        ],
        source_format="test",
    )

    report = run_audit(
        data, model_a="A", model_b="B", config=AuditConfig(bayesian=True)
    )
    finding = by_check(report.findings, "bayesian_win_probability")
    assert finding.status is Status.SKIP
    assert finding.details["assessed"] is False
    assert finding.details["n_decisive"] == 0
    assert "no paired scores" in finding.how_detected


def test_default_output_is_unchanged_and_flag_adds_one_finding_in_order():
    data = make_data([-1.0] * 20)
    default = run_audit(data, model_a="A", model_b="B")
    explicit_off = run_audit(
        data, model_a="A", model_b="B", config=AuditConfig(bayesian=False)
    )
    enabled = run_audit(
        data, model_a="A", model_b="B", config=AuditConfig(bayesian=True)
    )

    assert default.to_dict() == explicit_off.to_dict()
    enabled_without_bayesian = [
        finding.to_dict()
        for finding in enabled.findings
        if finding.details.get("check") != "bayesian_win_probability"
    ]
    assert enabled_without_bayesian == [finding.to_dict() for finding in default.findings]
    assert len(enabled.findings) == len(default.findings) + 1
    assert [finding.details.get("check") for finding in enabled.findings[:4]] == [
        "decision",
        "effect_size",
        "precision",
        "bayesian_win_probability",
    ]


def test_bayesian_advisory_does_not_change_verdict_or_drivers():
    data = make_data([-1.0] * 20)
    disabled = run_audit(
        data, model_a="A", model_b="B", config=AuditConfig(bayesian=False)
    )
    enabled = run_audit(
        data, model_a="A", model_b="B", config=AuditConfig(bayesian=True)
    )

    assert enabled.verdict.level is disabled.verdict.level
    assert [finding.to_dict() for finding in enabled.verdict.drivers] == [
        finding.to_dict() for finding in disabled.verdict.drivers
    ]


def test_preference_only_skip_does_not_change_verdict_or_drivers():
    data = EvalData(
        models=["A", "B"],
        examples=[
            Example(id=str(index), scores={}, preferences={"judge": "A"})
            for index in range(20)
        ],
        source_format="test",
    )
    disabled = run_audit(
        data, model_a="A", model_b="B", config=AuditConfig(bayesian=False)
    )
    enabled = run_audit(
        data, model_a="A", model_b="B", config=AuditConfig(bayesian=True)
    )

    assert enabled.verdict.level is disabled.verdict.level
    assert [finding.to_dict() for finding in enabled.verdict.drivers] == [
        finding.to_dict() for finding in disabled.verdict.drivers
    ]


def test_result_title_renders_in_every_human_report_format():
    report = run_audit(
        make_data([-1.0] * 20),
        model_a="A",
        model_b="B",
        config=AuditConfig(bayesian=True),
    )
    title = by_check(report.findings, "bayesian_win_probability").title

    assert "P(A wins more often than B on decisive examples)" in title
    assert "95% CrI for A win rate" in title
    # Rich wraps long titles mid-token (width= is ignored when TERM=dumb and
    # height is unset). Assert on whitespace-normalized text so the check is
    # about content, not terminal geometry.
    for rendered in (
        render_report(report, width=200),
        render_plain(report),
        render_markdown(report),
        render_html(report),
    ):
        assert _squash(title) in _squash(rendered)


def test_pass_and_skip_findings_follow_the_golden_rule():
    passed = audit_bayesian_win_probability(make_data([-1.0]), "A", "B")[0]
    skipped = audit_bayesian_win_probability(make_data([0.0]), "A", "B")[0]

    for finding in (passed, skipped):
        assert finding.why.strip()
        assert finding.how_detected.strip()
        assert finding.how_to_fix.strip()
    assert "latent decisive-example win rate" in passed.how_detected
    assert "posterior equals the prior" in skipped.how_detected
