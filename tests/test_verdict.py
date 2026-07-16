"""Tests for the overall confidence verdict."""

from evaltrust.audit.verdict import VerdictLevel, compute_verdict
from evaltrust.core.schema import Finding, Status


def finding(status, title="f"):
    return Finding(pillar="p", title=title, status=status,
                   why="w", how_detected="h", how_to_fix="x", details={})


def test_all_pass_is_high_confidence():
    v = compute_verdict([finding(Status.PASS), finding(Status.PASS)])
    assert v.level is VerdictLevel.HIGH


def test_any_warning_drops_to_moderate():
    v = compute_verdict([finding(Status.PASS), finding(Status.WARN)])
    assert v.level is VerdictLevel.MODERATE


def test_any_failure_is_low_even_with_passes():
    v = compute_verdict([finding(Status.PASS), finding(Status.WARN),
                         finding(Status.FAIL)])
    assert v.level is VerdictLevel.LOW


def test_skips_are_ignored():
    v = compute_verdict([finding(Status.PASS), finding(Status.SKIP)])
    assert v.level is VerdictLevel.HIGH


def test_all_skips_is_low_for_lack_of_evidence():
    v = compute_verdict([finding(Status.SKIP), finding(Status.SKIP)])
    assert v.level is VerdictLevel.LOW


def test_verdict_level_str_returns_value():
    assert str(VerdictLevel.HIGH) == "High Confidence"
    assert str(VerdictLevel.MODERATE) == "Moderate Confidence"
    assert str(VerdictLevel.LOW) == "Low Confidence"


def test_verdict_reports_the_failing_findings():
    fail = finding(Status.FAIL, title="not significant")
    v = compute_verdict([finding(Status.PASS), fail])
    assert fail in v.drivers


def test_moderate_reports_the_warnings():
    warn = finding(Status.WARN, title="small effect")
    v = compute_verdict([finding(Status.PASS), warn])
    assert warn in v.drivers


def test_verdict_has_a_plain_language_summary():
    v = compute_verdict([finding(Status.PASS)])
    assert v.summary.strip()


def decision(outcome, status):
    return Finding(pillar="Statistical Validity", title="t", status=status,
                   why="w", how_detected="h", how_to_fix="x",
                   details={"check": "decision", "outcome": outcome})


def test_equivalence_summary_does_not_claim_an_improvement():
    v = compute_verdict([decision("equivalent", Status.WARN)])
    assert "no real quality difference" in v.summary.lower()
    assert "probably real" not in v.summary.lower()


def test_inconclusive_summary_asks_for_more_evidence():
    v = compute_verdict([decision("inconclusive", Status.FAIL)])
    assert "enough evidence" in v.summary.lower() or "more" in v.summary.lower()
