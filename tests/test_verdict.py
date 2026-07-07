"""Tests for the overall confidence verdict."""

from evallab.audit.verdict import VerdictLevel, compute_verdict
from evallab.core.schema import Finding, Status


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
