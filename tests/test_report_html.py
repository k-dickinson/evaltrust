"""Tests for the HTML report renderer."""

from types import SimpleNamespace

from evaltrust.audit.runner import run_audit
from evaltrust.audit.verdict import VerdictLevel
from evaltrust.core.schema import EvalData, Example, Status
from evaltrust.report.html import render_html


def _make_report(level=VerdictLevel.HIGH, findings=None):
    """Minimal AuditReport stand-in for renderer tests."""
    verdict = SimpleNamespace(
        level=level,
        summary="The gap is statistically significant and practically meaningful.",
    )
    if findings is None:
        findings = [
            SimpleNamespace(
                pillar="Statistical",
                title="Sample size is adequate",
                status=Status.PASS,
                how_to_fix="",
                why="Enough examples to detect a real gap.",
                how_detected="Power analysis at alpha=0.05.",
            ),
            SimpleNamespace(
                pillar="Statistical",
                title="Significance test passed",
                status=Status.WARN,
                how_to_fix="Run more examples to be sure.",
                why="p-value is borderline.",
                how_detected="Permutation test.",
            ),
        ]
    return SimpleNamespace(
        verdict=verdict,
        model_a="gpt-4",
        model_b="claude-3",
        is_single=False,
        n_examples=50,
        source_format="generic",
        models_available=["gpt-4", "claude-3"],
        findings=findings,
    )


def test_html_contains_verdict_level():
    report = _make_report(level=VerdictLevel.HIGH)
    html = render_html(report)
    assert "High Confidence" in html


def test_html_contains_finding_titles():
    report = _make_report()
    html = render_html(report)
    for f in report.findings:
        assert f.title in html


def test_html_is_valid_document():
    html = render_html(_make_report())
    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html


def test_html_explain_includes_why_and_how_detected():
    report = _make_report()
    html = render_html(report, explain=True)
    flagged = [f for f in report.findings if f.status is not Status.PASS]
    for f in flagged:
        assert f.why in html
        assert f.how_detected in html


def test_html_labels_and_explains_an_unbounded_paired_effect_safely():
    data = EvalData(
        models=["A", "B"],
        examples=[
            Example(id=str(i), scores={"A": 1.0, "B": 2.0})
            for i in range(8)
        ],
        source_format="test",
    )
    report = run_audit(data)

    default = render_html(report)
    explained = render_html(report, explain=True)

    label = "Effect size: unbounded (zero variance in gap)"
    assert label in default
    assert label in explained
    assert "every paired gap is the same nonzero value" in explained
    for rendered in (default, explained):
        assert "infinite" not in rendered
        assert "+inf" not in rendered
        assert "-inf" not in rendered


def test_html_low_confidence_verdict():
    report = _make_report(level=VerdictLevel.LOW)
    html = render_html(report)
    assert "Low Confidence" in html
