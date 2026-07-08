"""Tests for the Markdown report renderer."""

from types import SimpleNamespace

from evaltrust.audit.verdict import VerdictLevel
from evaltrust.core.schema import Status
from evaltrust.report.markdown import render_markdown


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
                title="Significance test is borderline",
                status=Status.WARN,
                how_to_fix="Run more examples to be sure.",
                why="p-value is borderline.",
                how_detected="Permutation test.",
            ),
            SimpleNamespace(
                pillar="Repeatability",
                title="No repeated runs found",
                status=Status.SKIP,
                how_to_fix="Rerun the eval to measure stability.",
                why="Stability cannot be checked without reruns.",
                how_detected="Looked for repeated example ids.",
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


def test_markdown_contains_verdict_level():
    md = render_markdown(_make_report(level=VerdictLevel.HIGH))
    assert "High Confidence" in md


def test_markdown_contains_finding_titles():
    report = _make_report()
    md = render_markdown(report)
    for f in report.findings:
        assert f.title in md


def test_markdown_starts_with_a_heading():
    md = render_markdown(_make_report())
    assert md.startswith("# EvalTrust")
    assert md.endswith("\n")


def test_markdown_groups_findings_by_pillar():
    md = render_markdown(_make_report())
    assert "### Statistical" in md
    assert "### Repeatability" in md


def test_markdown_marks_statuses():
    md = render_markdown(_make_report())
    assert "✅" in md   # PASS
    assert "⚠️" in md  # WARN


def test_markdown_lists_fixes_for_flagged_findings():
    report = _make_report()
    md = render_markdown(report)
    assert "What to do" in md
    assert "Run more examples to be sure." in md
    # SKIPs land under the optional section, like the terminal report.
    assert "To check more" in md
    assert "Rerun the eval to measure stability." in md


def test_markdown_explain_includes_why_and_how_detected():
    report = _make_report()
    md = render_markdown(report, explain=True)
    flagged = [f for f in report.findings if f.status is not Status.PASS]
    for f in flagged:
        assert f.why in md
        assert f.how_detected in md


def test_markdown_without_explain_omits_detail():
    report = _make_report()
    md = render_markdown(report)
    assert "p-value is borderline." not in md


def test_markdown_low_confidence_verdict():
    md = render_markdown(_make_report(level=VerdictLevel.LOW))
    assert "Low Confidence" in md
