"""Tests for the terminal report rendering."""

from evaltrust.audit.runner import run_audit
from evaltrust.audit.suite import audit_suite
from evaltrust.core.schema import EvalData, Example
from evaltrust.report.terminal import render_markdown, render_report, render_suite_markdown


def make_data(scores_by_model, n):
    examples = [
        Example(id=str(i), scores={m: float(s[i]) for m, s in scores_by_model.items()})
        for i in range(n)
    ]
    return EvalData(models=list(scores_by_model), examples=examples,
                    source_format="test", metadata={})


def test_report_shows_header_and_models():
    report = run_audit(make_data({"A": [0] * 50, "B": [1] * 50}, 50))
    out = render_report(report)
    assert "EvalTrust" in out
    assert "A" in out and "B" in out


def test_clean_win_report_shows_high_confidence():
    report = run_audit(make_data({"A": [0] * 200, "B": [1] * 180 + [0] * 20}, 200))
    out = render_report(report)
    assert "High Confidence" in out


def test_noise_report_shows_low_confidence_and_the_failing_title():
    report = run_audit(make_data({"A": [0, 1] * 60, "B": [1, 0] * 60}, 120))
    out = render_report(report)
    assert "Low Confidence" in out
    assert "inconclusive" in out.lower()


def test_report_lists_every_pillar():
    report = run_audit(make_data({"A": [0] * 40, "B": [1] * 40}, 40))
    out = render_report(report)
    for pillar in ("Statistical Validity", "Benchmark Health",
                   "Repeatability", "Judge Reliability"):
        assert pillar in out


def test_report_surfaces_actions_for_flagged_findings():
    # A small, underpowered win yields WARNs whose fixes appear under "What to do".
    report = run_audit(make_data({"A": [0] * 8, "B": [1, 0] * 4}, 8))
    out = render_report(report)
    assert "What to do" in out


def test_explain_adds_detail_and_default_omits_it():
    report = run_audit(make_data({"A": [0, 1] * 60, "B": [1, 0] * 60}, 120))
    assert "Detail" not in render_report(report)
    assert "Detail" in render_report(report, explain=True)


def test_skip_guidance_shown_under_to_check_more():
    report = run_audit(make_data({"A": [0] * 40, "B": [1] * 36 + [0] * 4}, 40))
    out = render_report(report)
    assert "To check more" in out


def test_markdown_report_contains_verdict_and_findings():
    report = run_audit(make_data({"A": [0, 1] * 60, "B": [1, 0] * 60}, 120))
    out = render_markdown(report)
    assert out.startswith("# EvalTrust")
    assert "## Low Confidence" in out
    assert "### Statistical Validity" in out
    assert "inconclusive" in out.lower()


def test_markdown_explain_includes_detail():
    report = run_audit(make_data({"A": [0, 1] * 60, "B": [1, 0] * 60}, 120))
    assert "### Detail" not in render_markdown(report)
    assert "### Detail" in render_markdown(report, explain=True)


def test_suite_markdown_neutralises_metric_names():
    """Metric names come from the eval file and must not break the Markdown.

    A newline would start a new heading and a bare tag would render as raw HTML
    once the report is pasted into a PR comment.
    """
    hostile = "tone\n# Injected\n<img src=x>"
    suite = {hostile: make_data({"A": [0, 1] * 60, "B": [1, 0] * 60}, 120),
             "ok": make_data({"A": [0] * 80, "B": [1] * 80}, 80)}
    out = render_suite_markdown(audit_suite(suite, seed=0), explain=True)

    assert out.count("# EvalTrust") == 1
    assert "\n# Injected" not in out
    assert "<img" not in out
    assert "## tone # Injected &lt;img src=x&gt;" in out
    # every table row stays on one line
    rows = [ln for ln in out.splitlines() if ln.startswith("|")]
    assert len(rows) == 4  # header, separator, two metrics


def test_suite_markdown_explain_nests_metric_reports():
    suite = {
        "quality|safety": make_data({"A": [0, 1] * 60, "B": [1, 0] * 60}, 120),
        "tone": make_data({"A": [0] * 80, "B": [1] * 80}, 80),
    }
    report = audit_suite(suite, seed=0)
    out = render_suite_markdown(report, explain=True)
    assert out.count("# EvalTrust") == 1
    assert "| quality\\|safety |" in out
    assert "## quality|safety" in out
    assert "### " in out
