"""Render an audit report as Markdown.

Built to drop cleanly into a GitHub PR comment or a doc: plain CommonMark,
no HTML, statuses as emoji so a comment reads at a glance.
"""

from __future__ import annotations

from ..audit.runner import AuditReport
from ..audit.verdict import VerdictLevel
from ..core.schema import Status
from .terminal import _grouped, _metric_outcome, _others, _subtitle, _OUTCOME

_STATUS_MARK = {
    Status.PASS: "✅",
    Status.WARN: "⚠️",
    Status.FAIL: "❌",
    Status.SKIP: "➖",
}
_LEVEL_MARK = {
    VerdictLevel.HIGH: "🟢",
    VerdictLevel.MODERATE: "🟡",
    VerdictLevel.LOW: "🔴",
}


def _cell(s: object) -> str:
    """Make a value safe inside a Markdown table cell."""
    return str(s).replace("|", "\\|")


def _bullets(lines: list[str], heading: str, items: list[str]) -> None:
    if not items:
        return
    lines += ["", f"## {heading}", ""]
    lines += [f"- {item}" for item in items]


def render_markdown(report: AuditReport, explain: bool = False) -> str:
    """Render the report as Markdown, mirroring the terminal report's shape."""
    v = report.verdict
    lines = [f"# EvalTrust — {_subtitle(report)}"]
    others = _others(report)
    if others:
        lines += ["", f"*comparing the two strongest of "
                      f"{len(report.models_available)}; others: {', '.join(others)}*"]

    lines += ["", f"## {_LEVEL_MARK[v.level]} {v.level.value}", "", v.summary]

    for pillar, items in _grouped(report.findings).items():
        lines += ["", f"### {pillar}", ""]
        lines += [f"- {_STATUS_MARK[f.status]} {f.title}" for f in items]

    _bullets(lines, "What to do",
             [f.how_to_fix for f in report.findings
              if f.status in (Status.WARN, Status.FAIL)])
    _bullets(lines, "To check more",
             [f.how_to_fix for f in report.findings if f.status is Status.SKIP])

    if explain:
        flagged = [f for f in report.findings if f.status is not Status.PASS]
        if flagged:
            lines += ["", "## Detail"]
            for f in flagged:
                lines += ["", f"### {_STATUS_MARK[f.status]} {f.title}", "",
                          f.why, "", f.how_detected]

    return "\n".join(lines).rstrip() + "\n"


def render_suite_markdown(suite, explain: bool = False) -> str:
    """Render a multi-metric suite as Markdown: verdict, then a metric table."""
    first = next(iter(suite.reports.values()))
    k = len(suite.reports)
    lines = [f"# EvalTrust — {first.model_a} vs {first.model_b} · "
             f"{first.n_examples} examples · {k} metrics"]
    if suite.corrected_alpha != suite.alpha:
        lines += ["", f"*significance corrected for {k} metrics ({suite.correction})*"]

    lvl = suite.overall_level
    lines += ["", f"## {_LEVEL_MARK[lvl]} {lvl.value}", "",
              f"*(weakest of {k} metrics)*", "",
              "| Metric | Confidence | Outcome |", "|---|---|---|"]
    for metric, report in suite.reports.items():
        outcome = _metric_outcome(report)
        label = _OUTCOME.get(outcome, (outcome, ""))[0]
        level = report.verdict.level
        lines.append(f"| {_cell(metric)} | {_LEVEL_MARK[level]} "
                     f"{level.value.split()[0]} | {_cell(label)} |")

    if explain:
        for metric, report in suite.reports.items():
            lines += ["", "---", "", render_markdown(report, explain=True).rstrip()]
    else:
        lines += ["", "*Run a single metric, or add `--explain`, for the full breakdown.*"]

    return "\n".join(lines).rstrip() + "\n"
