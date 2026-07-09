"""Render an audit report to the terminal.

Verdict, checks grouped by pillar, then what to do. Per-flag reasoning is one
``--explain`` away.
"""

from __future__ import annotations

import io
from collections import OrderedDict

from rich.console import Console
from rich.text import Text

from ..audit.runner import AuditReport
from ..audit.verdict import VerdictLevel
from ..core.schema import Finding, Status

_SYMBOL = {
    Status.PASS: ("✓", "green"),
    Status.WARN: ("!", "yellow"),
    Status.FAIL: ("✗", "red"),
    Status.SKIP: ("–", "dim"),
}
_PLAIN_MARK = {Status.PASS: "ok  ", Status.WARN: "warn",
               Status.FAIL: "fail", Status.SKIP: "--  "}
_DOT = {VerdictLevel.HIGH: "green", VerdictLevel.MODERATE: "yellow",
        VerdictLevel.LOW: "red"}


def _grouped(findings) -> "OrderedDict[str, list[Finding]]":
    groups: OrderedDict[str, list[Finding]] = OrderedDict()
    for f in findings:
        groups.setdefault(f.pillar, []).append(f)
    return groups


def _subtitle(report: AuditReport) -> str:
    who = report.model_a if report.is_single else f"{report.model_a} vs {report.model_b}"
    return f"{who} · {report.n_examples} examples · {report.source_format}"


def _others(report: AuditReport) -> list[str]:
    if report.is_single:
        return []
    return [m for m in report.models_available
            if m not in (report.model_a, report.model_b)]


# --------------------------------------------------------------------------- #
# Rich (colour) rendering
# --------------------------------------------------------------------------- #

def _renderable(report: AuditReport, explain: bool = False) -> Text:
    v = report.verdict
    t = Text()

    t.append("EvalTrust  ", style="bold")
    t.append(_subtitle(report) + "\n", style="dim")
    others = _others(report)
    if others:
        t.append(f"comparing the two strongest of {len(report.models_available)}; "
                 f"others: {', '.join(others)}\n", style="dim")

    t.append("\n")
    t.append("● ", style=_DOT[v.level])
    t.append(f"{v.level.value}\n", style=f"bold {_DOT[v.level]}")
    t.append(v.summary + "\n")

    for pillar, items in _grouped(report.findings).items():
        t.append(f"\n{pillar}\n", style="bold")
        for f in items:
            sym, color = _SYMBOL[f.status]
            t.append(f"  {sym} ", style=color)
            t.append(f.title, style=("dim" if f.status is Status.SKIP else ""))
            t.append("\n")

    todo = [f.how_to_fix for f in report.findings
            if f.status in (Status.WARN, Status.FAIL)]
    _bullets(t, "What to do", todo)

    optional = [f.how_to_fix for f in report.findings if f.status is Status.SKIP]
    _bullets(t, "To check more", optional, style="dim")

    if explain:
        flagged = [f for f in report.findings if f.status is not Status.PASS]
        if flagged:
            t.append("\nDetail\n", style="bold")
            for f in flagged:
                sym, color = _SYMBOL[f.status]
                t.append(f"\n  {sym} ", style=color)
                t.append(f"{f.title}\n", style="bold")
                t.append(f"    {f.why}\n", style="dim")
                t.append(f"    {f.how_detected}\n", style="dim")
    return t


def _bullets(t: Text, heading: str, items: list[str], style: str = "") -> None:
    if not items:
        return
    t.append(f"\n{heading}\n", style=(f"bold {style}" if style else "bold"))
    for item in items:
        t.append("  • ", style=style or None)
        t.append(item + "\n", style=style or None)


def render_report(report: AuditReport, explain: bool = False, width: int = 90) -> str:
    """Render the report to a plain string (used for tests and piping)."""
    console = Console(record=True, width=width, file=io.StringIO())
    console.print(_renderable(report, explain=explain))
    return console.export_text()


def print_report(report: AuditReport, explain: bool = False) -> None:
    """Print the report to the real terminal with colour."""
    Console().print(_renderable(report, explain=explain))


# --------------------------------------------------------------------------- #
# Plain ASCII rendering
# --------------------------------------------------------------------------- #

_ASCII = str.maketrans({
    "·": "-", "–": "-", "—": "-", "•": "*", "●": "*",
    "’": "'", "‘": "'", "“": '"', "”": '"', "×": "x",
})


# --------------------------------------------------------------------------- #
# Multi-metric suite rendering
# --------------------------------------------------------------------------- #

_OUTCOME = {
    "significant": ("real improvement", "green"),
    "equivalent": ("no difference", "yellow"),
    "inconclusive": ("inconclusive", "red"),
}


def _metric_outcome(report: AuditReport) -> str:
    for f in report.findings:
        if f.details.get("check") == "decision":
            return f.details.get("outcome", "")
    return ""


def _suite_header(suite):
    first = next(iter(suite.reports.values()))
    return (first.model_a, first.model_b, first.n_examples, len(suite.reports))


def _suite_renderable(suite, explain: bool = False) -> Text:
    a, b, n, k = _suite_header(suite)
    t = Text()
    t.append("EvalTrust  ", style="bold")
    t.append(f"{a} vs {b} · {n} examples · {k} metrics\n", style="dim")
    if suite.corrected_alpha != suite.alpha:
        t.append(f"significance corrected for {k} metrics "
                 f"({suite.correction})\n", style="dim")

    lvl = suite.overall_level
    t.append("\n● ", style=_DOT[lvl])
    t.append(f"{lvl.value}", style=f"bold {_DOT[lvl]}")
    t.append(f"  (weakest of {k} metrics)\n\n", style="dim")

    width = max(len(m) for m in suite.reports)
    for metric, report in suite.reports.items():
        outcome = _metric_outcome(report)
        label, color = _OUTCOME.get(outcome, (outcome, ""))
        t.append(f"  {metric.ljust(width)}  ")
        t.append(f"{report.verdict.level.value.split()[0].ljust(9)}", style=_DOT[report.verdict.level])
        t.append(label + "\n", style=color)

    if explain:
        for metric, report in suite.reports.items():
            t.append(f"\n{'─' * 3} {metric} {'─' * 3}\n", style="bold")
            t.append(_renderable(report, explain=True))
    else:
        t.append("\nRun a single metric, or add --explain, for the full breakdown.\n",
                 style="dim")
    return t


def render_suite(suite, explain: bool = False, width: int = 90) -> str:
    console = Console(record=True, width=width, file=io.StringIO())
    console.print(_suite_renderable(suite, explain=explain))
    return console.export_text()


def print_suite(suite, explain: bool = False) -> None:
    Console().print(_suite_renderable(suite, explain=explain))


def render_suite_plain(suite, explain: bool = False) -> str:
    a, b, n, k = _suite_header(suite)
    lines = [f"EvalTrust  {a} vs {b} - {n} examples - {k} metrics"]
    if suite.corrected_alpha != suite.alpha:
        lines.append(f"significance corrected for {k} metrics ({suite.correction})")
    lines += ["", f"{suite.overall_level.value.upper()} (weakest of {k} metrics)", ""]

    width = max(len(m) for m in suite.reports)
    for metric, report in suite.reports.items():
        outcome = _OUTCOME.get(_metric_outcome(report), (_metric_outcome(report), ""))[0]
        level = report.verdict.level.value.split()[0]
        lines.append(f"  {metric.ljust(width)}  {level.ljust(9)} {outcome}")

    if explain:
        for metric, report in suite.reports.items():
            lines += ["", f"=== {metric} ===", render_plain(report, explain=True).rstrip()]

    return ("\n".join(lines).rstrip() + "\n").translate(_ASCII)


# --------------------------------------------------------------------------- #
# Diff (regression) rendering
# --------------------------------------------------------------------------- #

def _diff_renderable(diff) -> Text:
    t = Text()
    t.append("EvalTrust  audit comparison\n", style="bold")
    if not diff.changes:
        t.append("No change between the two audits.\n", style="green")
        return t

    regs = [c for c in diff.changes if c.regression]
    imps = [c for c in diff.changes if c.improvement]
    neutral = [c for c in diff.changes if not c.regression and not c.improvement]

    if regs:
        t.append("\nRegressions\n", style="bold red")
        for c in regs:
            t.append("  ✗ ", style="red")
            t.append(f"{c.scope} {c.field}: {c.old} → {c.new}\n")
    if imps:
        t.append("\nImprovements\n", style="bold green")
        for c in imps:
            t.append("  ✓ ", style="green")
            t.append(f"{c.scope} {c.field}: {c.old} → {c.new}\n")
    for c in neutral:
        t.append(f"  · {c.scope} {c.field}: {c.old} → {c.new}\n", style="dim")
    return t


def render_diff(diff, width: int = 90) -> str:
    console = Console(record=True, width=width, file=io.StringIO())
    console.print(_diff_renderable(diff))
    return console.export_text()


def print_diff(diff) -> None:
    Console().print(_diff_renderable(diff))


def render_diff_plain(diff) -> str:
    lines = ["EvalTrust audit comparison"]
    if not diff.changes:
        return "EvalTrust audit comparison\nNo change between the two audits.\n"
    regs = [c for c in diff.changes if c.regression]
    imps = [c for c in diff.changes if c.improvement]
    if regs:
        lines += ["", "Regressions"] + [
            f"  [worse] {c.scope} {c.field}: {c.old} -> {c.new}" for c in regs]
    if imps:
        lines += ["", "Improvements"] + [
            f"  [better] {c.scope} {c.field}: {c.old} -> {c.new}" for c in imps]
    return ("\n".join(lines).rstrip() + "\n").translate(_ASCII)


def render_plain(report: AuditReport, explain: bool = False) -> str:
    """Render the report as plain ASCII, safe for Windows, CI logs, and pipes."""
    v = report.verdict
    lines = ["EvalTrust  " + _subtitle(report)]
    others = _others(report)
    if others:
        lines.append(f"comparing the two strongest of "
                     f"{len(report.models_available)}; others: {', '.join(others)}")
    lines += ["", f"{v.level.value.upper()}", v.summary]

    for pillar, items in _grouped(report.findings).items():
        lines.append("")
        lines.append(pillar)
        for f in items:
            lines.append(f"  [{_PLAIN_MARK[f.status]}] {f.title}")

    todo = [f.how_to_fix for f in report.findings
            if f.status in (Status.WARN, Status.FAIL)]
    if todo:
        lines += ["", "What to do"] + [f"  - {x}" for x in todo]

    optional = [f.how_to_fix for f in report.findings if f.status is Status.SKIP]
    if optional:
        lines += ["", "To check more"] + [f"  - {x}" for x in optional]

    if explain:
        flagged = [f for f in report.findings if f.status is not Status.PASS]
        if flagged:
            lines += ["", "Detail"]
            for f in flagged:
                lines += [f"  [{_PLAIN_MARK[f.status]}] {f.title}",
                          f"    {f.why}", f"    {f.how_detected}"]

    return ("\n".join(lines).rstrip() + "\n").translate(_ASCII)


# --------------------------------------------------------------------------- #
# Markdown rendering (PR comments / docs)
# --------------------------------------------------------------------------- #

_MD_MARK = {Status.PASS: "pass", Status.WARN: "warn",
            Status.FAIL: "fail", Status.SKIP: "skip"}


def render_markdown(report: AuditReport, explain: bool = False) -> str:
    """Render the report as Markdown, suitable for PR comments and docs."""
    v = report.verdict
    lines = [f"# EvalTrust", "", f"**{_subtitle(report)}**", ""]
    others = _others(report)
    if others:
        lines.append(
            f"_Comparing the two strongest of {len(report.models_available)}; "
            f"others: {', '.join(others)}_"
        )
        lines.append("")
    lines += [f"## {v.level.value}", "", v.summary, ""]

    for pillar, items in _grouped(report.findings).items():
        lines.append(f"### {pillar}")
        lines.append("")
        for f in items:
            lines.append(f"- **[{_MD_MARK[f.status]}]** {f.title}")
        lines.append("")

    todo = [f.how_to_fix for f in report.findings
            if f.status in (Status.WARN, Status.FAIL)]
    if todo:
        lines += ["## What to do", ""] + [f"- {x}" for x in todo] + [""]

    optional = [f.how_to_fix for f in report.findings if f.status is Status.SKIP]
    if optional:
        lines += ["## To check more", ""] + [f"- {x}" for x in optional] + [""]

    if explain:
        flagged = [f for f in report.findings if f.status is not Status.PASS]
        if flagged:
            lines += ["## Detail", ""]
            for f in flagged:
                lines += [f"### [{_MD_MARK[f.status]}] {f.title}", "",
                          f"{f.why}", "", f"{f.how_detected}", ""]

    return "\n".join(lines).rstrip() + "\n"


def render_suite_markdown(suite, explain: bool = False) -> str:
    """Render a multi-metric suite as Markdown."""
    a, b, n, k = _suite_header(suite)
    lines = [
        "# EvalTrust",
        "",
        f"**{a} vs {b} · {n} examples · {k} metrics**",
        "",
    ]
    if suite.corrected_alpha != suite.alpha:
        lines += [
            f"_Significance corrected for {k} metrics ({suite.correction})_",
            "",
        ]
    lines += [
        f"## {suite.overall_level.value}",
        "",
        f"Weakest of {k} metrics.",
        "",
        "| Metric | Level | Outcome |",
        "| --- | --- | --- |",
    ]
    for metric, report in suite.reports.items():
        outcome = _OUTCOME.get(_metric_outcome(report), (_metric_outcome(report), ""))[0]
        level = report.verdict.level.value
        lines.append(f"| {metric} | {level} | {outcome} |")
    lines.append("")

    if explain:
        for metric, report in suite.reports.items():
            nested = render_markdown(report, explain=True)
            nested_lines = nested.splitlines()
            # Drop the outer H1 from nested to avoid multiple top-level titles
            if nested_lines and nested_lines[0].startswith("# "):
                nested_lines = nested_lines[1:]
            while nested_lines and nested_lines[0] == "":
                nested_lines = nested_lines[1:]
            lines += [f"## {metric}", ""] + nested_lines + [""]

    return "\n".join(lines).rstrip() + "\n"
