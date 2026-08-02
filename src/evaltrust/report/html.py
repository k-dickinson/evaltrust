"""Render an audit report as a standalone HTML page.

No external dependencies: CSS is inlined so the file is self-contained and can be
opened in a browser or attached to a CI artefact.
"""

from __future__ import annotations

import html as _html

from ..audit.runner import AuditReport
from ..audit.verdict import VerdictLevel
from ..core.schema import Finding, Status
from .terminal import (
    _detail_findings,
    _display_how_detected,
    _display_title,
    _grouped,
    _others,
    _subtitle,
)

_STATUS_COLOR = {
    Status.PASS: "#22c55e",
    Status.WARN: "#eab308",
    Status.FAIL: "#ef4444",
    Status.SKIP: "#9ca3af",
}
_STATUS_LABEL = {
    Status.PASS: "PASS",
    Status.WARN: "WARN",
    Status.FAIL: "FAIL",
    Status.SKIP: "SKIP",
}
_VERDICT_COLOR = {
    VerdictLevel.HIGH: "#22c55e",
    VerdictLevel.MODERATE: "#eab308",
    VerdictLevel.LOW: "#ef4444",
}

_CSS = """
  body { font-family: system-ui, sans-serif; max-width: 800px;
         margin: 2rem auto; padding: 0 1rem; color: #1f2937; }
  h1   { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.25rem; }
  .subtitle { color: #6b7280; font-size: 0.9rem; margin-bottom: 1.5rem; }
  .verdict  { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.5rem; }
  .summary  { margin-bottom: 1.5rem; }
  .pillar   { font-weight: 600; margin-top: 1.25rem; margin-bottom: 0.4rem; }
  .finding  { display: flex; align-items: center; gap: 0.5rem;
               margin: 0.2rem 0 0.2rem 1rem; font-size: 0.95rem; }
  .badge    { font-size: 0.75rem; font-weight: 600; padding: 0.1rem 0.4rem;
               border-radius: 4px; color: #fff; }
  .todo     { margin-top: 1.5rem; }
  .todo h2  { font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem; }
  .todo ul  { margin: 0; padding-left: 1.25rem; }
  .todo li  { margin: 0.25rem 0; font-size: 0.95rem; }
  .detail      { margin-top: 1.5rem; }
  .detail h2   { font-size: 1rem; font-weight: 600; }
  .detail-item { margin: 1rem 0 0 1rem; }
  .detail-item .title { font-weight: 600; }
  .detail-item .why,
  .detail-item .how   { color: #6b7280; font-size: 0.9rem; margin: 0.2rem 0; }
"""


def _e(s: object) -> str:
    return _html.escape(str(s))


def render_html_from_parts(
    subtitle: str,
    findings: "list[Finding]",
    *,
    title_suffix: str | None = None,
    verdict_color: str | None = None,
    verdict_label: str | None = None,
    verdict_summary: str | None = None,
    preamble_html: str | None = None,
    explain: bool = False,
) -> str:
    """Return a self-contained HTML page built from a subtitle string and findings.

    This is the shared implementation used by both the standard AuditReport path
    (via ``render_html``) and the run-level path (which has no verdict).  When
    ``verdict_color`` / ``verdict_label`` / ``verdict_summary`` are omitted, the
    verdict block is skipped, keeping run-level output in sync with the main path.
    """
    display_title = f"EvalTrust \u2014 {_e(title_suffix or subtitle)}"

    parts: list[str] = []
    p = parts.append

    p("<!DOCTYPE html>")
    p("<html lang='en'><head>")
    p("<meta charset='utf-8'>")
    p("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    p(f"<title>{display_title}</title>")
    p(f"<style>{_CSS}</style>")
    p("</head><body>")

    p("<h1>EvalTrust</h1>")
    p(f"<div class='subtitle'>{_e(subtitle)}</div>")

    if preamble_html:
        p(preamble_html)

    if verdict_label is not None:
        vc = verdict_color or "#6b7280"
        p(f"<div class='verdict' style='color:{vc}'>\u25cf {_e(verdict_label)}</div>")
        if verdict_summary is not None:
            p(f"<div class='summary'>{_e(verdict_summary)}</div>")

    for pillar, items in _grouped(findings).items():
        p(f"<div class='pillar'>{_e(pillar)}</div>")
        for f in items:
            fc = _STATUS_COLOR[f.status]
            p("<div class='finding'>")
            p(f"  <span class='badge' style='background:{fc}'>"
              f"{_STATUS_LABEL[f.status]}</span>")
            p(f"  {_e(_display_title(f))}")
            p("</div>")

    todo = [f.how_to_fix for f in findings if f.status in (Status.WARN, Status.FAIL)]
    if todo:
        p("<div class='todo'><h2>What to do</h2><ul>")
        for item in todo:
            p(f"  <li>{_e(item)}</li>")
        p("</ul></div>")

    optional = [f.how_to_fix for f in findings if f.status is Status.SKIP]
    if optional:
        p("<div class='todo'><h2>To check more</h2><ul>")
        for item in optional:
            p(f"  <li>{_e(item)}</li>")
        p("</ul></div>")

    if explain:
        flagged = _detail_findings(findings)
        if flagged:
            p("<div class='detail'><h2>Detail</h2>")
            for f in flagged:
                fc = _STATUS_COLOR[f.status]
                p("<div class='detail-item'>")
                p(f"  <div class='title'>"
                  f"<span class='badge' style='background:{fc}'>"
                  f"{_STATUS_LABEL[f.status]}</span> {_e(_display_title(f))}</div>")
                p(f"  <div class='why'>{_e(f.why)}</div>")
                p(f"  <div class='how'>{_e(_display_how_detected(f))}</div>")
                p("</div>")
            p("</div>")

    p("</body></html>")
    return "\n".join(parts)


def render_html(report: AuditReport, explain: bool = False) -> str:
    """Return a self-contained HTML page for *report*."""
    v = report.verdict
    others = _others(report)
    preamble = (
        f"<div class='subtitle'>comparing the two strongest of "
        f"{len(report.models_available)}; "
        f"others: {_e(', '.join(others))}</div>"
        if others else None
    )
    return render_html_from_parts(
        _subtitle(report),
        report.findings,
        verdict_color=_VERDICT_COLOR[v.level],
        verdict_label=v.level.value,
        verdict_summary=v.summary,
        preamble_html=preamble,
        explain=explain,
    )
