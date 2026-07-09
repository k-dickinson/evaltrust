"""Render an audit report as a standalone HTML page.

No external dependencies: CSS is inlined so the file is self-contained and can be
opened in a browser or attached to a CI artefact.
"""

from __future__ import annotations

import html as _html

from ..audit.runner import AuditReport
from ..audit.verdict import VerdictLevel
from ..core.schema import Status
from .terminal import _grouped, _others, _subtitle

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


def render_html(report: AuditReport, explain: bool = False) -> str:
    """Return a self-contained HTML page for *report*."""
    v = report.verdict
    vc = _VERDICT_COLOR[v.level]

    parts: list[str] = []
    p = parts.append

    p("<!DOCTYPE html>")
    p("<html lang='en'><head>")
    p("<meta charset='utf-8'>")
    p("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    p(f"<title>EvalTrust \u2014 {_e(_subtitle(report))}</title>")
    p(f"<style>{_CSS}</style>")
    p("</head><body>")

    p("<h1>EvalTrust</h1>")
    p(f"<div class='subtitle'>{_e(_subtitle(report))}</div>")

    others = _others(report)
    if others:
        p(f"<div class='subtitle'>comparing the two strongest of "
          f"{len(report.models_available)}; "
          f"others: {_e(', '.join(others))}</div>")

    p(f"<div class='verdict' style='color:{vc}'>\u25cf {_e(v.level.value)}</div>")
    p(f"<div class='summary'>{_e(v.summary)}</div>")

    for pillar, items in _grouped(report.findings).items():
        p(f"<div class='pillar'>{_e(pillar)}</div>")
        for f in items:
            fc = _STATUS_COLOR[f.status]
            p("<div class='finding'>")
            p(f"  <span class='badge' style='background:{fc}'>"
              f"{_STATUS_LABEL[f.status]}</span>")
            p(f"  {_e(f.title)}")
            p("</div>")

    todo = [f.how_to_fix for f in report.findings
            if f.status in (Status.WARN, Status.FAIL)]
    if todo:
        p("<div class='todo'><h2>What to do</h2><ul>")
        for item in todo:
            p(f"  <li>{_e(item)}</li>")
        p("</ul></div>")

    optional = [f.how_to_fix for f in report.findings if f.status is Status.SKIP]
    if optional:
        p("<div class='todo'><h2>To check more</h2><ul>")
        for item in optional:
            p(f"  <li>{_e(item)}</li>")
        p("</ul></div>")

    if explain:
        flagged = [f for f in report.findings if f.status is not Status.PASS]
        if flagged:
            p("<div class='detail'><h2>Detail</h2>")
            for f in flagged:
                fc = _STATUS_COLOR[f.status]
                p("<div class='detail-item'>")
                p(f"  <div class='title'>"
                  f"<span class='badge' style='background:{fc}'>"
                  f"{_STATUS_LABEL[f.status]}</span> {_e(f.title)}</div>")
                p(f"  <div class='why'>{_e(f.why)}</div>")
                p(f"  <div class='how'>{_e(f.how_detected)}</div>")
                p("</div>")
            p("</div>")

    p("</body></html>")
    return "\n".join(parts)
