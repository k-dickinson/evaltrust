"""EvalTrust: is your eval's result real, or just noise?

Given a score gap between two models, tells you whether it's a real improvement,
big enough to matter, and backed by enough data, as a High / Moderate / Low
verdict. Use the CLI (``evaltrust audit``) or this Python API:

    import evaltrust
    report = evaltrust.audit("results.json")
    print(report.verdict.level)
"""

from .api import audit, audit_suite
from .audit.runner import AuditReport, run_audit
from .audit.suite import SuiteReport
from .audit.verdict import UntrustworthyError, Verdict, VerdictLevel
from .core.schema import EvalData, Example, Finding, Status

__all__ = [
    "audit",
    "audit_suite",
    "run_audit",
    "AuditReport",
    "SuiteReport",
    "UntrustworthyError",
    "Verdict",
    "VerdictLevel",
    "EvalData",
    "Example",
    "Finding",
    "Status",
]
