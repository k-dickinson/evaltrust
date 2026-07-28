"""EvalTrust: is your eval's result real, or just noise?

Given a score gap between two models, tells you whether it's a real improvement,
big enough to matter, and backed by enough data, as a High / Moderate / Low
verdict. Use the CLI (``evaltrust audit``) or this Python API:

    import evaltrust
    report = evaltrust.audit("results.json")
    print(report.verdict.level)
"""

from importlib.metadata import PackageNotFoundError, version

from .api import audit, audit_contamination, audit_run_level, audit_suite
from .audit.contamination import ContaminationResult
from .audit.runner import AuditReport, run_audit
from .audit.suite import SuiteReport
from .audit.verdict import UntrustworthyError, Verdict, VerdictLevel
from .config import AuditConfig
from .core.schema import EvalData, Example, Finding, Preference, RunLevelData, Status
from .versions import METHODOLOGY_VERSION, SCHEMA_VERSION

try:
    __version__ = version("evaltrust")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "METHODOLOGY_VERSION",
    # --- top-level entry points ---
    "audit",
    "audit_run_level",
    "audit_contamination",
    "audit_suite",
    "run_audit",
    # --- config ---
    "AuditConfig",
    # --- report types ---
    "AuditReport",
    "SuiteReport",
    "ContaminationResult",
    # --- verdict ---
    "UntrustworthyError",
    "Verdict",
    "VerdictLevel",
    # --- schema ---
    "EvalData",
    "RunLevelData",
    "Example",
    "Finding",
    "Preference",
    "Status",
]
