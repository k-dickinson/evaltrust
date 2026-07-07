"""The overall confidence verdict.

Deliberately *not* a weighted mystery score. The rule is simple and auditable:

  - Any FAIL  -> LOW      (a load-bearing part of the conclusion is unsupported)
  - Any WARN  -> MODERATE (the conclusion holds but with real caveats)
  - All PASS  -> HIGH     (the evidence backs the conclusion)

SKIP findings are evidence you don't have; they never raise confidence. If there
is no usable evidence at all, confidence is LOW by definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..core.schema import Finding, Status


class VerdictLevel(Enum):
    HIGH = "High Confidence"
    MODERATE = "Moderate Confidence"
    LOW = "Low Confidence"


_SUMMARY = {
    VerdictLevel.HIGH: (
        "The evidence supports the conclusion. You can act on this comparison "
        "with confidence."
    ),
    VerdictLevel.MODERATE: (
        "The reported improvement is probably real, but the caveats below should "
        "be addressed before claiming superiority."
    ),
    VerdictLevel.LOW: (
        "The evidence does not support the conclusion. Do not ship on this "
        "result as-is — resolve the issues below first."
    ),
}

_NO_EVIDENCE = (
    "There was not enough evidence in the results to audit the comparison. Add "
    "per-example scores (and ideally repeated runs and a second judge)."
)


@dataclass(frozen=True)
class Verdict:
    level: VerdictLevel
    summary: str
    drivers: list[Finding]  # the findings that determined the level


def compute_verdict(findings: list[Finding]) -> Verdict:
    active = [f for f in findings if f.status is not Status.SKIP]

    if not active:
        return Verdict(VerdictLevel.LOW, _NO_EVIDENCE, [])

    fails = [f for f in active if f.status is Status.FAIL]
    warns = [f for f in active if f.status is Status.WARN]

    if fails:
        level, drivers = VerdictLevel.LOW, fails
    elif warns:
        level, drivers = VerdictLevel.MODERATE, warns
    else:
        level, drivers = VerdictLevel.HIGH, []

    return Verdict(level, _SUMMARY[level], drivers)
