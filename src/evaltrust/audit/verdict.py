"""The overall confidence verdict.

Any FAIL -> LOW, any WARN -> MODERATE, all PASS -> HIGH. SKIP never raises
confidence; with no usable evidence, confidence is LOW.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..core.schema import Finding, Status


class VerdictLevel(Enum):
    HIGH = "High Confidence"
    MODERATE = "Moderate Confidence"
    LOW = "Low Confidence"


_LEVEL_RANK = {VerdictLevel.LOW: 0, VerdictLevel.MODERATE: 1, VerdictLevel.HIGH: 2}


class UntrustworthyError(AssertionError):
    """Raised by ``raise_if_below`` when confidence is under the required level.

    Subclasses AssertionError so it reads as a clean failure in test frameworks.
    """


def coerce_level(level: "str | VerdictLevel") -> VerdictLevel:
    if isinstance(level, VerdictLevel):
        return level
    try:
        return VerdictLevel[str(level).strip().upper()]
    except KeyError:
        raise ValueError(
            f"Unknown confidence level {level!r}; use 'high', 'moderate', or 'low'."
        )


def enforce_level(actual: VerdictLevel, minimum, context: str = "") -> None:
    """Raise UntrustworthyError if ``actual`` is below ``minimum``."""
    minimum = coerce_level(minimum)
    if _LEVEL_RANK[actual] < _LEVEL_RANK[minimum]:
        where = f" for {context}" if context else ""
        raise UntrustworthyError(
            f"Evaluation is {actual.value}{where}, below the required "
            f"{minimum.value}."
        )


_SUMMARY = {
    VerdictLevel.HIGH: "The result holds up. You can act on it.",
    VerdictLevel.MODERATE: "Probably real, but check the flags below first.",
    VerdictLevel.LOW: "Don't ship on this. The problems below undercut the result.",
}

_NO_EVIDENCE = (
    "Not enough in the file to audit. Add per-example scores, and ideally "
    "repeated runs and a second judge."
)

# When the comparison reaches a specific outcome, the summary reflects it rather
# than assuming there was an improvement to defend.
_OUTCOME_SUMMARY = {
    "equivalent": "No real quality difference between the two. Pick on cost or speed.",
    "inconclusive": "Not enough data to call a winner. Collect more before deciding.",
}


def _decision_outcome(findings: list[Finding]) -> str | None:
    for f in findings:
        if f.details.get("check") == "decision":
            return f.details.get("outcome")
    return None


def _find(findings, check):
    for f in findings:
        if f.details.get("check") == check:
            return f
    return None


def _single_summary(findings: list[Finding]) -> str | None:
    """Summary for a single-model audit (score reliability, not a comparison)."""
    thresh = _find(findings, "threshold")
    if thresh is not None:
        return {
            "above": "The model clears the target. You can rely on it.",
            "below": "The model is below the target. It's not ready against this bar.",
            "inconclusive": ("Too close to the target to be sure. Collect more "
                             "examples before deciding."),
        }.get(thresh.details.get("outcome"))
    prec = _find(findings, "single_precision")
    if prec is not None:
        return ("The score is measured precisely enough to trust."
                if prec.details.get("precise")
                else "The score is too imprecise to trust yet. Collect more examples.")
    return None


@dataclass(frozen=True)
class Verdict:
    level: VerdictLevel
    summary: str
    drivers: list[Finding]  # the findings that determined the level

    def to_dict(self) -> dict:
        return {
            "level": self.level.name,
            "label": self.level.value,
            "summary": self.summary,
            "drivers": [f.title for f in self.drivers],
        }


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

    outcome = _decision_outcome(findings)
    summary = (_OUTCOME_SUMMARY.get(outcome)
               or _single_summary(findings)
               or _SUMMARY[level])
    return Verdict(level, summary, drivers)
