"""Compare two audits (JSON from ``--json``) to catch regressions between runs.

Flags where confidence dropped or a real win was lost since the last run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_LEVEL_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
# For "is the leader better?", significant > equivalent > inconclusive as evidence.
_OUTCOME_RANK = {"inconclusive": 0, "equivalent": 1, "significant": 2}


@dataclass(frozen=True)
class Change:
    scope: str        # "overall" or a metric name
    field: str        # "confidence" or "decision"
    old: str
    new: str
    regression: bool
    improvement: bool


@dataclass(frozen=True)
class DiffReport:
    changes: list[Change] = field(default_factory=list)

    @property
    def regressions(self) -> list[Change]:
        return [c for c in self.changes if c.regression]

    @property
    def has_regression(self) -> bool:
        return any(c.regression for c in self.changes)


def _decision_outcome(report: dict) -> str | None:
    for f in report.get("findings", []):
        if f.get("details", {}).get("check") == "decision":
            return f["details"].get("outcome")
    return None


def _ranked_change(scope, field_name, old, new, ranks) -> Change | None:
    if old is None or new is None or old == new:
        return None
    old_r, new_r = ranks.get(old, -1), ranks.get(new, -1)
    return Change(scope, field_name, old, new,
                  regression=new_r < old_r, improvement=new_r > old_r)


def _compare_report(scope, old, new, changes) -> None:
    c = _ranked_change(scope, "confidence",
                       old["verdict"]["level"], new["verdict"]["level"], _LEVEL_RANK)
    if c:
        changes.append(c)
    c = _ranked_change(scope, "decision",
                       _decision_outcome(old), _decision_outcome(new), _OUTCOME_RANK)
    if c:
        changes.append(c)


def compare(old: dict, new: dict) -> DiffReport:
    old_suite, new_suite = "metrics" in old, "metrics" in new
    if old_suite != new_suite:
        raise ValueError(
            "Can't compare a single-metric audit with a multi-metric suite.")

    changes: list[Change] = []
    if old_suite:
        c = _ranked_change("overall", "confidence",
                           old["overall_level"], new["overall_level"], _LEVEL_RANK)
        if c:
            changes.append(c)
        seen = list(old["metrics"]) + [m for m in new["metrics"]
                                       if m not in old["metrics"]]
        for m in seen:
            if m in old["metrics"] and m in new["metrics"]:
                _compare_report(m, old["metrics"][m], new["metrics"][m], changes)
    else:
        _compare_report("overall", old, new, changes)

    return DiffReport(changes)
