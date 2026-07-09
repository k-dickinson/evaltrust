"""Audit configuration: one place for a team's statistical policy.

Defaults here, overridable in ``.evaltrust.toml`` or a ``[tool.evaltrust]`` table
in ``pyproject.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

try:  # tomllib is stdlib on 3.11+; tomli is the backport for 3.10
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - depends on Python version
    import tomli as _toml


@dataclass(frozen=True)
class AuditConfig:
    alpha: float = 0.05                     # significance level
    equivalence_margin: float = 0.05        # largest negligible score gap
    power_target: float = 0.8               # target power for sample-size advice
    smallest_meaningful_effect: float = 0.2  # Cohen's d worth powering for
    precision_margin: float = 0.05          # target CI half-width for a single score
    saturation_fraction: float = 0.95       # mean/ceiling that counts as saturated
    min_spread: float = 0.01                # pooled std below which no discrimination
    judge_agreement_threshold: float = 0.8  # inter-judge agreement to pass
    reference_judge: str | None = None      # judge treated as ground truth (else auto)
    n_resamples: int = 10_000               # bootstrap / permutation resamples
    seed: int = 0                           # RNG seed (reproducibility)
    correction: str = "bonferroni"          # multi-metric correction: bonferroni | holm | none

    @classmethod
    def from_dict(cls, data: dict) -> "AuditConfig":
        """Build a config from a dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path: str | None = None, start_dir: str = ".") -> "AuditConfig":
        """Load config from a file.

        With an explicit ``path``, read that TOML file. Otherwise look in
        ``start_dir`` for ``.evaltrust.toml`` first, then a ``[tool.evaltrust]``
        table in ``pyproject.toml``. Falls back to defaults when none is found.
        """
        if path is not None:
            with open(path, "rb") as fh:
                return cls.from_dict(_toml.load(fh))

        base = Path(start_dir)
        dedicated = base / ".evaltrust.toml"
        if dedicated.exists():
            with open(dedicated, "rb") as fh:
                return cls.from_dict(_toml.load(fh))

        pyproject = base / "pyproject.toml"
        if pyproject.exists():
            with open(pyproject, "rb") as fh:
                table = _toml.load(fh).get("tool", {}).get("evaltrust")
            if table:
                return cls.from_dict(table)

        return cls()
