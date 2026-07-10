"""Audit configuration: one place for a team's statistical policy.

Defaults here, overridable in ``.evaltrust.toml`` or a ``[tool.evaltrust]`` table
in ``pyproject.toml``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, fields
from difflib import get_close_matches
from pathlib import Path
from types import MappingProxyType

try:  # tomllib is stdlib on 3.11+; tomli is the backport for 3.10
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - depends on Python version
    import tomli as _toml


def _unknown_keys_message(unknown: list[str], known: set[str]) -> str:
    """Name each unrecognised config key, with a did-you-mean when one is close."""
    described = []
    for key in sorted(unknown):
        close = get_close_matches(key, sorted(known), n=1)
        described.append(
            f"{key!r} (did you mean {close[0]!r}?)" if close else f"{key!r}")
    plural = "s" if len(described) > 1 else ""
    return (f"Unknown config key{plural}: {', '.join(described)}. "
            "The intended setting is NOT applied.")


def _validate_weights(weights: dict) -> None:
    """Raise ValueError for any weight that is zero, negative, or non-numeric.

    Called at config construction time so that load-time errors surface as
    clean exit-2 messages in the CLI rather than a ZeroDivisionError buried
    inside ``overall_level``.
    """
    for name, w in weights.items():
        if not isinstance(w, (int, float)):
            raise ValueError(
                f"metric_weights: weight for {name!r} must be a number, got {w!r}."
            )
        if w <= 0:
            raise ValueError(
                f"metric_weights: weight for {name!r} must be positive, got {w!r}."
            )
        if not math.isfinite(w):
            raise ValueError(
                f"metric_weights: weight for {name!r} must be finite, got {w!r}."
            )


@dataclass(frozen=True)
class AuditConfig:
    alpha: float = 0.05                     # significance level
    equivalence_margin: float = 0.05        # largest negligible score gap
    power_target: float = 0.8               # target power for sample-size advice
    smallest_meaningful_effect: float = 0.2  # Cohen's d worth powering for
    precision_margin: float = 0.05          # target CI half-width for a single score
    saturation_fraction: float = 0.95       # mean/ceiling that counts as saturated
    min_spread: float = 0.01                # pooled std below which no discrimination
    judge_agreement_threshold: float = 0.8  # inter-judge and binary calibration floor
    judge_correlation_threshold: float = 0.8  # continuous calibration Spearman floor
    reference_judge: str | None = None      # judge treated as ground truth (else auto)
    n_resamples: int = 10_000               # bootstrap / permutation resamples
    seed: int = 0                           # RNG seed (reproducibility)
    correction: str = "bonferroni"          # multi-metric correction: bonferroni | holm | none
    # metrics that must reach HIGH; any below HIGH → suite is LOW immediately
    gated_metrics: frozenset = field(default_factory=frozenset)
    # metric → positive relative weight for weighted-floor rollup; empty = weakest-metric rule
    # Stored as a MappingProxyType so the frozen dataclass is truly immutable and
    # remains hashable (via the __hash__ override below).
    metric_weights: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        # Coerce and validate metric_weights so callers using the constructor
        # directly (not from_dict) get the same guarantees.
        if not isinstance(self.metric_weights, MappingProxyType):
            _validate_weights(dict(self.metric_weights))
            # frozen=True means we can't do self.metric_weights = ...; use object.__setattr__
            object.__setattr__(
                self, "metric_weights", MappingProxyType(dict(self.metric_weights))
            )
        else:
            _validate_weights(dict(self.metric_weights))

        if not isinstance(self.gated_metrics, frozenset):
            if isinstance(self.gated_metrics, str):
                raise ValueError(
                    "gated_metrics must be a list or set of metric names, "
                    f"got a bare string: {self.gated_metrics!r}. "
                    "Did you mean [\"" + self.gated_metrics + "\"]?"
                )
            object.__setattr__(self, "gated_metrics", frozenset(self.gated_metrics))

    def __hash__(self) -> int:
        # MappingProxyType is not hashable by default; convert to a sorted
        # tuple of pairs so identical configs produce the same hash.
        return hash((
            self.alpha,
            self.equivalence_margin,
            self.power_target,
            self.smallest_meaningful_effect,
            self.precision_margin,
            self.saturation_fraction,
            self.min_spread,
            self.judge_agreement_threshold,
            self.reference_judge,
            self.n_resamples,
            self.seed,
            self.correction,
            self.gated_metrics,
            tuple(sorted(self.metric_weights.items())),
        ))

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "AuditConfig":
        """Build a config from a dict.

        Unknown keys are reported — a typo like ``alpah`` or
        ``equivalence-margin`` must not silently revert the team's policy to
        defaults. With ``strict`` they raise ``ValueError``; otherwise they
        warn and are ignored.

        Coerces TOML types to the annotated Python types:
        - ``gated_metrics``: list → frozenset
        - ``metric_weights``: dict → MappingProxyType (with positive-weight validation)
        """
        known = {f.name for f in fields(cls)}
        unknown = [k for k in data if k not in known]
        if unknown:
            message = _unknown_keys_message(unknown, known)
            if strict:
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
        filtered = {k: v for k, v in data.items() if k in known}

        if "gated_metrics" in filtered:
            raw_gates = filtered["gated_metrics"]
            if isinstance(raw_gates, str):
                raise ValueError(
                    "gated_metrics must be a list of metric names, "
                    f"got a bare string: {raw_gates!r}. "
                    "Did you mean [\"" + raw_gates + "\"]?"
                )
            filtered["gated_metrics"] = frozenset(raw_gates)

        if "metric_weights" in filtered:
            raw = dict(filtered["metric_weights"])
            _validate_weights(raw)
            filtered["metric_weights"] = MappingProxyType(raw)

        return cls(**filtered)

    @classmethod
    def load(cls, path: str | None = None, start_dir: str = ".") -> "AuditConfig":
        """Load config from a file.

        With an explicit ``path``, read that TOML file; unknown keys in it are
        an error, since the file was named on purpose. Otherwise look in
        ``start_dir`` for ``.evaltrust.toml`` first, then a ``[tool.evaltrust]``
        table in ``pyproject.toml``; there unknown keys warn but don't fail.
        Falls back to defaults when none is found.
        """
        if path is not None:
            with open(path, "rb") as fh:
                return cls.from_dict(_toml.load(fh), strict=True)

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