"""The canonical data model.

Every eval platform's output is mapped into these types by an adapter, and every
audit check reads only these types. This one-format-in-the-middle design is what
lets the statistics be written once and work for DeepEval, Promptfoo, LangSmith,
OpenEvals, or a plain CSV alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Status(Enum):
    """Outcome of a single audit check."""

    PASS = "pass"    # evidence supports trusting the conclusion
    WARN = "warn"    # a caveat that weakens confidence
    FAIL = "fail"    # the conclusion is not supported by the evidence
    SKIP = "skip"    # the data needed for this check is absent

    def __str__(self) -> str:
        return self.value


class Preference(Enum):
    """A non-model outcome for a pairwise preference judgment."""

    TIE = "tie"


@dataclass(frozen=True)
class Example:
    """One evaluated item and its score(s).

    ``scores`` maps model -> final score. The optional fields carry the extra
    evidence that unlocks more checks:
      - ``runs``:   model -> list of scores from repeated evaluations.
      - ``judges``: judge -> {model -> score} when several judges scored the item.
      - ``preferences``: judge -> winning model id, or ``Preference.TIE``.
      - ``attributes``: free-form tags for slicing (e.g. category, difficulty,
        language). Values are stored as strings.
    """

    id: str
    scores: dict[str, float]
    runs: dict[str, list[float]] | None = None
    judges: dict[str, dict[str, float]] | None = None
    preferences: dict[str, str | Preference] | None = None
    attributes: dict[str, str] | None = None


@dataclass(frozen=True)
class EvalData:
    """A whole evaluation, normalised into canonical form."""

    models: list[str]
    examples: list[Example]
    source_format: str
    metadata: dict = field(default_factory=dict)

    @property
    def n_examples(self) -> int:
        return len(self.examples)

    @property
    def has_runs(self) -> bool:
        return any(ex.runs for ex in self.examples)

    @property
    def has_judges(self) -> bool:
        return any(ex.judges for ex in self.examples)

    @property
    def has_preferences(self) -> bool:
        return any(ex.preferences for ex in self.examples)

    def paired_scores(
        self, model_a: str, model_b: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-example scores for two models, aligned over examples that have both.

        Examples missing either model's score are dropped so the comparison stays
        genuinely paired.
        """
        a_vals, b_vals = [], []
        for ex in self.examples:
            if model_a in ex.scores and model_b in ex.scores:
                a_vals.append(ex.scores[model_a])
                b_vals.append(ex.scores[model_b])
        return np.array(a_vals, dtype=float), np.array(b_vals, dtype=float)

    def differences(self, model_a: str, model_b: str) -> np.ndarray:
        """Paired differences ``score_B - score_A`` over shared examples."""
        a, b = self.paired_scores(model_a, model_b)
        return b - a


@dataclass(frozen=True)
class Finding:
    """One audit result, structured around EvalTrust's Golden Rule.

    Every finding must answer: why is this a problem, how did we detect it, and
    how do I fix it.
    """

    pillar: str
    title: str
    status: Status
    why: str
    how_detected: str
    how_to_fix: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """A JSON-serializable representation for the API and ``--json`` output."""
        return {
            "pillar": self.pillar,
            "title": self.title,
            "status": self.status.name,
            "why": self.why,
            "how_detected": self.how_detected,
            "how_to_fix": self.how_to_fix,
            "details": _jsonable(self.details),
        }


def _jsonable(value):
    """Best-effort conversion of numpy/other scalars into plain JSON types.

    Non-finite floats (``inf``/``-inf``/``nan``) — which a zero-variance Cohen's d
    or an infinite signal-to-noise ratio can produce — become ``null``. Emitting
    them verbatim would make ``json.dumps`` write the non-standard ``Infinity`` /
    ``NaN`` tokens that strict JSON parsers reject.
    """
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)  # numpy scalar -> python scalar
    if callable(item):
        try:
            scalar = value.item()
        except (ValueError, TypeError):
            pass
        else:
            return _jsonable(scalar)  # re-check finiteness for numpy floats
    return str(value)
