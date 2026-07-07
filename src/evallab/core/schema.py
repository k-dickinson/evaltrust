"""The canonical data model.

Every eval platform's output is mapped into these types by an adapter, and every
audit check reads only these types. This one-format-in-the-middle design is what
lets the statistics be written once and work for DeepEval, Promptfoo, LangSmith,
OpenEvals, or a plain CSV alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Status(Enum):
    """Outcome of a single audit check."""

    PASS = "pass"    # evidence supports trusting the conclusion
    WARN = "warn"    # a caveat that weakens confidence
    FAIL = "fail"    # the conclusion is not supported by the evidence
    SKIP = "skip"    # the data needed for this check is absent


@dataclass(frozen=True)
class Example:
    """One evaluated item and its score(s).

    ``scores`` maps model -> final score. The optional fields carry the extra
    evidence that unlocks more checks:
      - ``runs``:   model -> list of scores from repeated evaluations.
      - ``judges``: judge -> {model -> score} when several judges scored the item.
    """

    id: str
    scores: dict[str, float]
    runs: dict[str, list[float]] | None = None
    judges: dict[str, dict[str, float]] | None = None


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
    """One audit result, structured around EvalLab's Golden Rule.

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
