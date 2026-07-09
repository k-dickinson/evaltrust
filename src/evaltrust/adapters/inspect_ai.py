"""Inspect (UK AISI) ``.json`` eval-log adapter.

An Inspect ``EvalLog`` has the model under ``eval.model`` and a list of
``samples``, each with per-scorer results under ``sample.scores``
(``{scorer: {"value": ...}}``). A log holds one model, so compare two runs with
``evaltrust audit runA.json runB.json``. The first scorer is the audited metric.
"""

from __future__ import annotations

from ..core.schema import EvalData
from .common import Record, coerce_score, records_to_suite

# inspect_ai/scorer/_metric.py: CORRECT="C", INCORRECT="I", PARTIAL="P", NOANSWER="N",
# mapped to 1 / 0 / 0.5 / 0 by value_to_float.
_GRADES = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}


def _score_to_float(value) -> float:
    """Map an Inspect score value to a float; raise if it isn't a scalar score."""
    if isinstance(value, str) and value in _GRADES:
        return _GRADES[value]
    return coerce_score(value)   # numbers, booleans, and word/number strings


class InspectAdapter:
    source_format = "inspect"

    def detect(self, raw) -> bool:
        if not isinstance(raw, dict):
            return False
        ev = raw.get("eval")
        samples = raw.get("samples")
        if not (isinstance(ev, dict) and "model" in ev
                and isinstance(samples, list) and samples):
            return False
        # Fingerprint: a sample carries a `scores` map of scorer -> Score (an
        # object with a `value`), which a plain record under "samples" lacks.
        return any(
            isinstance(s, dict) and isinstance(s.get("scores"), dict)
            and any(isinstance(v, dict) and "value" in v
                    for v in s["scores"].values())
            for s in samples)

    def parse(self, raw) -> EvalData:
        if not self.detect(raw):
            raise ValueError("Not an Inspect eval log")
        raw_model = raw["eval"].get("model")
        model = raw_model if isinstance(raw_model, str) and raw_model else "model"

        records: list[Record] = []
        skipped = 0
        for idx, sample in enumerate(raw["samples"]):
            if not isinstance(sample, dict):
                continue
            sid = sample.get("id")
            ex_id = str(sid) if sid is not None else str(idx)
            scores = sample.get("scores")
            if not isinstance(scores, dict):
                continue
            for scorer, score in scores.items():
                # Count present-but-unusable entries so Data Quality isn't understated.
                if not isinstance(score, dict) or "value" not in score:
                    skipped += 1
                    continue
                try:
                    value = _score_to_float(score["value"])
                except (ValueError, TypeError):
                    skipped += 1
                    continue
                records.append(Record(ex_id, model, value, metric=str(scorer)))

        if not records:
            raise ValueError("No scored samples found in the Inspect eval log")

        # First scorer is the audited metric (fan-out is the generic path's job).
        suite = records_to_suite(records, self.source_format, {"skipped_rows": skipped})
        return next(iter(suite.values()))
