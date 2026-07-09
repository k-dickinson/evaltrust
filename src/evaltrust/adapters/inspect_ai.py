"""Inspect (UK AISI) ``.json`` eval-log adapter.

Inspect evaluates one model per run and writes an ``EvalLog``: a JSON document
with the model under ``eval.model`` and a list of ``samples``, each carrying
per-scorer results under ``sample.scores`` (``{scorer: {"value": ...}}``). A log
contains one model, so you compare two runs with
``evaltrust audit runA.json runB.json``.

Detection is structural: a top-level ``eval`` with a ``model``, plus a ``samples``
list where some entry carries a ``scores`` map (``scorer -> Score``). Keying on
that map separates a real log from a plain record list nested under ``samples``,
so this adapter must sit before the generic fallback in the registry.

A scorer's ``value`` becomes a float the way Inspect's ``value_to_float`` maps the
grade constants (``C``/``I``/``P``/``N`` -> 1 / 0 / 0.5 / 0); everything else goes
through ``coerce_score``. A value that can't be read as a score is skipped and
counted, so one unscored sample never sinks the file. Multiple scorers yield a
single-metric ``EvalData`` on the first scorer; repeated ``epoch``s of one sample
become that example's repeated runs.
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
        # Fingerprint: a sample carries a `scores` map of scorer -> Score, where a
        # Score is an object with a `value`. Requiring a Score-shaped entry (not
        # any dict) keeps a plain record nested under "samples" from being claimed
        # here. A recognised log whose scores are all unusable is still claimed,
        # and parse() then fails with a clear Inspect-specific error.
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
