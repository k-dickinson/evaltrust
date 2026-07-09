"""Inspect (UK AISI) ``.json`` eval-log adapter.

Inspect evaluates one model per run and writes an ``EvalLog``: a single JSON
document with the model under ``eval.model`` and a list of ``samples``, each
carrying per-scorer results under ``sample.scores`` (``{scorer: {"value": ...}}``).
A single log therefore contains one model — you compare two runs with
``evaltrust audit runA.json runB.json``, like the DeepEval and OpenEvals adapters.

Detection is structural: a top-level ``eval`` object with a ``model``, plus a
``samples`` list in which some entry carries a ``scores`` map (``scorer -> Score``).
Keying on that ``scores`` map is what separates a real log from a plain record list
nested under ``samples`` (which has flat ``score`` fields), so this adapter must sit
before the generic fallback in the registry.

A scorer's ``value`` is turned into a float the way Inspect's own ``value_to_float``
maps the grade constants — ``C``/``I``/``P``/``N`` (CORRECT/INCORRECT/PARTIAL/
NOANSWER) → 1 / 0 / 0.5 / 0 — and everything else goes through the shared
``coerce_score`` (numbers, booleans, and yes/no/true/false, plus the wider
pass/fail vocabulary EvalTrust accepts). A value that can't be read as a score
(``null``, a list, a dict) is skipped and counted, like the CSV/record path, so
one unscored sample never sinks the whole file. Several scorers yield a single-
metric ``EvalData`` on the first scorer, as the OpenEvals adapter does; repeated
``epoch``s of one sample become that example's repeated runs.

Grounding: the fixture and mapping are derived from the upstream ``inspect_ai``
package (UK AISI) — see the PR description for the exact commit and file paths —
not from a live export.
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
        # The fingerprint: an Inspect sample carries a `scores` map of
        # scorer -> Score, and a Score is an object with a `value`. Requiring an
        # actual Score-shaped entry (not merely any dict) keeps a plain/native
        # record that happens to sit under "samples" (a flat `score`, or `scores`
        # keyed model -> number) from being claimed here. This recognises the
        # format structurally; a recognised log whose scores are all unusable
        # (e.g. every value null) is still claimed, and parse() then fails with a
        # clear, Inspect-specific error rather than a generic "unknown format".
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
                continue            # an unscored sample contributes no rows to drop
            for scorer, score in scores.items():
                # Count every present-but-unusable entry, like the CSV path counts
                # unreadable cells, so the Data Quality finding isn't understated.
                if not isinstance(score, dict) or "value" not in score:
                    skipped += 1     # malformed Score entry (not a {"value": ...})
                    continue
                try:
                    value = _score_to_float(score["value"])
                except (ValueError, TypeError):
                    skipped += 1     # value can't be read as a score (null/list/dict)
                    continue
                records.append(Record(ex_id, model, value, metric=str(scorer)))

        if not records:
            raise ValueError("No scored samples found in the Inspect eval log")

        # A dedicated adapter yields one EvalData; the metric fan-out is reserved
        # for the generic record path, so the first scorer is the audited metric.
        suite = records_to_suite(records, self.source_format, {"skipped_rows": skipped})
        return next(iter(suite.values()))
