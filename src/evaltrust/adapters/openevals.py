"""OpenEvals adapter (langchain-ai/openevals).

OpenEvals evaluates one model per run and returns a list of
``EvaluatorResult`` dicts — each with a ``key`` (metric name) and a
``score``.  A single export therefore contains one model; users compare
two runs with ``evaltrust audit runA.json runB.json``.

Detection is structural: a non-empty list whose first element contains
both ``key`` and ``score`` fields.  The ``comment`` field is optional
but common; other keys (``input``, ``output``, ``metadata``, ...) are
tolerated and ignored.
"""

from __future__ import annotations

from ..core.schema import EvalData
from .common import Record, coerce_score, records_to_suite


def _looks_like_openevals(raw) -> bool:
    if not isinstance(raw, list) or not raw:
        return False
    first = raw[0]
    return isinstance(first, dict) and "key" in first and "score" in first


class OpenEvalsAdapter:
    source_format = "openevals"

    def detect(self, raw) -> bool:
        return _looks_like_openevals(raw)

    def parse(self, raw) -> EvalData:
        if not isinstance(raw, list) or not raw:
            raise ValueError("No OpenEvals results list found")

        records: list[Record] = []
        for idx, row in enumerate(raw):
            if not isinstance(row, dict):
                continue
            metric = str(row.get("key") or "score")
            raw_score = row.get("score")
            if raw_score is None:
                continue
            ex_id = str(row.get("input", idx))
            records.append(Record(ex_id, "model", coerce_score(raw_score), metric=metric))

        suite = records_to_suite(records, self.source_format)
        # Return the first (or only) metric dataset as the primary EvalData
        return next(iter(suite.values()))
