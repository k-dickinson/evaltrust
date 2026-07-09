"""DeepEval adapter.

DeepEval evaluates one system per run, so a single export contains one model;
you compare two runs with ``evaltrust audit runA.json runB.json``. This adapter
reads DeepEval's evaluation-result export (from ``evaluate(...)`` /
``deepeval test run``), tolerating both the snake_case (``test_results`` /
``metrics_data``) and camelCase (``testCases`` / ``metricsData``) shapes.

Per test case the score is the case's ``success`` (pass/fail), falling back to
the mean of its metric scores when ``success`` is absent. If the export records a
model under ``hyperparameters``, that name is used; otherwise the model is left
generic and the file name supplies the label when pairing two runs.
"""

from __future__ import annotations

import numpy as np

from ..core.schema import EvalData
from .common import Record, coerce_score, records_to_evaldata


def _cases(raw):
    if not isinstance(raw, dict):
        return None
    for key in ("test_results", "testCases", "test_cases"):
        if isinstance(raw.get(key), list):
            return raw[key]
    return None


def _looks_like_deepeval(rows) -> bool:
    if not rows or not isinstance(rows[0], dict):
        return False
    first = rows[0]
    return any(k in first for k in ("metrics_data", "metricsData")) or (
        "success" in first and ("name" in first or "input" in first))


class DeepEvalAdapter:
    source_format = "deepeval"

    def detect(self, raw) -> bool:
        rows = _cases(raw)
        return rows is not None and _looks_like_deepeval(rows)

    def parse(self, raw) -> EvalData:
        rows = _cases(raw)
        if rows is None:
            raise ValueError("No DeepEval test-results array found")

        model = str(raw.get("hyperparameters", {}).get("model") or "model")

        records: list[Record] = []
        for idx, row in enumerate(rows):
            ex_id = str(row.get("name", idx))
            records.append(Record(ex_id, model, _case_score(row)))
        return records_to_evaldata(records, self.source_format)


def _case_score(row: dict) -> float:
    if "success" in row:
        return coerce_score(row["success"])
    metrics = row.get("metrics_data") or row.get("metricsData") or []
    scores = [coerce_score(m["score"]) for m in metrics if m.get("score") is not None]
    if scores:
        return float(np.mean(scores))
    raise ValueError(f"DeepEval test case {row.get('name', '?')} has no score")
