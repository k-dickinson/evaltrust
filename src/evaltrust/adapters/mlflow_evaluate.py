"""MLflow evaluate adapter.

Reads the per-row table ``mlflow.evaluate()`` logs as ``eval_results_table.json``:
one row per example, with input/target/output columns plus one column per metric.
MLflow's default evaluator names a metric's per-row score column ``<metric>/score``
(or ``<metric>/<version>/score`` for a versioned built-in like ``toxicity/v1``),
except ``token_count`` and ``latency``, which keep their bare name. The table
itself is a pandas DataFrame dumped with ``to_json(orient="split", index=False)``,
i.e. a ``{"columns": [...], "data": [[...]]}`` object -- also accepted wrapped as
``{"eval_results_table": ...}``, matching ``result.tables["eval_results_table"]``.
One model per ``mlflow.evaluate()`` run; multiple metric columns fan out into a
metric suite.
"""

from __future__ import annotations

import re
from collections import OrderedDict

from ..core.schema import EvalData
from .common import Record, coerce_score, records_to_suite

_SCORE_COL = re.compile(r"^(?P<metric>.+)/score$")
_BARE_METRIC_COLS = {"token_count", "latency"}


def _rows_from_split(table: dict) -> list[dict] | None:
    columns = table.get("columns")
    data = table.get("data")
    if not (isinstance(columns, list) and columns and isinstance(data, list) and data):
        return None
    if not all(isinstance(c, str) for c in columns):
        return None
    rows = []
    for row in data:
        if not isinstance(row, list) or len(row) != len(columns):
            return None
        rows.append(dict(zip(columns, row)))
    return rows


def _table(raw) -> list[dict] | None:
    if not isinstance(raw, dict):
        return None
    if "eval_results_table" in raw:
        inner = raw["eval_results_table"]
        if isinstance(inner, dict):
            return _rows_from_split(inner)
        if isinstance(inner, list) and inner and all(isinstance(r, dict) for r in inner):
            return inner
        return None
    return _rows_from_split(raw)


def _score_columns(rows: list[dict]) -> "OrderedDict[str, str]":
    """Metric name -> column key, for every MLflow-style score column present."""
    found: "OrderedDict[str, str]" = OrderedDict()
    for key in rows[0]:
        m = _SCORE_COL.match(key)
        if m:
            found[m.group("metric")] = key
        elif key in _BARE_METRIC_COLS:
            found[key] = key
    return found


class MlflowEvaluateAdapter:
    source_format = "mlflow_evaluate"

    def detect(self, raw) -> bool:
        rows = _table(raw)
        return bool(rows) and bool(_score_columns(rows))

    def _to_suite(self, raw) -> "OrderedDict[str, EvalData]":
        rows = _table(raw)
        if not rows:
            raise ValueError("No MLflow eval_results_table rows found")
        metric_cols = _score_columns(rows)
        if not metric_cols:
            raise ValueError("No MLflow-style metric score columns found")

        model = "model"
        records: list[Record] = []
        skipped = 0
        for idx, row in enumerate(rows):
            for metric, col in metric_cols.items():
                if col not in row:
                    skipped += 1
                    continue
                try:
                    score = coerce_score(row[col])
                except ValueError:
                    skipped += 1
                    continue
                records.append(Record(str(idx), model, score, metric=metric))

        if not records:
            raise ValueError("No usable MLflow evaluate metric scores found")
        return records_to_suite(records, self.source_format, {"skipped_rows": skipped})

    def parse(self, raw) -> EvalData:
        # Single-audit path: first metric column (by column order) is audited.
        return next(iter(self._to_suite(raw).values()))

    def parse_suite(self, raw) -> "OrderedDict[str, EvalData]":
        # Suite path: every metric column fans out into its own metric.
        return self._to_suite(raw)
