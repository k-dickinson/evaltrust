"""Generic JSON adapters: native nested output and flat record lists.

These cover the long tail: anything that isn't a recognised tool but is still,
structurally, "examples with per-model scores." Between them and the CSV adapter,
scores in almost any shape can be audited without reformatting.
"""

from __future__ import annotations

from ..core.schema import EvalData, Example
from .common import (
    DEFAULT_METRIC,
    ID_KEYS,
    JUDGE_KEYS,
    METRIC_KEYS,
    MODEL_KEYS,
    Record,
    SCORE_KEYS,
    coerce_score,
    records_to_evaldata,
)

# Keys under which tools commonly nest a list of result rows.
_LIST_WRAPPERS = ("results", "data", "rows", "outputs", "samples", "records",
                  "predictions", "evaluations")


def _first_alias(keys, aliases) -> str | None:
    """Return the actual key whose lower-cased name is in ``aliases``."""
    lowered = {str(k).lower(): k for k in keys}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _find_record_list(raw) -> list | None:
    """Locate a list of dict rows, either raw itself or nested under a wrapper."""
    if isinstance(raw, list) and raw and all(isinstance(r, dict) for r in raw):
        return raw
    if isinstance(raw, dict):
        for key in _LIST_WRAPPERS:
            got = _first_alias(raw.keys(), (key,))
            if got and isinstance(raw[got], list) and raw[got] \
                    and all(isinstance(r, dict) for r in raw[got]):
                return raw[got]
    return None


def dicts_to_records(rows: list[dict], skipped: list | None = None) -> list[Record]:
    """Extract (example, model, score) records from dict rows.

    Handles long format (model + score columns) and wide format (one score column
    per model, non-numeric columns ignored). Rows with an unreadable score are
    skipped, with a reason appended to ``skipped`` if given.
    """
    keys = rows[0].keys()
    id_key = _first_alias(keys, ID_KEYS)
    model_key = _first_alias(keys, MODEL_KEYS)
    judge_key = _first_alias(keys, JUDGE_KEYS)
    score_key = _first_alias(keys, SCORE_KEYS)
    metric_key = _first_alias(keys, METRIC_KEYS)

    records: list[Record] = []
    for idx, row in enumerate(rows):
        ex_id = str(row[id_key]) if id_key and row.get(id_key) is not None else str(idx)
        judge = str(row[judge_key]) if judge_key and row.get(judge_key) is not None else None
        metric = (str(row[metric_key]) if metric_key and row.get(metric_key) is not None
                  else DEFAULT_METRIC)

        if model_key and score_key:
            try:
                score = coerce_score(row[score_key])
            except ValueError:
                if skipped is not None:
                    skipped.append(f"row {idx}: unreadable score {row.get(score_key)!r}")
                continue
            records.append(Record(ex_id, str(row[model_key]), score, judge, metric))
        else:
            reserved = {k for k in (id_key, judge_key, metric_key) if k}
            for col, val in row.items():
                if col in reserved:
                    continue
                try:
                    score = coerce_score(val)
                except ValueError:
                    continue
                records.append(Record(ex_id, str(col), score, judge, metric))

    if not records:
        raise ValueError("No (example, model, score) records could be extracted")
    return records


class GenericRecordsAdapter:
    source_format = "generic"

    def detect(self, raw) -> bool:
        return _find_record_list(raw) is not None

    def parse(self, raw) -> EvalData:
        rows = _find_record_list(raw)
        if rows is None:
            raise ValueError("No record list found for the generic adapter")
        skipped: list = []
        records = dicts_to_records(rows, skipped)
        return records_to_evaldata(records, self.source_format,
                                   {"skipped_rows": len(skipped)})


class NativeNestedAdapter:
    """Structured JSON: {"examples": [{"id", "scores": {model: score}, ...}]}."""

    source_format = "native"

    def detect(self, raw) -> bool:
        return (
            isinstance(raw, dict)
            and isinstance(raw.get("examples"), list)
            and bool(raw["examples"])
            and isinstance(raw["examples"][0], dict)
            and "scores" in raw["examples"][0]
        )

    def parse(self, raw) -> EvalData:
        examples = []
        models: list[str] = []
        for i, ex in enumerate(raw["examples"]):
            scores = {str(m): coerce_score(s) for m, s in ex["scores"].items()}
            for m in scores:
                if m not in models:
                    models.append(m)
            runs = ex.get("runs")
            runs = ({str(m): [coerce_score(v) for v in vs] for m, vs in runs.items()}
                    if runs else None)
            judges = ex.get("judges")
            judges = ({str(j): {str(m): coerce_score(v) for m, v in mv.items()}
                       for j, mv in judges.items()} if judges else None)
            examples.append(Example(id=str(ex.get("id", i)), scores=scores,
                                    runs=runs, judges=judges))

        if raw.get("models"):
            models = [str(m) for m in raw["models"]]
        return EvalData(models=models, examples=examples,
                        source_format=self.source_format,
                        metadata=raw.get("metadata", {}))
