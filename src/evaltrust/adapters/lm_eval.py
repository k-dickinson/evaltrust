"""Line adapter for lm-evaluation-harness sample logs."""

from __future__ import annotations

from pathlib import Path
import re

from .common import Record, coerce_score


# lm_eval/evaluator.py builds these fields before evaluation_tracker.py writes
# each sample. Its ``metrics`` list names the task-defined metric fields.
_RESERVED_FIELDS = {
    "doc_id",
    "doc",
    "target",
    "arguments",
    "resps",
    "filtered_resps",
    "filter",
    "metrics",
    "doc_hash",
    "prompt_hash",
    "target_hash",
}
_TIMESTAMP_SUFFIX = re.compile(
    r"^(?P<task>.+)_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?$"
)


def _model_from_path(path: Path | None) -> str:
    stem = path.stem if path is not None else "model"
    if stem.startswith("samples_"):
        stem = stem[len("samples_"):]
    match = _TIMESTAMP_SUFFIX.fullmatch(stem)
    return match.group("task") if match else stem


class LMEvalAdapter:
    source_format = "lm-eval"

    def detect_lines(self, rows: list[dict]) -> bool:
        if not rows or not isinstance(rows[0], dict):
            return False
        first = rows[0]
        # lm-eval samples have no model key, so generic long rows cannot match.
        return (
            "doc_id" in first
            and ("resps" in first or "filtered_resps" in first)
            and "model" not in first
        )

    def parse_lines(
        self, rows: list[dict], *, path: Path | None = None
    ) -> tuple[list[Record], dict]:
        if not self.detect_lines(rows):
            raise ValueError("Not an lm-eval sample log")

        model = _model_from_path(path)
        records: list[Record] = []
        skipped = 0
        for row in rows:
            doc_id = row.get("doc_id")
            if doc_id is None:
                skipped += 1
                continue
            declared_metrics = row.get("metrics")
            metric_fields = (
                declared_metrics
                if isinstance(declared_metrics, list)
                else row.keys()
            )
            for metric in metric_fields:
                if metric in _RESERVED_FIELDS:
                    continue
                raw = row.get(metric)
                try:
                    score = coerce_score(raw)
                except ValueError:
                    skipped += 1
                    continue
                records.append(Record(str(doc_id), model, score, metric=str(metric)))

        if not records:
            raise ValueError("No scored samples found in the lm-eval sample log")
        return records, {"skipped_rows": skipped, "model_name_inferred": True}
