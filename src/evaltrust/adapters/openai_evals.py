"""Line adapter for OpenAI Evals (openai/evals) JSONL log output.

A run writes a line-delimited log: a leading ``spec`` object carrying the run
metadata (the model under ``completion_fns``), a stream of per-sample events
(``sampling``, ``match``, ...), and a trailing ``final_report``. Basic evals
(Match/Includes/FuzzyMatch/MultipleChoice) grade each sample with a ``match``
event whose ``data.correct`` bool is the score. One model per run; compare two
runs with ``evaltrust audit runA.jsonl runB.jsonl``.

Field names come from openai/evals ``evals/record.py`` (event and spec shape)
and ``evals/base.py`` ``RunSpec.completion_fns``. Model-graded evals record a
``choice`` and a config-mapped ``score`` instead of a ``correct`` bool; those are
skipped-and-counted here and left for a follow-up.
"""

from __future__ import annotations

from pathlib import Path

from .common import Record, coerce_score


def _model_from_spec(rows: list[dict]) -> str:
    """The evaluated model, from the spec line's ``completion_fns`` (first entry)."""
    for row in rows:
        spec = row.get("spec") if isinstance(row, dict) else None
        if isinstance(spec, dict):
            fns = spec.get("completion_fns")
            if isinstance(fns, list) and fns:
                return str(fns[0])
    return "model"


class OpenAIEvalsAdapter:
    source_format = "openai-evals"

    def detect_lines(self, rows: list[dict]) -> bool:
        if not rows:
            return False
        for row in rows:
            if not isinstance(row, dict):
                continue
            spec = row.get("spec")
            if isinstance(spec, dict) and (
                "completion_fns" in spec or "eval_name" in spec
            ):
                return True
            # A per-sample event: unmistakably OpenAI Evals, and distinct from the
            # generic/lm-eval row shapes.
            if {"event_id", "sample_id", "type", "data"} <= row.keys():
                return True
        return False

    def parse_lines(
        self, rows: list[dict], *, path: Path | None = None
    ) -> tuple[list[Record], dict]:
        if not self.detect_lines(rows):
            raise ValueError("Not an OpenAI Evals log")

        model = _model_from_spec(rows)
        records: list[Record] = []
        skipped = 0
        for row in rows:
            # Only ``match`` events carry a per-sample score; other lines
            # (sampling, spec, final_report) are not score rows, so not counted.
            if not isinstance(row, dict) or row.get("type") != "match":
                continue
            sample_id = row.get("sample_id")
            data = row.get("data")
            if sample_id is None or not isinstance(data, dict) or "correct" not in data:
                skipped += 1             # a grade row we couldn't read
                continue
            try:
                score = coerce_score(data["correct"])
            except (ValueError, TypeError):
                skipped += 1
                continue
            records.append(Record(str(sample_id), model, score, metric="accuracy"))

        if not records:
            raise ValueError("No match events found in the OpenAI Evals log")
        return records, {"skipped_rows": skipped}
