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

from .common import Record


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
        # Require the run-level spec signature (``completion_fns``), the field
        # unique to an OpenAI Evals log. Generic event keys alone are not enough:
        # this adapter runs before generic JSONL extraction, so an unrelated
        # event stream must fall through rather than be claimed and then fail.
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            spec = row.get("spec")
            if isinstance(spec, dict) and isinstance(spec.get("completion_fns"), list):
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
            # The format contract is ``data.correct: bool``. A missing sample id,
            # non-dict data, or a non-bool ``correct`` is a grade row we can't
            # trust, so skip and count it rather than coerce a stray number.
            if (
                sample_id is None
                or not isinstance(data, dict)
                or not isinstance(data.get("correct"), bool)
            ):
                skipped += 1
                continue
            score = 1.0 if data["correct"] else 0.0
            records.append(Record(str(sample_id), model, score, metric="accuracy"))

        if not records:
            raise ValueError("No match events found in the OpenAI Evals log")
        return records, {"skipped_rows": skipped}
