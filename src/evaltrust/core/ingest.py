"""Load an evaluation file from disk and normalise it to canonical EvalData.

Reads the file, routes JSON through structural auto-detection and CSV through the
shared record extractor, and returns EvalData. The user never thinks about
formats.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from collections import OrderedDict

from .pairing import merge_two, primary_model
from .schema import EvalData
from ..adapters.common import (
    DEFAULT_METRIC,
    records_to_evaldata,
    records_to_suite,
)
from ..adapters.generic import _find_record_list, dicts_to_records
from ..adapters.registry import UnknownFormatError, detect_adapter


def _load_csv(text: str) -> EvalData:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise UnknownFormatError("The CSV file has no data rows.")
    skipped: list = []
    records = dicts_to_records(rows, skipped)
    return records_to_evaldata(records, "csv", {"skipped_rows": len(skipped)})


def _load_json(text: str) -> EvalData:
    raw = json.loads(text)
    return detect_adapter(raw).parse(raw)


def _is_json_array_document(text: str) -> bool:
    """True when the file is a single JSON array rather than line-delimited rows.

    A ``.jsonl`` record starts with ``{``. A file starting with ``[`` is a JSON
    array mis-named ``.jsonl``, so we route it back through JSON detection.
    """
    return text.lstrip().startswith("[")


def _parse_jsonl_dicts(text: str, name: str) -> list[dict]:
    """Parse line-delimited JSON into a list of record dicts.

    Splits only on ``\\r``/``\\n`` (not ``str.splitlines()``) so a Unicode line
    separator inside a value can't tear a record. A non-object line raises
    ``ValueError`` naming the 1-based line number.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    rows: list[dict] = []
    for i, line in enumerate(normalised.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Could not parse line {i} of '{name}' as JSON (column {e.colno}): "
                f"{e.msg}. Each line of a .jsonl file must be one JSON record."
            ) from e
        if not isinstance(obj, dict):
            raise ValueError(
                f"Could not read line {i} of '{name}': expected one JSON object per "
                f"line, got a JSON {type(obj).__name__}. A JSON array belongs in a "
                f".json file."
            )
        rows.append(obj)
    if not rows:
        raise UnknownFormatError("The JSONL file has no data rows.")
    return rows


def _load_jsonl(text: str, name: str) -> EvalData:
    if _is_json_array_document(text):
        return _load_json(text)
    skipped: list = []
    records = dicts_to_records(_parse_jsonl_dicts(text, name), skipped)
    return records_to_evaldata(records, "jsonl", {"skipped_rows": len(skipped)})


def load(path: str) -> EvalData:
    """Read ``path`` and return canonical EvalData.

    Routes by extension (``.json`` / ``.jsonl`` / ``.csv``); anything else is
    tried as JSON, then JSONL, then CSV.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such evaluation file: {path}")

    text = p.read_text()
    suffix = p.suffix.lower()

    if suffix == ".csv":
        return _load_csv(text)
    if suffix == ".json":
        try:
            return _load_json(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Could not parse '{p.name}' as JSON (line {e.lineno}, "
                f"column {e.colno}). Check that the file is valid JSON."
            ) from e
    if suffix == ".jsonl":
        return _load_jsonl(text, p.name)

    try:
        return _load_json(text)
    except (json.JSONDecodeError, UnknownFormatError):
        pass
    try:
        return _load_jsonl(text, p.name)
    except (ValueError, UnknownFormatError):
        return _load_csv(text)


def load_suite(path: str) -> "OrderedDict[str, EvalData]":
    """Load a file as a metric -> dataset map.

    A file with a ``metric`` column becomes a multi-entry suite; everything else
    becomes a single entry keyed ``"score"``. Routing follows ``load()``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such evaluation file: {path}")
    text = p.read_text()
    suffix = p.suffix.lower()

    def _suite_from_rows(rows, fmt) -> "OrderedDict[str, EvalData]":
        skipped: list = []
        records = dicts_to_records(rows, skipped)
        return records_to_suite(records, fmt, {"skipped_rows": len(skipped)})

    def _suite_from_json() -> "OrderedDict[str, EvalData]":
        # Only the generic record list can carry a metric column.
        raw = json.loads(text)
        adapter = detect_adapter(raw)
        if adapter.source_format == "generic":
            return _suite_from_rows(_find_record_list(raw), "generic")
        return OrderedDict([(DEFAULT_METRIC, adapter.parse(raw))])

    if suffix == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows:
            raise UnknownFormatError("The CSV file has no data rows.")
        return _suite_from_rows(rows, "csv")

    if suffix == ".jsonl":
        # A JSON array mis-named .jsonl is really one JSON document; route it
        # through detection so a metric column still fans out into a suite.
        if _is_json_array_document(text):
            return _suite_from_json()
        return _suite_from_rows(_parse_jsonl_dicts(text, p.name), "jsonl")

    # JSON (or fallback).
    try:
        return _suite_from_json()
    except (json.JSONDecodeError, UnknownFormatError):
        if suffix == ".json":
            raise
        try:
            return _suite_from_rows(_parse_jsonl_dicts(text, p.name), "jsonl")
        except (ValueError, UnknownFormatError):
            return _suite_from_rows(list(csv.DictReader(io.StringIO(text))), "csv")


def load_comparison(
    paths: list[str],
    label_a: str | None = None,
    label_b: str | None = None,
) -> EvalData:
    """Load one multi-model file, or pair two single-model files into a comparison.

    With two files, each must contain exactly one model. Labels default to the
    models' own names, falling back to the file stems if those names collide, and
    are overridden by ``label_a`` / ``label_b`` when given.
    """
    if len(paths) == 1:
        return load(paths[0])
    if len(paths) != 2:
        raise ValueError("Provide one results file, or two to compare.")

    data_a, data_b = load(paths[0]), load(paths[1])
    model_a, model_b = primary_model(data_a), primary_model(data_b)

    if model_a != model_b:
        la, lb = model_a, model_b
    else:
        la, lb = Path(paths[0]).stem, Path(paths[1]).stem
    la, lb = label_a or la, label_b or lb
    if la == lb:
        lb = f"{lb}_2"

    return merge_two(data_a, data_b, la, lb)
