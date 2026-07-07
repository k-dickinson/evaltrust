"""Load an evaluation file from disk and normalise it to canonical EvalData.

The user runs ``evaltrust audit results.json`` (or ``.csv``) and never thinks about
formats. This module reads the file, routes JSON through structural auto-detection
and CSV through the shared record extractor, and returns EvalData.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .pairing import merge_two, primary_model
from .schema import EvalData
from ..adapters.common import records_to_evaldata
from ..adapters.generic import dicts_to_records
from ..adapters.registry import UnknownFormatError, detect_adapter


def _load_csv(text: str) -> EvalData:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise UnknownFormatError("The CSV file has no data rows.")
    return records_to_evaldata(dicts_to_records(rows), "csv")


def _load_json(text: str) -> EvalData:
    raw = json.loads(text)
    return detect_adapter(raw).parse(raw)


def load(path: str) -> EvalData:
    """Read ``path`` and return canonical EvalData.

    Routing is by extension, with a content fallback: a ``.json`` file goes
    through JSON auto-detection, a ``.csv`` file through the CSV reader, and
    anything else is tried as JSON then CSV.
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

    try:
        return _load_json(text)
    except (json.JSONDecodeError, UnknownFormatError):
        return _load_csv(text)


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
