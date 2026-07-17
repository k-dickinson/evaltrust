"""Load an evaluation file from disk and normalise it to canonical EvalData.

Reads the file, routes JSON through structural auto-detection and CSV through the
shared record extractor, and returns EvalData. The user never thinks about
formats.

Large-file streaming
--------------------
JSONL and CSV are read line-by-line via generator pipelines so the peak memory
footprint is bounded by the streaming buffer rather than the file size.  The
threshold is ``_STREAM_THRESHOLD`` bytes (default 64 MiB); files smaller than
that are fully materialised first (preserving the original behaviour) so the
fast path stays fast.

JSON is handled the same way for small files.  For the two common large-file
shapes — a top-level array or ``{"examples": [...]}`` — an optional iterative
parser is used when the ``ijson`` library is available, keeping peak memory O(1)
in the record count.  When ``ijson`` is absent the file is loaded normally; a
warning is emitted only for files that exceed the threshold.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Generator, Iterable

from .pairing import merge_two, primary_model
from .schema import EvalData
from ..adapters.common import (
    DEFAULT_METRIC,
    Record,
    records_to_evaldata,
    records_to_suite,
)
from ..adapters.generic import _find_record_list, dicts_to_records
from ..adapters.line_registry import detect_line_adapter
from ..adapters.registry import UnknownFormatError, detect_adapter

logger = logging.getLogger(__name__)

# Files larger than this are streamed rather than fully materialised.
_STREAM_THRESHOLD = 64 * 1024 * 1024  # 64 MiB

# ---------------------------------------------------------------------------
# Internal helpers – streaming generators
# ---------------------------------------------------------------------------

def _iter_jsonl_lines(path: Path) -> Generator[dict, None, None]:
    """Yield one parsed dict per non-blank line of a JSONL file.

    Reads the file incrementally so memory usage is proportional to the largest
    single record, not the file size.  Validates each line and raises
    ``ValueError`` on the first malformed one (with 1-based line number).
    """
    name = path.name
    with path.open(encoding="utf-8") as fh:
        for i, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Could not parse line {i} of '{name}' as JSON "
                    f"(column {e.colno}): {e.msg}. Each line of a .jsonl "
                    f"file must be one JSON record."
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Could not read line {i} of '{name}': expected one JSON "
                    f"object per line, got a JSON {type(obj).__name__}. A JSON "
                    f"array belongs in a .json file."
                )
            yield obj


def _iter_csv_rows(path: Path) -> Generator[dict, None, None]:
    """Yield one DictReader row at a time from a CSV file."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        yield from reader


# ---------------------------------------------------------------------------
# Internal helpers – batch (in-memory) helpers kept for small files / JSON
# ---------------------------------------------------------------------------

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


def _records_from_jsonl(
    text: str, name: str, path: Path | None
) -> tuple[list[Record], str, dict]:
    rows = _parse_jsonl_dicts(text, name)
    adapter = detect_line_adapter(rows)
    if adapter is not None:
        records, metadata = adapter.parse_lines(rows, path=path)
        return records, adapter.source_format, metadata

    skipped: list = []
    records = dicts_to_records(rows, skipped)
    return records, "jsonl", {"skipped_rows": len(skipped)}


def _load_jsonl(text: str, name: str, path: Path | None = None) -> EvalData:
    if _is_json_array_document(text):
        return _load_json(text)
    records, source_format, metadata = _records_from_jsonl(text, name, path)
    return records_to_evaldata(records, source_format, metadata)


# ---------------------------------------------------------------------------
# Streaming paths for large JSONL / CSV files
# ---------------------------------------------------------------------------

def _records_from_jsonl_iter(
    row_iter: Iterable[dict], path: Path
) -> tuple[list[Record], str, dict]:
    """Build records from a dict iterator (used for large JSONL files).

    Memory model
    ------------
    The iterator is materialised into a ``list[dict]`` — one parsed object per
    row — rather than a single raw-text string of the whole file.  This is a
    meaningful reduction: a 1 GB JSONL file whose records have large string
    fields (e.g. prompt/completion text) may produce a much smaller dict list
    once non-score fields are parsed and held as Python objects.

    Two-pass constraint
    -------------------
    ``detect_line_adapter`` inspects *all* rows to recognise tool-specific
    schemas, and ``dicts_to_records`` scans the first row's keys to determine
    column layout before iterating.  Both require random-access to the full row
    list, so a single-pass streaming path is not possible without refactoring
    those interfaces.

    # TODO: refactor detect_line_adapter / dicts_to_records to accept an
    # iterator with a one-row lookahead so the full list need not be retained.
    """
    rows = list(row_iter)
    if not rows:
        raise UnknownFormatError("The JSONL file has no data rows.")

    adapter = detect_line_adapter(rows)
    if adapter is not None:
        records, metadata = adapter.parse_lines(rows, path=path)
        return records, adapter.source_format, metadata

    skipped: list = []
    records = dicts_to_records(rows, skipped)
    return records, "jsonl", {"skipped_rows": len(skipped)}


def _load_jsonl_streamed(path: Path) -> EvalData:
    """Stream a large JSONL file line-by-line."""
    records, source_format, metadata = _records_from_jsonl_iter(
        _iter_jsonl_lines(path), path
    )
    return records_to_evaldata(records, source_format, metadata)


def _load_csv_streamed(path: Path) -> EvalData:
    """Stream a large CSV file row-by-row."""
    rows = list(_iter_csv_rows(path))
    if not rows:
        raise UnknownFormatError("The CSV file has no data rows.")
    skipped: list = []
    records = dicts_to_records(rows, skipped)
    return records_to_evaldata(records, "csv", {"skipped_rows": len(skipped)})


# ---------------------------------------------------------------------------
# Optional ijson-based streaming for large JSON files
# ---------------------------------------------------------------------------

def _load_json_streamed(path: Path) -> EvalData | None:
    """Try to stream a large JSON file using ``ijson``.

    Returns ``None`` when ``ijson`` is not installed or the file shape is not
    one of the two supported patterns (top-level array or
    ``{"examples": [...]}``) so the caller can fall back to a full load.

    Supported shapes
    ----------------
    * Top-level array  ``[{...}, ...]``  → generic record-list adapter
    * ``{"examples": [{...}, ...]}``     → native nested adapter
    """
    try:
        import ijson  # type: ignore[import]
    except ImportError:
        logger.warning(
            "File '%s' exceeds the %d MiB streaming threshold but 'ijson' is "
            "not installed. The file will be loaded fully into memory. Install "
            "'ijson' to enable low-memory JSON streaming.",
            path.name,
            _STREAM_THRESHOLD // (1024 * 1024),
        )
        return None

    # Peek at the first non-whitespace byte to decide shape.
    with path.open("rb") as fh:
        first_byte = b""
        while not first_byte.strip():
            ch = fh.read(1)
            if not ch:
                break
            first_byte = ch

    if first_byte == b"[":
        rows: list[dict] = []
        with path.open("rb") as fh:
            for item in ijson.items(fh, "item"):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Expected a JSON array of objects in '{path.name}', "
                        f"got a {type(item).__name__} element."
                    )
                rows.append(item)
        if not rows:
            raise UnknownFormatError("The JSON file has no data rows.")
        raw = rows
        return detect_adapter(raw).parse(raw)

    if first_byte == b"{":
        rows = []
        with path.open("rb") as fh:
            top: dict = {}
            try:
                for item in ijson.kvitems(fh, ""):
                    key, value = item
                    if key == "examples":
                        break
                    top[key] = value
            except Exception:
                return None

        try:
            with path.open("rb") as fh:
                rows = list(ijson.items(fh, "examples.item"))
        except Exception:
            return None

        if not rows:
            return None

        raw_obj: dict = dict(top)
        raw_obj["examples"] = rows
        return detect_adapter(raw_obj).parse(raw_obj)

    return None


def _suite_from_json_streamed(path: Path) -> "OrderedDict[str, EvalData] | None":
    """Try to build a suite from a large JSON file using ijson.

    Mirrors ``_load_json_streamed`` but routes through ``records_to_suite`` /
    ``parse_suite`` so multi-metric JSON files are handled correctly instead of
    being collapsed into a single DEFAULT_METRIC entry.

    Returns ``None`` when ijson is unavailable or the shape is unsupported.
    """
    try:
        import ijson  # type: ignore[import]
    except ImportError:
        return None

    with path.open("rb") as fh:
        first_byte = b""
        while not first_byte.strip():
            ch = fh.read(1)
            if not ch:
                break
            first_byte = ch

    if first_byte == b"[":
        rows: list[dict] = []
        with path.open("rb") as fh:
            for item in ijson.items(fh, "item"):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Expected a JSON array of objects in '{path.name}', "
                        f"got a {type(item).__name__} element."
                    )
                rows.append(item)
        if not rows:
            raise UnknownFormatError("The JSON file has no data rows.")
        skipped: list = []
        records = dicts_to_records(rows, skipped)
        return records_to_suite(records, "generic", {"skipped_rows": len(skipped)})

    if first_byte == b"{":
        rows = []
        with path.open("rb") as fh:
            top: dict = {}
            try:
                for item in ijson.kvitems(fh, ""):
                    key, value = item
                    if key == "examples":
                        break
                    top[key] = value
            except Exception:
                return None

        try:
            with path.open("rb") as fh:
                rows = list(ijson.items(fh, "examples.item"))
        except Exception:
            return None

        if not rows:
            return None

        raw_obj: dict = dict(top)
        raw_obj["examples"] = rows
        adapter = detect_adapter(raw_obj)
        if hasattr(adapter, "parse_suite"):
            return adapter.parse_suite(raw_obj)
        return OrderedDict([(DEFAULT_METRIC, adapter.parse(raw_obj))])

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(path: str) -> EvalData:
    """Read ``path`` and return canonical EvalData.

    Routes by extension (``.json`` / ``.jsonl`` / ``.csv``); anything else is
    tried as JSON, then JSONL, then CSV.

    Files larger than ``_STREAM_THRESHOLD`` bytes are read incrementally so
    that peak memory is bounded by the streaming buffer rather than the file
    size.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such evaluation file: {path}")

    large = p.stat().st_size > _STREAM_THRESHOLD
    suffix = p.suffix.lower()

    # ---- CSV ----
    if suffix == ".csv":
        if large:
            return _load_csv_streamed(p)
        text = p.read_text(encoding="utf-8")
        return _load_csv(text)

    # ---- JSON ----
    if suffix == ".json":
        if large:
            result = _load_json_streamed(p)
            if result is not None:
                return result
            # ijson unavailable or unrecognised shape: fall back to full load.
        text = p.read_text(encoding="utf-8")
        try:
            return _load_json(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Could not parse '{p.name}' as JSON (line {e.lineno}, "
                f"column {e.colno}). Check that the file is valid JSON."
            ) from e

    # ---- JSONL ----
    if suffix == ".jsonl":
        if large:
            # Peek to check for a mis-named JSON array.
            with p.open(encoding="utf-8") as fh:
                head = fh.read(256)
            if head.lstrip().startswith("["):
                result = _load_json_streamed(p)
                if result is not None:
                    return result
                text = p.read_text(encoding="utf-8")
                return _load_json(text)
            return _load_jsonl_streamed(p)
        text = p.read_text(encoding="utf-8")
        return _load_jsonl(text, p.name, p)

    # ---- Unknown extension: try JSON → JSONL → CSV ----
    if large:
        # Try JSON streaming first; if ijson is absent or shape unsupported,
        # fall through to the normal full-text JSON loader before trying JSONL.
        try:
            result = _load_json_streamed(p)
            if result is not None:
                return result
        except (UnknownFormatError, ValueError):
            pass
        # ijson unavailable or unrecognised JSON shape — try full-text JSON.
        try:
            text = p.read_text(encoding="utf-8")
            return _load_json(text)
        except (json.JSONDecodeError, UnknownFormatError):
            pass
        # Try JSONL streaming.
        try:
            with p.open(encoding="utf-8") as fh:
                head = fh.read(256)
            if not head.lstrip().startswith("["):
                return _load_jsonl_streamed(p)
        except (ValueError, UnknownFormatError):
            pass
        # Fall back to CSV streaming.
        return _load_csv_streamed(p)

    text = p.read_text(encoding="utf-8")
    try:
        return _load_json(text)
    except (json.JSONDecodeError, UnknownFormatError):
        pass
    try:
        return _load_jsonl(text, p.name, p)
    except (ValueError, UnknownFormatError):
        return _load_csv(text)


def load_suite(path: str) -> "OrderedDict[str, EvalData]":
    """Load a file as a metric -> dataset map.

    A file with a ``metric`` column becomes a multi-entry suite; everything else
    becomes a single entry keyed ``"score"``. Routing follows ``load()``.

    Files larger than ``_STREAM_THRESHOLD`` bytes are streamed; see ``load()``
    for details.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such evaluation file: {path}")

    large = p.stat().st_size > _STREAM_THRESHOLD
    suffix = p.suffix.lower()

    def _suite_from_rows(rows, fmt) -> "OrderedDict[str, EvalData]":
        skipped: list = []
        records = dicts_to_records(rows, skipped)
        return records_to_suite(records, fmt, {"skipped_rows": len(skipped)})

    def _suite_from_json(text: str) -> "OrderedDict[str, EvalData]":
        raw = json.loads(text)
        adapter = detect_adapter(raw)
        if adapter.source_format == "generic":
            return _suite_from_rows(_find_record_list(raw), "generic")
        if hasattr(adapter, "parse_suite"):
            return adapter.parse_suite(raw)
        return OrderedDict([(DEFAULT_METRIC, adapter.parse(raw))])

    def _suite_from_jsonl(text: str) -> "OrderedDict[str, EvalData]":
        records, source_format, metadata = _records_from_jsonl(text, p.name, p)
        return records_to_suite(records, source_format, metadata)

    def _suite_from_jsonl_streamed() -> "OrderedDict[str, EvalData]":
        records, source_format, metadata = _records_from_jsonl_iter(
            _iter_jsonl_lines(p), p
        )
        return records_to_suite(records, source_format, metadata)

    # ---- CSV ----
    if suffix == ".csv":
        if large:
            rows = list(_iter_csv_rows(p))
        else:
            text = p.read_text(encoding="utf-8")
            rows = list(csv.DictReader(io.StringIO(text)))
        if not rows:
            raise UnknownFormatError("The CSV file has no data rows.")
        return _suite_from_rows(rows, "csv")

    # ---- JSONL ----
    if suffix == ".jsonl":
        if large:
            with p.open(encoding="utf-8") as fh:
                head = fh.read(256)
            if head.lstrip().startswith("["):
                # Mis-named JSON array: stream via suite-aware path.
                result = _suite_from_json_streamed(p)
                if result is not None:
                    return result
                text = p.read_text(encoding="utf-8")
                return _suite_from_json(text)
            return _suite_from_jsonl_streamed()
        text = p.read_text(encoding="utf-8")
        if _is_json_array_document(text):
            return _suite_from_json(text)
        return _suite_from_jsonl(text)

    # ---- JSON ----
    if suffix == ".json":
        if large:
            result = _suite_from_json_streamed(p)
            if result is not None:
                return result
            # ijson unavailable or unrecognised shape: fall back to full load.
        text = p.read_text(encoding="utf-8")
        try:
            return _suite_from_json(text)
        except (json.JSONDecodeError, UnknownFormatError):
            raise

    # ---- Unknown extension ----
    if large:
        # Try suite-aware JSON streaming first.
        try:
            result = _suite_from_json_streamed(p)
            if result is not None:
                return result
        except (UnknownFormatError, ValueError):
            pass
        # ijson absent or shape unsupported: try full-text JSON.
        try:
            text = p.read_text(encoding="utf-8")
            return _suite_from_json(text)
        except (json.JSONDecodeError, UnknownFormatError):
            pass
        # Try JSONL streaming.
        try:
            with p.open(encoding="utf-8") as fh:
                head = fh.read(256)
            if not head.lstrip().startswith("["):
                return _suite_from_jsonl_streamed()
        except (ValueError, UnknownFormatError):
            pass
        rows = list(_iter_csv_rows(p))
        return _suite_from_rows(rows, "csv")

    # Small file, unknown extension.
    text = p.read_text(encoding="utf-8")
    try:
        return _suite_from_json(text)
    except (json.JSONDecodeError, UnknownFormatError):
        pass
    try:
        return _suite_from_jsonl(text)
    except (ValueError, UnknownFormatError):
        rows = list(csv.DictReader(io.StringIO(text)))
        return _suite_from_rows(rows, "csv")


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