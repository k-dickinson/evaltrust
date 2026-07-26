"""Load an evaluation file from disk and normalise it to canonical EvalData.

Reads the file, routes JSON through structural auto-detection and CSV through the
shared record extractor, and returns EvalData. The user never thinks about
formats.

Large-file streaming
--------------------
**JSONL and CSV** are read line-by-line via generator pipelines so the raw file
is never fully materialised as a single Python string.  The threshold is
``_STREAM_THRESHOLD`` bytes (default 64 MiB); files smaller than that are fully
materialised first (preserving the original behaviour) so the fast path stays
fast.

The win over the original ``Path.read_text()`` approach is *string elimination*:
the raw file bytes are never held as a single Python object.  Peak memory is
proportional to the list of parsed row dicts, not the raw file string.
Full single-pass O(1)-in-row-count streaming requires refactoring
``detect_line_adapter`` / ``dicts_to_records`` to accept a one-row lookahead
iterator; that is tracked in a TODO comment inside ``_records_from_jsonl_iter``.

**JSON** streaming is a best-effort enhancement only.  For the two common shapes
— a top-level array or ``{"examples": [...]}`` — the optional ``ijson`` library
is used when available, keeping peak memory proportional to the largest single
record.  When ``ijson`` is absent the file falls back to a full ``read_text()``
load and a warning is emitted.  The memory guarantee (peak bounded by buffer,
not file size) therefore applies to JSONL and CSV unconditionally, and to JSON
only when ``ijson`` is installed (``pip install 'evaltrust[streaming]'``).
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
    """Fallback record builder for wide-format and preference JSONL files.

    This function is only reached when ``_stream_records_from_jsonl`` cannot
    handle the file — specifically wide-format files (no ``model`` column,
    score columns are model names) and preference files.  Those formats require
    ``_known_models`` / ``_is_preference_column`` which must scan all rows to
    determine column layout, making true row-by-row streaming impossible without
    a deeper refactor of those interfaces.

    For the common long-format case (``model`` + ``score`` columns),
    ``_load_jsonl_streamed`` / ``_suite_from_jsonl_streamed`` route through
    ``_stream_records_from_jsonl`` instead, which processes rows one at a time.

    The iterator is materialised here into a ``list[dict]`` — one parsed object
    per row — which is meaningfully smaller than the raw file string eliminated
    by ``_iter_jsonl_lines``.
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
    """Stream a large JSONL file with minimal memory.

    For long-format files (``model`` + ``score`` columns) records are extracted
    row-by-row without ever building the full row list.  Wide-format and
    preference files fall back to ``_records_from_jsonl_iter`` which
    materialises a list of dicts (but never a raw file string).
    """
    result = _stream_records_from_jsonl(path)
    if result is not None:
        records, source_format, metadata = result
        return records_to_evaldata(records, source_format, metadata)
    # Fallback: wide-format or preference — materialise row dicts only.
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
# ijson helpers
# ---------------------------------------------------------------------------

def _ijson_import():
    """Import ijson or raise ImportError with a clear message."""
    try:
        import ijson  # type: ignore[import]
        return ijson
    except ImportError:
        return None


def _ijson_parse_errors(ijson):
    """Return the tuple of ijson-specific parse exception types.

    Catching only these (rather than bare ``Exception``) lets real I/O errors
    (``OSError``, ``PermissionError``, etc.) propagate to the caller.
    """
    # ijson guarantees JSONError; IncompleteJSONError is a subclass on all backends.
    errors = [ijson.JSONError]
    if hasattr(ijson, "IncompleteJSONError"):
        errors.append(ijson.IncompleteJSONError)
    return tuple(errors)


def _normalize_ijson_value(v):
    """Convert ijson Decimal numbers to float so coerce_score accepts them.

    ijson yields ``decimal.Decimal`` for all JSON numbers to preserve precision.
    ``coerce_score`` accepts ``int`` and ``float`` but not ``Decimal``, so we
    convert here at the boundary.  Non-numeric values are returned unchanged.
    """
    try:
        from decimal import Decimal
        if isinstance(v, Decimal):
            return float(v)
    except ImportError:
        pass
    return v


def _normalize_ijson_dict(d: dict) -> dict:
    """Recursively convert Decimal values in an ijson-parsed dict to float."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _normalize_ijson_dict(v)
        elif isinstance(v, list):
            out[k] = [
                _normalize_ijson_dict(i) if isinstance(i, dict)
                else _normalize_ijson_value(i)
                for i in v
            ]
        else:
            out[k] = _normalize_ijson_value(v)
    return out


def _peek_first_byte(path: Path) -> bytes:
    """Return the first non-whitespace byte of a file without reading it all."""
    with path.open("rb") as fh:
        while True:
            ch = fh.read(1)
            if not ch or not ch.strip():
                if not ch:
                    return b""
                continue
            return ch


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
    ijson = _ijson_import()
    if ijson is None:
        logger.warning(
            "File '%s' exceeds the %d MiB streaming threshold but 'ijson' is "
            "not installed. The file will be loaded fully into memory. "
            "Install it with: pip install 'evaltrust[streaming]'",
            path.name,
            _STREAM_THRESHOLD // (1024 * 1024),
        )
        return None

    parse_errors = _ijson_parse_errors(ijson)
    first_byte = _peek_first_byte(path)

    if first_byte == b"[":
        rows: list[dict] = []
        with path.open("rb") as fh:
            for item in ijson.items(fh, "item"):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Expected a JSON array of objects in '{path.name}', "
                        f"got a {type(item).__name__} element."
                    )
                rows.append(_normalize_ijson_dict(item))
        if not rows:
            raise UnknownFormatError("The JSON file has no data rows.")
        return detect_adapter(rows).parse(rows)

    if first_byte == b"{":
        # Collect ALL top-level keys in one pass, then stream the examples array
        # in a second pass.  Stopping at the first "examples" key would silently
        # drop any top-level fields that appear after it in the file.
        top: dict = {}
        try:
            with path.open("rb") as fh:
                for key, value in ijson.kvitems(fh, ""):
                    if key != "examples":
                        top[key] = _normalize_ijson_value(value)
        except parse_errors:
            return None

        rows = []
        try:
            with path.open("rb") as fh:
                for item in ijson.items(fh, "examples.item"):
                    rows.append(_normalize_ijson_dict(item))
        except parse_errors:
            return None

        if not rows:
            return None

        raw_obj = dict(top)
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
    ijson = _ijson_import()
    if ijson is None:
        logger.warning(
            "File '%s' exceeds the %d MiB streaming threshold but 'ijson' is "
            "not installed. The suite will be loaded fully into memory. "
            "Install it with: pip install 'evaltrust[streaming]'",
            path.name,
            _STREAM_THRESHOLD // (1024 * 1024),
        )
        return None

    parse_errors = _ijson_parse_errors(ijson)
    first_byte = _peek_first_byte(path)

    if first_byte == b"[":
        rows: list[dict] = []
        with path.open("rb") as fh:
            for item in ijson.items(fh, "item"):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Expected a JSON array of objects in '{path.name}', "
                        f"got a {type(item).__name__} element."
                    )
                rows.append(_normalize_ijson_dict(item))
        if not rows:
            raise UnknownFormatError("The JSON file has no data rows.")
        skipped: list = []
        records = dicts_to_records(rows, skipped)
        return records_to_suite(records, "generic", {"skipped_rows": len(skipped)})

    if first_byte == b"{":
        # Collect ALL top-level keys first, then stream examples separately.
        top: dict = {}
        try:
            with path.open("rb") as fh:
                for key, value in ijson.kvitems(fh, ""):
                    if key != "examples":
                        top[key] = _normalize_ijson_value(value)
        except parse_errors:
            return None

        rows = []
        try:
            with path.open("rb") as fh:
                for item in ijson.items(fh, "examples.item"):
                    rows.append(_normalize_ijson_dict(item))
        except parse_errors:
            return None

        if not rows:
            return None

        raw_obj = dict(top)
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
    that the raw file is never held as a single string in memory.
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
        result = _stream_records_from_jsonl(p)
        if result is not None:
            records, source_format, metadata = result
        else:
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
        try:
            result = _suite_from_json_streamed(p)
            if result is not None:
                return result
        except (UnknownFormatError, ValueError):
            pass
        try:
            text = p.read_text(encoding="utf-8")
            return _suite_from_json(text)
        except (json.JSONDecodeError, UnknownFormatError):
            pass
        try:
            with p.open(encoding="utf-8") as fh:
                head = fh.read(256)
            if not head.lstrip().startswith("["):
                return _suite_from_jsonl_streamed()
        except (ValueError, UnknownFormatError):
            pass
        rows = list(_iter_csv_rows(p))
        return _suite_from_rows(rows, "csv")

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


# ---------------------------------------------------------------------------
# Streaming record extraction (avoids full list materialisation for long-format)
# ---------------------------------------------------------------------------

def _stream_records_from_jsonl(
    path: Path,
) -> tuple[list[Record], str, dict] | None:
    """Attempt to extract records from a large JSONL file with minimal memory.

    Peeks the first row to detect the file layout:

    * **lm-eval / openai-evals** (tool-specific adapters): these only inspect
      ``rows[0]`` for detection, so we peek one row, check, and if matched
      stream the rest row-by-row through the adapter's ``parse_lines``.
      Note: ``parse_lines`` still receives a list — but we build it
      incrementally so only the current row is ever held alongside the result
      list, not the raw file string.

    * **Long-format generic** (has ``model`` + ``score`` column in row 0):
      extract records row-by-row; column layout is fixed from row 0 so no
      second pass is needed.

    * **Wide-format / preference**: fall back to ``None`` so the caller
      materialises the full row list via ``_records_from_jsonl_iter``.

    Returns ``(records, source_format, metadata)`` or ``None`` on fallback.
    """
    from ..adapters.common import (
        DEFAULT_METRIC,
        coerce_score,
        Record,
    )
    from ..adapters.generic import (
        _first_alias,
        ID_KEYS,
        MODEL_KEYS,
        SCORE_KEYS,
        JUDGE_KEYS,
        METRIC_KEYS,
        PREFERENCE_KEYS,
    )

    gen = _iter_jsonl_lines(path)

    # Peek the first row.
    try:
        first = next(gen)
    except StopIteration:
        raise UnknownFormatError("The JSONL file has no data rows.")

    # --- Try lm-eval adapter (only checks rows[0]) ---
    lm_adapter = None
    from ..adapters.lm_eval import LMEvalAdapter
    _lm = LMEvalAdapter()
    if _lm.detect_lines([first]):
        lm_adapter = _lm

    # --- Try openai-evals (scans for a spec row; peek up to 50 rows) ---
    from ..adapters.openai_evals import OpenAIEvalsAdapter
    _oai = OpenAIEvalsAdapter()
    oai_adapter = None
    if not lm_adapter:
        # OpenAI Evals detection scans for a spec row anywhere in the file.
        # We can't do this without reading ahead, so fall back for this case.
        if _oai.detect_lines([first]):
            oai_adapter = _oai

    if lm_adapter or oai_adapter:
        adapter = lm_adapter or oai_adapter
        # Stream remaining rows into a list — still avoids the raw string.
        rows = [first] + list(gen)
        records, metadata = adapter.parse_lines(rows, path=path)
        return records, adapter.source_format, metadata

    # --- Generic long-format: model + score keys detectable from row 0 ---
    keys = first.keys()
    model_key = _first_alias(keys, MODEL_KEYS)
    score_key = _first_alias(keys, SCORE_KEYS)
    id_key = _first_alias(keys, ID_KEYS)
    judge_key = _first_alias(keys, JUDGE_KEYS)
    metric_key = _first_alias(keys, METRIC_KEYS)
    preference_candidate = _first_alias(keys, PREFERENCE_KEYS)

    # Wide-format or preference: need full scan — fall back.
    if not (model_key and score_key) or preference_candidate:
        return None

    # Long-format: stream row-by-row.
    records: list[Record] = []
    skipped: list[str] = []

    def _process_row(idx: int, row: dict) -> None:
        ex_id = (
            str(row[id_key])
            if id_key and row.get(id_key) is not None
            else str(idx)
        )
        judge = (
            str(row[judge_key])
            if judge_key and row.get(judge_key) is not None
            else None
        )
        metric = (
            str(row[metric_key])
            if metric_key and row.get(metric_key) is not None
            else DEFAULT_METRIC
        )
        try:
            score = coerce_score(row[score_key])
        except (ValueError, KeyError):
            skipped.append(f"row {idx}: unreadable score {row.get(score_key)!r}")
            return
        records.append(Record(ex_id, str(row[model_key]), score, judge, metric))

    _process_row(0, first)
    for i, row in enumerate(gen, start=1):
        _process_row(i, row)

    if not records:
        raise ValueError("No (example, model, score) records could be extracted")

    return records, "jsonl", {"skipped_rows": len(skipped)}