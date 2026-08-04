"""Hugging Face Lighteval details adapter.

Reads Lighteval's per-sample ``Detail`` records. Each row carries a document,
a metric dict of sample-level scores, and optionally a model response. JSON
exports of ``--save-details`` output use either the column names observed in
recent official reference Parquet files (``doc``, ``metric``,
``model_response``) or the underscore-prefixed names in the current public
Hugging Face documentation (``__doc__``, ``__metric__``,
``__model_response__``). Both exact aliases are accepted.

Lighteval's aggregate ``results_{timestamp}.json`` (``config_general``,
``results``, ``config_tasks``, …) holds task-level summaries only. It is
recognised so ingestion can fail with a specific error rather than falling
through to a generic adapter, but it cannot be mapped to per-example canonical
rows.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from ..core.schema import EvalData
from .common import Record, coerce_score, records_to_suite

# Top-level keys on the aggregate results file (EvaluationTracker.save_results).
_AGGREGATE_KEYS = frozenset(
    {"config_general", "results", "versions", "config_tasks", "summary_tasks", "summary_general"}
)

# Keys that are never per-sample detail task lists when flattening a dict export.
_NON_DETAIL_KEYS = _AGGREGATE_KEYS | {"details", "summary_general"}

# Exact field-name aliases: recent official Parquet columns, and the current
# public documentation's underscore-prefixed names. No other variants.
_DOC_KEYS = ("doc", "__doc__")
_METRIC_KEYS = ("metric", "__metric__")
_MODEL_RESPONSE_KEYS = ("model_response", "__model_response__")

# Preferred metric for the single-audit path, in priority order.
_PRIMARY_METRICS = (
    "acc",
    "exact_match",
    "em",
    "qem",
    "pass@1",
    "maj@1",
)

_DEFAULT_MODEL = "model"


@dataclass(frozen=True)
class _DetailFields:
    """Alias-resolved fields from one Lighteval detail row."""

    doc: dict
    metric: dict
    model_response: dict | None = None


def _is_aggregate_results(raw: dict) -> bool:
    """Return True when *raw* looks like Lighteval's aggregate results.json."""
    if not isinstance(raw, dict):
        return False
    cg = raw.get("config_general")
    if not isinstance(cg, dict):
        return False
    # lighteval_sha is the strongest structural marker for this format.
    if "lighteval_sha" not in cg:
        return False
    results = raw.get("results")
    config_tasks = raw.get("config_tasks")
    return isinstance(results, dict) and isinstance(config_tasks, dict)


def _resolve_aliased_field(entry: dict, keys: tuple[str, ...]):
    """Return the value for an exact alias pair, or raise on conflict.

    If both aliases are present and equal, either value is returned. If both
    are present but differ, raise ``ValueError`` so the row is not silently
    accepted. If neither is present, return ``None``.
    """
    present = [(key, entry[key]) for key in keys if key in entry]
    if not present:
        return None
    if len(present) == 1:
        return present[0][1]
    (_, left), (_, right) = present[0], present[1]
    if left == right:
        return left
    raise ValueError(
        f"conflicting Lighteval detail fields {keys[0]!r} and {keys[1]!r}"
    )


def _extract_detail_fields(entry: dict) -> _DetailFields | None:
    """Resolve doc/metric/(optional) model_response via exact aliases.

    Returns ``None`` when required fields are absent or not dicts (not a detail
    row). Raises ``ValueError`` when both aliases of a field are present and
    conflict — callers treat that as a malformed row.
    """
    if not isinstance(entry, dict):
        return None
    doc = _resolve_aliased_field(entry, _DOC_KEYS)
    metric = _resolve_aliased_field(entry, _METRIC_KEYS)
    if not isinstance(doc, dict) or not isinstance(metric, dict):
        return None
    model_response = _resolve_aliased_field(entry, _MODEL_RESPONSE_KEYS)
    if model_response is not None and not isinstance(model_response, dict):
        # Present but not a mapping: treat as absent optional field rather
        # than rejecting a otherwise-valid scored row.
        model_response = None
    return _DetailFields(doc=doc, metric=metric, model_response=model_response)


def _looks_like_detail(entry: dict) -> bool:
    """Return True when *entry* matches a Lighteval Detail dict."""
    try:
        fields = _extract_detail_fields(entry)
    except ValueError:
        # Conflicting aliases are still structurally Lighteval-shaped; detect
        # so parse can skip with a counted malformed row rather than falling
        # through to Generic.
        return _has_any_detail_alias(entry)
    if fields is None:
        return False
    # Doc fingerprint: query + choices + gold_index (see requests.Doc).
    if not all(k in fields.doc for k in ("query", "choices", "gold_index")):
        return False
    # At least one numeric metric score must be present for detection parity
    # with parse().
    return any(_metric_score(fields.metric, name) is not None for name in fields.metric)


def _has_any_detail_alias(entry: dict) -> bool:
    """True when *entry* carries both a doc-shaped and metric-shaped alias pair.

    Used only so a conflicting-alias row is still claimed by this adapter
    (and then skipped) instead of being mis-routed.
    """
    if not isinstance(entry, dict):
        return False
    has_doc = any(k in entry and isinstance(entry[k], dict) for k in _DOC_KEYS)
    has_metric = any(k in entry and isinstance(entry[k], dict) for k in _METRIC_KEYS)
    if not (has_doc and has_metric):
        return False
    # Prefer a real doc fingerprint when available so bare {"doc": {}, ...}
    # does not claim unrelated JSON.
    for key in _DOC_KEYS:
        doc = entry.get(key)
        if isinstance(doc, dict) and all(
            k in doc for k in ("query", "choices", "gold_index")
        ):
            return True
    return False


def _metric_score(metric: dict, name: str) -> float | None:
    if name.endswith("_stderr"):
        return None
    raw = metric.get(name)
    if raw is None:
        return None
    try:
        return coerce_score(raw)
    except (ValueError, TypeError):
        return None


def _example_id(doc: dict, task_hint: str | None = None) -> str | None:
    doc_id = doc.get("id")
    if doc_id is None:
        return None
    task = doc.get("task_name") or task_hint
    ex_id = str(doc_id)
    if isinstance(task, str) and task:
        return f"{task}:{ex_id}"
    return ex_id


def _model_name(raw) -> str:
    if isinstance(raw, dict):
        cg = raw.get("config_general")
        if isinstance(cg, dict):
            name = cg.get("model_name")
            if isinstance(name, str) and name.strip():
                return name.strip()
            mc = cg.get("model_config")
            if isinstance(mc, dict):
                name = mc.get("model_name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
    return _DEFAULT_MODEL


def _detail_rows(raw) -> list[tuple[dict, str | None]] | None:
    """Return ``(detail_dict, task_hint)`` pairs extracted from supported shapes."""
    if isinstance(raw, list):
        if not raw:
            return None
        if not any(
            isinstance(row, dict) and _looks_like_detail(row) for row in raw[:10]
        ):
            return None
        return [(row, None) for row in raw if isinstance(row, dict)]

    if not isinstance(raw, dict):
        return None

    details = raw.get("details")
    if isinstance(details, list) and details:
        if any(
            isinstance(row, dict) and _looks_like_detail(row) for row in details[:10]
        ):
            return [(row, None) for row in details if isinstance(row, dict)]

    rows: list[tuple[dict, str | None]] = []
    for key, value in raw.items():
        if key in _NON_DETAIL_KEYS:
            continue
        if not isinstance(value, list) or not value or not isinstance(value[0], dict):
            continue
        if not any(
            isinstance(row, dict) and _looks_like_detail(row) for row in value[:10]
        ):
            continue
        task_hint = str(key) if "|" in str(key) or ":" in str(key) else None
        for row in value:
            if isinstance(row, dict):
                rows.append((row, task_hint))

    return rows or None


def _parse_to_records(
    raw,
    model: str,
) -> tuple[list[Record], int]:
    pairs = _detail_rows(raw)
    if pairs is None:
        raise ValueError("Not a Lighteval per-sample details export")

    records: list[Record] = []
    skipped = 0

    for entry, task_hint in pairs:
        try:
            fields = _extract_detail_fields(entry)
        except ValueError:
            # Conflicting aliases: skip and count, do not pick a side.
            skipped += 1
            continue
        if fields is None:
            skipped += 1
            continue

        ex_id = _example_id(fields.doc, task_hint)
        if ex_id is None:
            skipped += 1
            continue

        row_had_record = False
        bad_metrics = 0
        for name in fields.metric:
            score = _metric_score(fields.metric, name)
            if score is None:
                if fields.metric.get(name) is not None:
                    bad_metrics += 1
                continue
            records.append(Record(ex_id, model, score, metric=str(name)))
            row_had_record = True

        if not row_had_record:
            skipped += 1
        else:
            skipped += bad_metrics

    return records, skipped


def _pick_primary_metric(suite: "OrderedDict[str, EvalData]") -> str | None:
    keys = suite.keys()
    for preferred in _PRIMARY_METRICS:
        if preferred in keys:
            return preferred
    return next(iter(keys), None)


class LightevalAdapter:
    source_format = "lighteval"

    def detect(self, raw) -> bool:
        if _is_aggregate_results(raw):
            return True
        pairs = _detail_rows(raw)
        return pairs is not None and len(pairs) > 0

    def _to_suite(self, raw) -> "OrderedDict[str, EvalData]":
        if _is_aggregate_results(raw):
            raise ValueError(
                "Lighteval results.json contains only aggregate task metrics "
                "(config_general, results, config_tasks, …). EvalTrust needs "
                "per-example scores. Re-run with --save-details and point "
                "evaltrust at a JSON details export (a list of "
                "{doc|__doc__, metric|__metric__, model_response|__model_response__} "
                "rows, or a task-grouped dict), or convert details_*.parquet "
                "to that shape."
            )

        model = _model_name(raw)
        records, skipped = _parse_to_records(raw, model)
        if not records:
            raise ValueError(
                "No parsable per-sample metrics found in the Lighteval details "
                "export. Expected objects with a doc/__doc__ (query, choices, "
                "gold_index, id) and a metric/__metric__ dict of numeric scores."
            )
        return records_to_suite(
            records, self.source_format, {"skipped_rows": skipped}
        )

    def parse(self, raw) -> EvalData:
        suite = self._to_suite(raw)
        primary = _pick_primary_metric(suite)
        if primary is not None and primary in suite:
            return suite[primary]
        return next(iter(suite.values()))

    def parse_suite(self, raw) -> "OrderedDict[str, EvalData]":
        return self._to_suite(raw)
