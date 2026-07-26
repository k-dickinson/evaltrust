"""Langfuse score-export adapter.

Reads score rows from Langfuse's public Scores API. The current v3 response is
wrapped in ``{"data": [...], "meta": {...}}`` and identifies the scored trace
through ``subject``; legacy v2 exports use flat ``traceId`` fields (and may also
arrive ``{"data": [...], "meta": {...}}``-wrapped). Both shapes are accepted so
saved exports remain usable after a Langfuse upgrade.

One export represents one model/run.  Score names become metrics, and users
compare two exports with ``evaltrust audit runA.json runB.json``.

Two correctness hazards are guarded against explicitly rather than silently
mishandled:

* **Partial pages.** A ``meta`` block that still has pages left (v3's ``cursor``,
  or v2's ``page < totalPages``) means the export is incomplete; parsing it
  would silently audit a subset of the data. This is rejected with an
  actionable error instead.
* **Duplicate (trace, metric) scores.** A trace-level score and one or more
  observation-level scores sharing the same trace and metric name are not
  automatically the same measurement, and averaging/keeping-one would be a
  policy decision this adapter has no basis for. Duplicates raise rather than
  guess.

Schema references:
https://langfuse.com/docs/evaluation/scores/data-model
https://cloud.langfuse.com/generated/api/openapi.yml
"""

from __future__ import annotations

from collections import OrderedDict

from ..core.schema import EvalData
from .common import Record, coerce_score, records_to_suite

_SUPPORTED_DATA_TYPES = {"NUMERIC", "BOOLEAN", "CATEGORICAL"}
_UNSUPPORTED_DATA_TYPES = {"TEXT", "CORRECTION"}


def _score_rows(raw) -> list | None:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return raw["data"]
    return None


def _meta(raw) -> dict | None:
    if isinstance(raw, dict) and isinstance(raw.get("meta"), dict):
        return raw["meta"]
    return None


def _check_pagination(meta: dict) -> None:
    """Raise if ``meta`` shows this wrapped response is not itself the whole
    export.

    A single response's ``data`` array only ever holds that one page's rows -
    being the *last* page (v2's ``page == totalPages``) does not mean the
    earlier pages' scores are included too. v2's ``page``/``totalPages`` are
    exact counts, so completeness is only provable when there was only ever
    one page (``totalPages == 1``). v3's ``cursor`` can only prove the
    *opposite* - that a first/middle page has more results after it - it
    can't prove an independently saved last page also contains everything
    before it, since v3 carries no page count to check against. Either way,
    a genuinely combined multi-page export must be supplied as a bare JSON
    list (no `meta` wrapper), which bypasses this check entirely.
    """
    cursor = meta.get("cursor")
    if cursor:
        raise ValueError(
            "This Langfuse export is only one page of a paginated response "
            f"(meta.cursor = {cursor!r}, so at least one more page exists). "
            "Auditing a partial export would silently drop scores. Fetch every "
            "page by following meta.cursor, concatenate each page's `data` "
            "array into one combined bare JSON list (no `meta` wrapper), and "
            "run `evaltrust audit` on that."
        )

    page, total_pages = meta.get("page"), meta.get("totalPages")
    has_page_counts = (
        isinstance(page, (int, float)) and not isinstance(page, bool)
        and isinstance(total_pages, (int, float)) and not isinstance(total_pages, bool)
    )
    if has_page_counts and not (page == 1 and total_pages == 1):
        raise ValueError(
            f"This Langfuse export is page {page} of {total_pages} - even if "
            "it is the last page, its `data` array only holds that page's "
            "rows, not the earlier pages'. Auditing it would silently drop "
            "scores. Fetch every page, concatenate each page's `data` array "
            "into one combined bare JSON list (no `meta` wrapper), and run "
            "`evaltrust audit` on that."
        )


def _looks_like_score(row) -> bool:
    if not isinstance(row, dict):
        return False
    if not isinstance(row.get("name"), str):
        return False
    # A score row carries its value either numerically (`value`) or as text
    # (`stringValue` - used by CATEGORICAL, BOOLEAN, and TEXT/CORRECTION rows).
    # Requiring only `value` misses TEXT-only exports and CATEGORICAL rows that
    # never populate `value` at all, which would wrongly fall through to
    # "unrecognised format" instead of a specific unsupported-type error.
    if "value" not in row and "stringValue" not in row:
        return False

    # Legacy v2 exports expose the trace directly.  V3 can include a typed
    # subject, or omit it unless `fields=subject` was requested; recognize that
    # core v3 shape too so parse() can return a useful error instead of letting a
    # generic adapter claim it.
    if "traceId" in row or isinstance(row.get("subject"), dict):
        return True
    return (
        "dataType" in row
        and {"id", "projectId", "source"}.issubset(row)
    )


def _is_v3_row(row: dict) -> bool:
    """Distinguish a v3 (``ScoreV3``) row from a legacy flat (``Score``) row.

    v3 rows are identified by ``projectId`` (required on every ``BaseScoreV3``)
    or a typed ``subject``; legacy rows instead carry flat ``traceId`` /
    ``sessionId`` / ``observationId`` fields and never a ``projectId``. This
    matters because the two versions disagree on where a CATEGORICAL score's
    label lives: v3's ``value`` *is* the category string, while legacy exports
    put a (config-defaulted-to-zero) number in ``value`` and the label in
    ``stringValue``.
    """
    if "projectId" in row or isinstance(row.get("subject"), dict):
        return True
    if "traceId" in row or "sessionId" in row or "observationId" in row:
        return False
    return True


def _resolve_trace_id(row: dict) -> tuple[str | None, str | None]:
    """Return ``(trace_id, skip_reason)``. ``skip_reason`` is ``None`` on success."""
    trace_id = row.get("traceId")
    if trace_id is not None and str(trace_id):
        return str(trace_id), None

    subject = row.get("subject")
    if not isinstance(subject, dict):
        # No flat traceId and no subject at all: a v3 row exported without
        # `fields=subject`, or a shape missing trace info either way.
        return None, "missing_subject"

    kind = subject.get("kind")
    if kind == "trace" and subject.get("id") is not None:
        return str(subject["id"]), None
    if kind == "observation" and subject.get("traceId") is not None:
        return str(subject["traceId"]), None
    if kind in ("trace", "observation"):
        # A trace/observation subject missing the id it needs - treat like a
        # missing subject rather than guessing at a trace id.
        return None, "missing_subject"
    # session, experiment (dataset run), or any other subject kind: EvalTrust
    # compares models per trace and has no notion of a session/experiment score.
    return None, "unsupported_subject"


def _score_value(row: dict) -> float:
    data_type = str(row.get("dataType") or "").upper()
    if data_type in _UNSUPPORTED_DATA_TYPES:
        raise ValueError(f"Langfuse {data_type} scores are not numeric")
    if data_type and data_type not in _SUPPORTED_DATA_TYPES:
        raise ValueError(f"Unknown Langfuse score data type: {data_type}")

    if data_type == "CATEGORICAL":
        if _is_v3_row(row):
            # v3's CategoricalScoreV3.value IS the category string - there is no
            # stringValue field in v3 at all, and configId (when present) is
            # just provenance, not a signal to reinterpret value as a number.
            return coerce_score(row.get("value"))
        # Legacy CategoricalScore: value is a number that only means something
        # when configId links it to a score config; the schema says it
        # "defaults to 0" without one. Without a configId, fall back to the
        # human-readable stringValue and require it to coerce unambiguously (a
        # pass/fail-style word or a bare number) rather than trust a
        # zero-by-default value.
        if row.get("configId") is not None:
            return coerce_score(row.get("value"))
        return coerce_score(row.get("stringValue"))

    # NUMERIC and BOOLEAN both carry their value directly: numeric 0/1 for
    # legacy BOOLEAN, a JSON boolean for v3 BOOLEAN - coerce_score handles both.
    return coerce_score(row.get("value"))


def _no_usable_scores_error(reasons: dict[str, int], total_rows: int) -> ValueError:
    """Pick the most specific explanation for why nothing was usable."""
    if reasons.get("unsupported_subject"):
        return ValueError(
            "Every row in this Langfuse export scores a session or experiment "
            "rather than a trace (subject.kind is not 'trace' or 'observation'). "
            "EvalTrust compares models per trace and cannot audit session- or "
            "experiment-level scores; export trace-level scores instead."
        )
    if reasons.get("missing_subject") and not (
        reasons.get("unsupported_type") or reasons.get("ambiguous_value")
    ):
        return ValueError(
            "No usable trace-level scores found in the Langfuse export. For "
            "the v3 Scores API, request fields=subject so each score can be "
            "associated with its trace, then re-export."
        )
    if reasons.get("unsupported_type") and not (
        reasons.get("missing_subject") or reasons.get("ambiguous_value")
    ):
        return ValueError(
            "Every score in this Langfuse export has an unsupported data type. "
            "TEXT and CORRECTION scores are free-text, not numeric, and are "
            "always skipped; only NUMERIC, BOOLEAN, and CATEGORICAL scores can "
            "be audited."
        )
    return ValueError(
        "No usable trace-level scores found in the Langfuse export "
        f"({total_rows} row(s) examined). For the v3 Scores API, request "
        "fields=subject; only NUMERIC, BOOLEAN, and coercible CATEGORICAL "
        "scores can be audited - TEXT/CORRECTION scores and ambiguous "
        "categorical values are skipped."
    )


class LangfuseAdapter:
    source_format = "langfuse"

    def detect(self, raw) -> bool:
        rows = _score_rows(raw)
        return bool(rows) and any(_looks_like_score(row) for row in rows)

    def _to_suite(self, raw) -> "OrderedDict[str, EvalData]":
        rows = _score_rows(raw)
        if not rows:
            raise ValueError("No Langfuse score rows found")

        meta = _meta(raw)
        if meta is not None:
            _check_pagination(meta)

        records: list[Record] = []
        seen_trace_metric: set[tuple[str, str]] = set()
        reasons: dict[str, int] = {}
        skipped = 0

        def _skip(reason: str) -> None:
            nonlocal skipped
            skipped += 1
            reasons[reason] = reasons.get(reason, 0) + 1

        for row in rows:
            if not isinstance(row, dict):
                _skip("malformed_row")
                continue

            metric = row.get("name")
            if not isinstance(metric, str) or not metric:
                _skip("missing_metric")
                continue

            trace_id, reason = _resolve_trace_id(row)
            if trace_id is None:
                _skip(reason or "missing_subject")
                continue

            try:
                score = _score_value(row)
            except (TypeError, ValueError):
                data_type = str(row.get("dataType") or "").upper()
                if data_type in _UNSUPPORTED_DATA_TYPES or (
                    data_type and data_type not in _SUPPORTED_DATA_TYPES
                ):
                    _skip("unsupported_type")
                else:
                    _skip("ambiguous_value")
                continue

            key = (trace_id, metric)
            if key in seen_trace_metric:
                raise ValueError(
                    f"Found more than one score named {metric!r} for trace "
                    f"{trace_id!r}. This usually means a trace-level score and "
                    "one or more observation-level scores share the same trace "
                    "and metric name. EvalTrust does not average, discard, or "
                    "pick one of them - combining them requires an explicit "
                    "aggregation policy (max, latest, per-observation, ...) that "
                    "needs maintainer input. De-duplicate to one score per "
                    "(trace, metric) before running `evaltrust audit`."
                )
            seen_trace_metric.add(key)
            records.append(Record(trace_id, "model", score, metric=metric))

        if not records:
            raise _no_usable_scores_error(reasons, len(rows))
        return records_to_suite(
            records, self.source_format, {"skipped_rows": skipped}
        )

    def parse(self, raw) -> EvalData:
        # Single-audit path: use the first score name as the metric.
        return next(iter(self._to_suite(raw).values()))

    def parse_suite(self, raw) -> "OrderedDict[str, EvalData]":
        # Suite path: every score name becomes its own metric.
        return self._to_suite(raw)
