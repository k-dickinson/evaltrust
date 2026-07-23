"""HELM per-instance results adapter.

Reads HELM's ``per_instance_stats.json`` — a list of per-instance blocks where
each entry has an ``instance_id`` and a ``stats`` list.  Each stat object has a
``name`` sub-object (with a ``name`` field for the metric name, and optionally
``split``) and a scalar ``mean`` value.

The model name is generic (``"model"``) because a HELM run covers one model per
run directory; users compare two runs with ``evaltrust audit runA.json runB.json``.

On the single-audit path ``parse()`` uses only the first recognised correctness
metric (``exact_match``, ``quasi_exact_match``, or the first numeric stat found).
On the suite path ``parse_suite()`` fans every named metric out into its own
``EvalData``.

Detection fingerprint: a non-empty list whose entries carry both ``instance_id``
and a ``stats`` list of ``{name: {...}, ...}`` objects — HELM's per-instance
signature as described in the issue comment.
"""

from __future__ import annotations

from collections import OrderedDict

from ..core.schema import EvalData
from .common import Record, coerce_score, records_to_evaldata, records_to_suite

# Metric names HELM commonly uses to represent correctness, in preference order.
# When parse() needs to pick one metric for the single-audit path it tries these
# in order and falls back to the first parsable stat it finds.
_CORRECTNESS_METRICS = (
    "exact_match",
    "quasi_exact_match",
    "f1_score",
    "rouge_l",
    "bleu_1",
)

_DEFAULT_MODEL = "model"


def _is_helm_entry(entry) -> bool:
    """Return True if *entry* looks like a HELM per-instance block."""
    if not isinstance(entry, dict):
        return False
    if "instance_id" not in entry:
        return False
    stats = entry.get("stats")
    if not isinstance(stats, list) or not stats:
        return False
    # At least one stat must be a dict carrying a nested ``name`` object.
    return any(
        isinstance(s, dict) and isinstance(s.get("name"), dict)
        for s in stats
    )


def _stat_name(stat: dict) -> str | None:
    """Extract the metric name string from a HELM stat object."""
    name_obj = stat.get("name")
    if not isinstance(name_obj, dict):
        return None
    name = name_obj.get("name")
    return str(name) if name is not None else None


def _stat_value(stat: dict) -> float | None:
    """Extract the numeric value from a HELM stat object; return None on failure."""
    raw = stat.get("mean")
    if raw is None:
        return None
    try:
        return coerce_score(raw)
    except (ValueError, TypeError):
        return None


def _parse_to_records(
    raw: list,
    model: str,
) -> tuple[list[Record], int]:
    """Convert raw HELM JSON to a flat list of Records.

    Returns ``(records, skipped_rows)`` where ``skipped_rows`` counts stat
    entries that were present but could not be parsed (no name, non-numeric
    mean).  Instance entries that yield no records at all are also counted once.
    """
    records: list[Record] = []
    skipped = 0

    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            # Not a HELM entry at all — not our data, not counted.
            continue

        instance_id = entry.get("instance_id")
        ex_id = str(instance_id) if instance_id is not None else str(idx)

        stats = entry.get("stats")
        if not isinstance(stats, list):
            skipped += 1
            continue

        entry_had_record = False
        entry_bad_stats = 0
        for stat in stats:
            if not isinstance(stat, dict):
                entry_bad_stats += 1
                continue
            metric = _stat_name(stat)
            if metric is None:
                entry_bad_stats += 1
                continue
            value = _stat_value(stat)
            if value is None:
                entry_bad_stats += 1
                continue
            records.append(Record(ex_id, model, value, metric=metric))
            entry_had_record = True

        if not entry_had_record:
            # Every stat in this entry was unparsable — count the whole instance
            # once (matching how OpenEvals counts a row with score=None as one
            # skipped row), not once per bad stat.
            skipped += 1
        else:
            # Instance produced at least one record; count only the bad stats
            # inside it (partial-skip, like lm-eval counts individual bad metrics).
            skipped += entry_bad_stats

    return records, skipped


def _pick_primary_metric(records: list[Record]) -> str | None:
    """Return the best single metric name for the single-audit path.

    Prefers known correctness metrics (in priority order) then falls back to the
    first metric that actually appears in the records.
    """
    present = dict.fromkeys(r.metric for r in records)  # preserves insertion order
    for preferred in _CORRECTNESS_METRICS:
        if preferred in present:
            return preferred
    return next(iter(present), None)


class HelmAdapter:
    """Adapter for HELM per-instance result files (``per_instance_stats.json``)."""

    source_format = "helm"

    def detect(self, raw) -> bool:
        """Return True iff *raw* looks like a HELM per-instance stats list."""
        if not isinstance(raw, list) or not raw:
            return False
        # At least one entry must carry HELM's per-instance fingerprint.
        return any(_is_helm_entry(entry) for entry in raw[:10])

    def _to_suite(self, raw: list) -> "OrderedDict[str, EvalData]":
        records, skipped = _parse_to_records(raw, _DEFAULT_MODEL)
        if not records:
            raise ValueError(
                "No parsable per-instance stats found in the HELM result file. "
                "Expected a list of objects with 'instance_id' and a 'stats' "
                "list of {name: {name: <str>}, mean: <float>} entries."
            )
        metadata = {"skipped_rows": skipped}
        return records_to_suite(records, self.source_format, metadata)

    def parse(self, raw) -> EvalData:
        """Single-audit path: return the primary correctness metric only."""
        suite = self._to_suite(raw)
        primary = _pick_primary_metric(
            [r for r in _flatten_suite_records(suite)]
        )
        if primary is not None and primary in suite:
            return suite[primary]
        # Fall back to the first metric in the suite.
        return next(iter(suite.values()))

    def parse_suite(self, raw) -> "OrderedDict[str, EvalData]":
        """Suite path: every HELM metric becomes its own EvalData."""
        return self._to_suite(raw)


def _flatten_suite_records(suite: "OrderedDict[str, EvalData]") -> list[Record]:
    """Reconstruct a flat record list from a suite (used to pick primary metric)."""
    records = []
    for metric, data in suite.items():
        for ex in data.examples:
            for model, score in ex.scores.items():
                records.append(Record(ex.id, model, score, metric=metric))
    return records
