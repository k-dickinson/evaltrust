"""HELM per-instance results adapter.

Reads HELM's ``per_instance_stats.json`` — a list of per-instance blocks where
each entry has an ``instance_id`` and a ``stats`` list.  Each stat object has a
``name`` sub-object (with a ``name`` field for the metric name, and optionally
``split``) and a scalar ``mean`` value.

The model name is generic (``"model"``) because a HELM run covers one model per
run directory; users compare two runs with ``evaltrust audit runA.json runB.json``.

On the single-audit path ``parse()`` uses only the first recognised correctness
metric (``exact_match``, ``quasi_exact_match``, or the first quality stat found).
On the suite path ``parse_suite()`` fans every *quality* metric out into its own
``EvalData``, filtering HELM's bookkeeping stats (``num_trials``,
``num_prompt_tokens``, ``finish_reason_*``, etc.) which are constant across
instances and carry no quality signal.

Detection fingerprint: a non-empty list whose entries carry both ``instance_id``
and a ``stats`` list of ``{name: {...}, ...}`` objects — HELM's per-instance
signature as described in the issue comment.
"""

from __future__ import annotations

from collections import OrderedDict

from ..core.schema import EvalData
from .common import Record, coerce_score, records_to_evaldata, records_to_suite

# ---------------------------------------------------------------------------
# Metric classification
# ---------------------------------------------------------------------------

# Metric names HELM commonly uses to represent correctness, in preference order.
# When parse() needs to pick one metric for the single-audit path it tries these
# in order and falls back to the first quality stat it finds.
_CORRECTNESS_METRICS = (
    "exact_match",
    "quasi_exact_match",
    "f1_score",
    "rouge_l",
    "bleu_1",
)

# HELM bookkeeping stats that are constant across instances and carry no quality
# signal.  Real files include many more (num_prompt_tokens, finish_reason_*,
# num_references, num_completions, …).  We filter these out of parse_suite() so
# a suite audit isn't flooded with meaningless comparisons.  The single-audit
# path is already safe because it prefers known correctness metrics.
_BOOKKEEPING_STATS: frozenset[str] = frozenset(
    {
        "num_trials",
        "num_prompt_tokens",
        "num_output_tokens",
        "num_references",
        "num_completions",
        "num_instances",
        "num_train_instances",
        "prompt_truncated",
        "finish_reason_length",
        "finish_reason_stop",
        "finish_reason_endoftext",
        "finish_reason_unknown",
    }
)


def _is_quality_stat(metric_name: str) -> bool:
    """Return True if *metric_name* is a real quality signal, not bookkeeping.

    Rejects exact members of ``_BOOKKEEPING_STATS`` and any name that starts
    with ``finish_reason_`` (HELM generates one per distinct finish reason and
    the set varies by model/scenario).
    """
    if metric_name in _BOOKKEEPING_STATS:
        return False
    if metric_name.startswith("finish_reason_"):
        return False
    return True


_DEFAULT_MODEL = "model"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
    quality_only: bool = False,
) -> tuple[list[Record], int]:
    """Convert raw HELM JSON to a flat list of Records.

    Parameters
    ----------
    raw:
        The parsed ``per_instance_stats.json`` list.
    model:
        The model label to attach to every record (always ``"model"``).
    quality_only:
        When True, skip bookkeeping stats so only quality metrics are included.
        Used by the suite path; the single-audit path leaves this False because
        it selects the primary metric afterwards.

    Returns
    -------
    ``(records, skipped_rows)`` where ``skipped_rows`` counts:
    - individual stat entries that could not be parsed (bad name, non-numeric mean)
    - whole instance entries that yield zero records (missing/bad stats list, or
      missing instance_id)
    """
    records: list[Record] = []
    skipped = 0

    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            # Not a HELM entry at all — not our data, not counted.
            continue

        instance_id = entry.get("instance_id")
        if instance_id is None:
            # Missing instance_id: skip + count the whole entry to avoid
            # synthetic index-based IDs colliding with real instance_ids.
            skipped += 1
            continue
        ex_id = str(instance_id)

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
            if quality_only and not _is_quality_stat(metric):
                # Silently drop bookkeeping stats — they are not errors.
                continue
            value = _stat_value(stat)
            if value is None:
                entry_bad_stats += 1
                continue
            records.append(Record(ex_id, model, value, metric=metric))
            entry_had_record = True

        if not entry_had_record:
            # Every stat in this entry was unparsable (or filtered) — count the
            # whole instance once, not once per bad stat.
            skipped += 1
        else:
            # Instance produced at least one record; count only the bad stats.
            skipped += entry_bad_stats

    return records, skipped


def _pick_primary_metric(suite: "OrderedDict[str, EvalData]") -> str | None:
    """Return the best single metric name for the single-audit path.

    Reads ``suite.keys()`` directly — no need to reconstruct records from the
    already-grouped suite.  Prefers known correctness metrics (in priority order)
    then falls back to the first key present.
    """
    keys = suite.keys()
    for preferred in _CORRECTNESS_METRICS:
        if preferred in keys:
            return preferred
    return next(iter(keys), None)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class HelmAdapter:
    """Adapter for HELM per-instance result files (``per_instance_stats.json``)."""

    source_format = "helm"

    def detect(self, raw) -> bool:
        """Return True iff *raw* looks like a HELM per-instance stats list."""
        if not isinstance(raw, list) or not raw:
            return False
        # At least one entry must carry HELM's per-instance fingerprint.
        return any(_is_helm_entry(entry) for entry in raw[:10])

    def _to_suite(
        self, raw: list, quality_only: bool = False
    ) -> "OrderedDict[str, EvalData]":
        """Shared parsing core used by both ``parse`` and ``parse_suite``."""
        records, skipped = _parse_to_records(raw, _DEFAULT_MODEL, quality_only=quality_only)
        if not records:
            raise ValueError(
                "No parsable per-instance stats found in the HELM result file. "
                "Expected a list of objects with 'instance_id' and a 'stats' "
                "list of {name: {name: <str>}, mean: <float>} entries."
            )
        metadata = {"skipped_rows": skipped}
        return records_to_suite(records, self.source_format, metadata)

    def parse(self, raw) -> EvalData:
        """Single-audit path: return the primary correctness metric only.

        Builds the full suite (all stats, no bookkeeping filter) so that
        correctness metrics are always available even in mixed files, then picks
        the best one by reading ``suite.keys()`` directly.
        """
        suite = self._to_suite(raw, quality_only=False)
        primary = _pick_primary_metric(suite)
        if primary is not None and primary in suite:
            return suite[primary]
        # Fall back to the first metric in the suite.
        return next(iter(suite.values()))

    def parse_suite(self, raw) -> "OrderedDict[str, EvalData]":
        """Suite path: every *quality* HELM metric becomes its own EvalData.

        Bookkeeping stats (``num_trials``, ``num_prompt_tokens``,
        ``finish_reason_*``, etc.) are silently filtered so a suite audit only
        surfaces meaningful comparisons.
        """
        return self._to_suite(raw, quality_only=True)
