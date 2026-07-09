"""Shared ingestion primitives.

Normalises every tool's output to a stream of ``Record``s and groups them, so each
adapter only has to say where the rows are and what the columns are called.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from ..core.schema import EvalData, Example

# Alias tables shared by the structural adapters. Lower-cased on lookup.
ID_KEYS = ("id", "example_id", "test_id", "case_id", "testidx", "test_idx",
           "index", "idx", "input", "question", "prompt", "test", "query")
MODEL_KEYS = ("model", "provider", "system", "variant", "candidate", "engine",
              "providerid", "provider_id", "model_name", "label", "name")
SCORE_KEYS = ("score", "pass", "passed", "success", "correct", "result",
              "value", "metric_score", "rating", "grade", "reward")
JUDGE_KEYS = ("judge", "evaluator", "grader", "rater", "judge_model")
METRIC_KEYS = ("metric", "metric_name", "criterion", "dimension", "check_name",
               "aspect")

DEFAULT_METRIC = "score"

_TRUE = {"pass", "passed", "true", "yes", "correct", "success", "y", "t", "1", "win"}
_FALSE = {"fail", "failed", "false", "no", "incorrect", "failure", "n", "f", "0", "loss"}


@dataclass(frozen=True)
class Record:
    """One (example, model, score) observation for a named metric.

    ``metric`` lets a single file carry several metrics per example (correctness,
    safety, ...). When there is only one metric it defaults to ``"score"``.
    """

    example_id: str
    model: str
    score: float
    judge: str | None = None
    metric: str = DEFAULT_METRIC


def coerce_score(raw) -> float:
    """Turn the many spellings of a score into a float.

    Accepts numbers, booleans, numeric strings, and pass/fail-style words. Raises
    on anything it can't confidently interpret, rather than guessing.
    """
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        try:
            return float(s)
        except ValueError:
            pass
        if s in _TRUE:
            return 1.0
        if s in _FALSE:
            return 0.0
    raise ValueError(f"Cannot interpret {raw!r} as a score")


def records_to_evaldata(
    records: list[Record], source_format: str, metadata: dict | None = None
) -> EvalData:
    """Group flat records into canonical examples.

    Per (example, model): judge records become a per-judge map (score = mean over
    judges); otherwise repeated records are runs (score = mean over runs).
    """
    if not records:
        raise ValueError("No records found to build an evaluation from")

    # example_id -> model -> {"plain": [scores], "judges": {judge: score}}
    grouped: OrderedDict[str, OrderedDict[str, dict]] = OrderedDict()
    model_order: list[str] = []

    for rec in records:
        models = grouped.setdefault(rec.example_id, OrderedDict())
        cell = models.setdefault(rec.model, {"plain": [], "judges": OrderedDict()})
        if rec.model not in model_order:
            model_order.append(rec.model)
        if rec.judge is not None:
            cell["judges"][rec.judge] = rec.score
        else:
            cell["plain"].append(rec.score)

    examples = []
    for ex_id, models in grouped.items():
        scores: dict[str, float] = {}
        runs: dict[str, list[float]] = {}
        judges: dict[str, dict[str, float]] = {}

        for model, cell in models.items():
            if cell["judges"]:
                for judge, sc in cell["judges"].items():
                    judges.setdefault(judge, {})[model] = sc
                scores[model] = float(np.mean(list(cell["judges"].values())))
            else:
                vals = cell["plain"]
                if len(vals) > 1:
                    runs[model] = vals
                scores[model] = float(np.mean(vals))

        examples.append(Example(
            id=ex_id,
            scores=scores,
            runs=runs or None,
            judges=judges or None,
        ))

    return EvalData(
        models=model_order,
        examples=examples,
        source_format=source_format,
        metadata=metadata or {},
    )


def records_to_suite(
    records: list[Record], source_format: str, metadata: dict | None = None
) -> "OrderedDict[str, EvalData]":
    """Split records by metric into one canonical dataset per metric.

    A file with a single metric yields a one-entry suite keyed ``"score"``, so the
    same code path handles single- and multi-metric inputs. Metric order follows
    first appearance.
    """
    if not records:
        raise ValueError("No records found to build an evaluation from")

    by_metric: "OrderedDict[str, list[Record]]" = OrderedDict()
    for rec in records:
        by_metric.setdefault(rec.metric, []).append(rec)

    return OrderedDict(
        (metric, records_to_evaldata(recs, source_format, metadata))
        for metric, recs in by_metric.items()
    )
