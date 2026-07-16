"""Generic JSON adapters: native nested output and flat record lists.

These cover the long tail: anything that isn't a recognised tool but is still,
structurally, "examples with per-model scores." Between them and the CSV adapter,
scores in almost any shape can be audited without reformatting.
"""

from __future__ import annotations

from ..core.schema import EvalData, Example, Preference
from .common import (
    DEFAULT_PREFERENCE_JUDGE,
    DEFAULT_METRIC,
    ID_KEYS,
    JUDGE_KEYS,
    METRIC_KEYS,
    MODEL_KEYS,
    PREFERENCE_KEYS,
    PreferenceRecord,
    Record,
    SCORE_KEYS,
    coerce_score,
    records_to_evaldata,
)

# Keys under which tools commonly nest a list of result rows.
_LIST_WRAPPERS = ("results", "data", "rows", "outputs", "samples", "records",
                  "predictions", "evaluations")


def _first_alias(keys, aliases) -> str | None:
    """Return the actual key whose lower-cased name is in ``aliases``."""
    lowered = {str(k).lower(): k for k in keys}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _find_record_list(raw) -> list | None:
    """Locate a list of dict rows, either raw itself or nested under a wrapper."""
    if isinstance(raw, list) and raw and all(isinstance(r, dict) for r in raw):
        return raw
    if isinstance(raw, dict):
        for key in _LIST_WRAPPERS:
            got = _first_alias(raw.keys(), (key,))
            if got and isinstance(raw[got], list) and raw[got] \
                    and all(isinstance(r, dict) for r in raw[got]):
                return raw[got]
    return None


def _coerce_preference(
    raw, declared_models: tuple[str, ...] = ()
) -> str | Preference:
    """Map the reserved tie token while preserving string model ids."""
    if raw is Preference.TIE:
        return raw
    if isinstance(raw, str):
        value = raw.strip()
        if value.casefold() == Preference.TIE.value:
            if any(
                model.casefold() == Preference.TIE.value
                for model in declared_models
            ):
                raise ValueError(
                    f"Preference value {value!r} is ambiguous because a declared "
                    "model is also named 'tie'. Rename that model for generic "
                    "preference ingestion, or use the typed EvalData schema."
                )
            return Preference.TIE
        if value:
            if declared_models and value not in declared_models:
                raise ValueError(
                    f"unknown preference winner {value!r}; known models are "
                    f"{list(declared_models)!r}"
                )
            return value
    raise ValueError(f"Cannot interpret {raw!r} as a pairwise preference")


def _present(value) -> bool:
    return value is not None and not (
        isinstance(value, str) and not value.strip())


def _is_preference_column(rows, key) -> bool:
    """Distinguish a preference alias from a wide score model with that name."""
    values = [row.get(key) for row in rows if _present(row.get(key))]
    if not values:
        return False
    for value in values:
        if value is Preference.TIE:
            return True
        if not isinstance(value, str):
            return False
        if value.strip().casefold() == Preference.TIE.value:
            return True
        try:
            coerce_score(value)
        except ValueError:
            continue
        return False
    return True


def _known_models(rows, model_key, columns, preference_key=None) -> tuple[str, ...]:
    """Return explicit long-format models or wide columns with actual scores."""
    if model_key:
        models = []
        for row in rows:
            value = row.get(model_key)
            if _present(value) and str(value) not in models:
                models.append(str(value))
        if models:
            return tuple(models)

    preference_models = set()
    if preference_key:
        for row in rows:
            value = row.get(preference_key)
            if (
                isinstance(value, str)
                and value.strip()
                and value.strip().casefold() != Preference.TIE.value
            ):
                preference_models.add(value.strip())
    models = []
    for column in columns:
        if str(column) in preference_models:
            models.append(str(column))
            continue
        for row in rows:
            try:
                coerce_score(row.get(column))
            except ValueError:
                continue
            models.append(str(column))
            break
    return tuple(models)


def dicts_to_records(
    rows: list[dict], skipped: list | None = None
) -> list[Record | PreferenceRecord]:
    """Extract (example, model, score) records from dict rows.

    Handles long format (model + score columns) and wide format (one score column
    per model, non-numeric columns ignored). Rows with an unreadable score are
    skipped, with a reason appended to ``skipped`` if given.
    """
    keys = rows[0].keys()
    id_key = _first_alias(keys, ID_KEYS)
    model_key = _first_alias(keys, MODEL_KEYS)
    judge_key = _first_alias(keys, JUDGE_KEYS)
    score_key = _first_alias(keys, SCORE_KEYS)
    preference_candidate = _first_alias(keys, PREFERENCE_KEYS)
    preference_key = (
        preference_candidate
        if preference_candidate and _is_preference_column(rows, preference_candidate)
        else None
    )
    metric_key = _first_alias(keys, METRIC_KEYS)

    reserved = {k for k in (id_key, judge_key, metric_key) if k}
    if model_key and score_key:
        reserved.update((model_key, score_key))
    if preference_key:
        reserved.add(preference_key)
    wide_columns = [column for column in keys if column not in reserved]
    declared_models = (
        _known_models(rows, model_key, wide_columns, preference_key)
        if preference_key else ()
    )

    records: list[Record | PreferenceRecord] = []
    for idx, row in enumerate(rows):
        ex_id = str(row[id_key]) if id_key and row.get(id_key) is not None else str(idx)
        judge = str(row[judge_key]) if judge_key and row.get(judge_key) is not None else None
        metric = (str(row[metric_key]) if metric_key and row.get(metric_key) is not None
                  else DEFAULT_METRIC)

        if preference_key and _present(row.get(preference_key)):
            records.append(PreferenceRecord(
                ex_id,
                _coerce_preference(row[preference_key], declared_models),
                judge or DEFAULT_PREFERENCE_JUDGE,
                metric,
                declared_models,
            ))

        if model_key and score_key:
            try:
                score = coerce_score(row[score_key])
            except ValueError:
                if skipped is not None:
                    skipped.append(f"row {idx}: unreadable score {row.get(score_key)!r}")
                continue
            records.append(Record(ex_id, str(row[model_key]), score, judge, metric))
        else:
            for col, val in row.items():
                if col in reserved:
                    continue
                try:
                    score = coerce_score(val)
                except ValueError:
                    continue
                records.append(Record(ex_id, str(col), score, judge, metric))

    if not records:
        raise ValueError("No (example, model, score) records could be extracted")
    return records


class GenericRecordsAdapter:
    source_format = "generic"

    def detect(self, raw) -> bool:
        return _find_record_list(raw) is not None

    def parse(self, raw) -> EvalData:
        rows = _find_record_list(raw)
        if rows is None:
            raise ValueError("No record list found for the generic adapter")
        skipped: list = []
        records = dicts_to_records(rows, skipped)
        return records_to_evaldata(records, self.source_format,
                                   {"skipped_rows": len(skipped)})


def _coerce_runs(runs):
    """Coerce a model -> run-list mapping. A run list must stay aligned, so if any
    value in it is unreadable the whole model's runs are dropped. A non-dict block
    or non-iterable list is dropped too, never raised. Empty -> None."""
    if not isinstance(runs, dict) or not runs:
        return None
    out = {}
    for m, vs in runs.items():
        try:
            out[str(m)] = [coerce_score(v) for v in vs]
        except (ValueError, TypeError):   # unreadable value or non-iterable list
            continue
    return out or None


def _coerce_judges(judges):
    """Coerce a judge -> {model -> score} mapping, dropping any (judge, model)
    whose score is unreadable. A non-dict block or non-dict per-judge map is
    dropped too, never raised. Empty judges (or empty per-judge maps) -> None."""
    if not isinstance(judges, dict) or not judges:
        return None
    out = {}
    for j, mv in judges.items():
        if not isinstance(mv, dict):
            continue
        scored = {}
        for m, v in mv.items():
            try:
                scored[str(m)] = coerce_score(v)
            except (ValueError, TypeError):
                continue
        if scored:
            out[str(j)] = scored
    return out or None


class NativeNestedAdapter:
    """Structured JSON: {"examples": [{"id", "scores": {model: score}, ...}]}."""

    source_format = "native"

    def detect(self, raw) -> bool:
        return (
            isinstance(raw, dict)
            and isinstance(raw.get("examples"), list)
            and bool(raw["examples"])
            and isinstance(raw["examples"][0], dict)
            and "scores" in raw["examples"][0]
        )

    def parse(self, raw) -> EvalData:
        examples = []
        models: list[str] = []
        skipped = 0
        declared_models = tuple(str(model) for model in (raw.get("models") or ()))
        scored_models = tuple(dict.fromkeys(
            str(model)
            for example in raw["examples"]
            for model in example["scores"]
        ))
        for i, ex in enumerate(raw["examples"]):
            scores = {}
            for m, s in ex["scores"].items():
                try:
                    scores[str(m)] = coerce_score(s)
                except ValueError:
                    skipped += 1   # count only unreadable main scores
            for m in scores:
                if m not in models:
                    models.append(m)
            runs = _coerce_runs(ex.get("runs"))
            judges = _coerce_judges(ex.get("judges"))
            preferences = ex.get("preferences")
            if preferences is not None and not isinstance(preferences, dict):
                raise ValueError(
                    "Native example preferences must be a judge-to-winner map"
                )
            known_models = declared_models or scored_models
            preferences = ({
                str(judge): _coerce_preference(winner, known_models)
                for judge, winner in preferences.items()
            } if preferences else None)
            raw_attributes = ex.get("attributes")
            if raw_attributes is not None and not isinstance(raw_attributes, dict):
                raise ValueError(
                    "Native example attributes must be a mapping of tag names to values"
                )
            attributes = (
                {str(k): str(v) for k, v in raw_attributes.items()}
                if raw_attributes else None
            )
            examples.append(Example(id=str(ex.get("id", i)), scores=scores,
                                    runs=runs, judges=judges,
                                    preferences=preferences,
                                    attributes=attributes))

        if raw.get("models"):
            models = [str(m) for m in raw["models"]]
        metadata = dict(raw.get("metadata", {}))
        metadata["skipped_rows"] = skipped
        return EvalData(models=models, examples=examples,
                        source_format=self.source_format, metadata=metadata)
