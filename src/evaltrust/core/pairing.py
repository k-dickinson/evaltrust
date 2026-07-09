"""Pair two single-model evaluation files into one A-vs-B comparison.

Single-system tools (DeepEval, LangSmith, OpenEvals) export one model per run;
this pairs two such files by example id. Multi-model files are audited directly.
"""

from __future__ import annotations

from .schema import EvalData, Example


def primary_model(data: EvalData) -> str:
    """The single model in a one-model file, or an error explaining the fix."""
    if len(data.models) != 1:
        raise ValueError(
            f"Expected one model per file for a two-file comparison, but "
            f"'{data.source_format}' contained {len(data.models)} "
            f"({', '.join(map(str, data.models))}). Audit this file on its own "
            f"instead; it already has models to compare."
        )
    return data.models[0]


def merge_two(
    data_a: EvalData, data_b: EvalData, label_a: str, label_b: str
) -> EvalData:
    """Merge two one-model files, keeping examples present in both."""
    model_a = primary_model(data_a)
    model_b = primary_model(data_b)
    b_by_id = {ex.id: ex for ex in data_b.examples}

    examples: list[Example] = []
    for exa in data_a.examples:
        exb = b_by_id.get(exa.id)
        if exb is None or model_a not in exa.scores or model_b not in exb.scores:
            continue

        scores = {label_a: exa.scores[model_a], label_b: exb.scores[model_b]}

        runs = {}
        if exa.runs and model_a in exa.runs:
            runs[label_a] = exa.runs[model_a]
        if exb.runs and model_b in exb.runs:
            runs[label_b] = exb.runs[model_b]

        judges: dict[str, dict[str, float]] = {}
        for src_ex, model, label in ((exa, model_a, label_a),
                                     (exb, model_b, label_b)):
            if src_ex.judges:
                for judge, per_model in src_ex.judges.items():
                    if model in per_model:
                        judges.setdefault(judge, {})[label] = per_model[model]

        examples.append(Example(id=exa.id, scores=scores,
                                runs=runs or None, judges=judges or None))

    if not examples:
        raise ValueError(
            "The two files share no common example ids, so they can't be paired. "
            "Make sure both evaluations used the same examples with matching ids."
        )

    return EvalData(
        models=[label_a, label_b],
        examples=examples,
        source_format=f"{data_a.source_format}+{data_b.source_format}",
        metadata={},
    )
