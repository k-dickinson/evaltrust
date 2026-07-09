"""Tests for pairing two single-system eval files into one A-vs-B comparison.

Tools like DeepEval evaluate one system per run, so their exports carry a single
model. To audit A vs B you point EvalTrust at two such files and it pairs them by
example id.
"""

import pytest

from evaltrust.core.pairing import merge_two, primary_model
from evaltrust.core.schema import EvalData, Example


def single(model, rows, runs=None, judges=None, fmt="test"):
    """rows: {id: score}. runs: {id: [..]}. judges: {id: {judge: score}}."""
    examples = []
    for ex_id, score in rows.items():
        ex_runs = {model: runs[ex_id]} if runs and ex_id in runs else None
        ex_judges = ({j: {model: s} for j, s in judges[ex_id].items()}
                     if judges and ex_id in judges else None)
        examples.append(Example(id=ex_id, scores={model: float(score)},
                                runs=ex_runs, judges=ex_judges))
    return EvalData(models=[model], examples=examples, source_format=fmt, metadata={})


def test_primary_model_returns_the_only_model():
    assert primary_model(single("gpt", {"q1": 1})) == "gpt"


def test_primary_model_rejects_multi_model_files():
    data = EvalData(models=["A", "B"],
                    examples=[Example("q1", {"A": 1, "B": 0})],
                    source_format="test", metadata={})
    with pytest.raises(ValueError):
        primary_model(data)


def test_merge_pairs_two_files_by_example_id():
    a = single("gpt", {"q1": 1, "q2": 0})
    b = single("claude", {"q1": 0, "q2": 1})
    merged = merge_two(a, b, "gpt", "claude")
    assert merged.models == ["gpt", "claude"]
    assert merged.n_examples == 2
    assert merged.examples[0].scores == {"gpt": 1.0, "claude": 0.0}


def test_merge_keeps_only_shared_example_ids():
    a = single("gpt", {"q1": 1, "q2": 0, "q3": 1})
    b = single("claude", {"q1": 0, "q2": 1})  # no q3
    merged = merge_two(a, b, "gpt", "claude")
    assert merged.n_examples == 2


def test_merge_raises_when_no_shared_ids():
    a = single("gpt", {"q1": 1})
    b = single("claude", {"q2": 0})
    with pytest.raises(ValueError):
        merge_two(a, b, "gpt", "claude")


def test_merge_preserves_runs_from_both_files():
    a = single("gpt", {"q1": 1}, runs={"q1": [1, 0, 1]})
    b = single("claude", {"q1": 0}, runs={"q1": [0, 0, 1]})
    ex = merge_two(a, b, "gpt", "claude").examples[0]
    assert ex.runs == {"gpt": [1.0, 0.0, 1.0], "claude": [0.0, 0.0, 1.0]}


def test_merge_preserves_judges_from_both_files():
    a = single("gpt", {"q1": 1}, judges={"q1": {"j1": 1, "j2": 0}})
    b = single("claude", {"q1": 0}, judges={"q1": {"j1": 0, "j2": 0}})
    ex = merge_two(a, b, "gpt", "claude").examples[0]
    assert ex.judges == {"j1": {"gpt": 1.0, "claude": 0.0},
                         "j2": {"gpt": 0.0, "claude": 0.0}}


def test_merge_sums_skipped_rows_from_both_inputs():
    a = single("gpt", {"q1": 1, "q2": 0})
    b = single("claude", {"q1": 0, "q2": 1})
    a.metadata["skipped_rows"] = 2
    b.metadata["skipped_rows"] = 3
    merged = merge_two(a, b, "gpt", "claude")
    assert merged.metadata["skipped_rows"] == 5


def test_merge_counts_unmatched_examples_from_both_sides():
    a = single("gpt", {"q1": 1, "q2": 0, "q3": 1})       # q3 only in A
    b = single("claude", {"q1": 0, "q2": 1, "q4": 0})    # q4 only in B
    merged = merge_two(a, b, "gpt", "claude")
    assert merged.n_examples == 2
    assert merged.metadata["unmatched_examples"] == 2


def test_merge_counts_examples_dropped_for_missing_scores():
    a = single("gpt", {"q1": 1, "q2": 0})
    b = single("claude", {"q1": 0})
    b.examples.append(Example(id="q2", scores={}))  # q2 present but unscored
    merged = merge_two(a, b, "gpt", "claude")
    assert merged.n_examples == 1
    assert merged.metadata["unmatched_examples"] == 2


def test_merge_with_full_overlap_reports_zero_unmatched():
    a = single("gpt", {"q1": 1, "q2": 0})
    b = single("claude", {"q1": 0, "q2": 1})
    merged = merge_two(a, b, "gpt", "claude")
    assert merged.metadata["unmatched_examples"] == 0
    assert merged.metadata["skipped_rows"] == 0
