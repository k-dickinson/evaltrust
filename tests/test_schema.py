"""Tests for the canonical data model that every adapter maps into."""

import json

import numpy as np
import pytest

from evaltrust.core.schema import EvalData, Example, Finding, Status


def _data(**kw):
    examples = kw.pop("examples")
    return EvalData(
        models=kw.pop("models", ["A", "B"]),
        examples=examples,
        source_format=kw.pop("source_format", "test"),
        metadata=kw.pop("metadata", {}),
    )


def test_example_defaults_have_no_runs_or_judges():
    ex = Example(id="q1", scores={"A": 1.0, "B": 0.0})
    assert ex.runs is None
    assert ex.judges is None


def test_n_examples_counts_examples():
    d = _data(examples=[Example("q1", {"A": 1, "B": 0}),
                        Example("q2", {"A": 0, "B": 1})])
    assert d.n_examples == 2


def test_paired_scores_aligns_two_models_by_example():
    d = _data(examples=[Example("q1", {"A": 1.0, "B": 0.0}),
                        Example("q2", {"A": 0.0, "B": 1.0})])
    a, b = d.paired_scores("A", "B")
    assert list(a) == [1.0, 0.0]
    assert list(b) == [0.0, 1.0]


def test_paired_scores_skips_examples_missing_a_model():
    d = _data(examples=[Example("q1", {"A": 1.0, "B": 0.0}),
                        Example("q2", {"A": 0.0}),          # no B
                        Example("q3", {"A": 1.0, "B": 1.0})])
    a, b = d.paired_scores("A", "B")
    assert len(a) == 2 and len(b) == 2


def test_differences_are_b_minus_a():
    d = _data(examples=[Example("q1", {"A": 1.0, "B": 0.0}),
                        Example("q2", {"A": 0.0, "B": 1.0})])
    diffs = d.differences("A", "B")
    assert isinstance(diffs, np.ndarray)
    assert list(diffs) == [-1.0, 1.0]


def test_has_runs_true_only_when_any_example_has_runs():
    without = _data(examples=[Example("q1", {"A": 1, "B": 0})])
    with_runs = _data(examples=[
        Example("q1", {"A": 1, "B": 0}, runs={"A": [1, 1], "B": [0, 0]})])
    assert without.has_runs is False
    assert with_runs.has_runs is True


def test_paired_run_differences_yields_per_example_run_diffs():
    d = _data(examples=[
        Example("q1", {"A": 0.5, "B": 0.6}, runs={"A": [0, 1], "B": [1, 1]}),
        Example("q2", {"A": 0, "B": 1}),                       # no runs -> excluded
        Example("q3", {"A": 1, "B": 0}, runs={"A": [1, 1, 1], "B": [0, 0]}),
        Example("q4", {"A": 1, "B": 0}, runs={"A": [1, 1]}),   # runs for one model -> excluded
    ])
    groups = d.paired_run_differences("A", "B")
    assert len(groups) == 2                     # q1 and q3 only
    assert list(groups[0]) == [1.0, 0.0]        # [1,1] - [0,1]
    assert list(groups[1]) == [-1.0, -1.0]      # aligned to len 2: [0,0] - [1,1]


def test_paired_run_differences_single_run_yields_length_one_array():
    # Graceful degradation: one run per model is a valid (size-1) cluster.
    d = _data(examples=[Example("q1", {"A": 1, "B": 0}, runs={"A": [1], "B": [0]})])
    groups = d.paired_run_differences("A", "B")
    assert len(groups) == 1
    assert list(groups[0]) == [-1.0]            # [0] - [1]


def test_paired_run_differences_empty_without_runs():
    d = _data(examples=[Example(str(i), {"A": 0, "B": 1}) for i in range(5)])
    assert d.paired_run_differences("A", "B") == []


def test_has_judges_true_only_when_any_example_has_judges():
    without = _data(examples=[Example("q1", {"A": 1, "B": 0})])
    with_judges = _data(examples=[
        Example("q1", {"A": 1, "B": 0}, judges={"gpt": {"A": 1, "B": 0}})])
    assert without.has_judges is False
    assert with_judges.has_judges is True


def test_finding_carries_the_golden_rule_fields():
    f = Finding(
        pillar="Statistical Validity",
        title="Improvement is significant",
        status=Status.PASS,
        why="w", how_detected="h", how_to_fix="f", details={"p": 0.01},
    )
    assert f.status is Status.PASS
    assert f.details["p"] == 0.01


def test_status_has_the_four_levels():
    assert {s.name for s in Status} == {"PASS", "WARN", "FAIL", "SKIP"}


def test_status_str_returns_value():
    assert str(Status.PASS) == "pass"
    assert str(Status.WARN) == "warn"
    assert str(Status.FAIL) == "fail"
    assert str(Status.SKIP) == "skip"


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_to_dict_serializes_non_finite_floats_as_null(bad):
    # An infinite Cohen's d (zero-variance gap) or an infinite SNR reaches the
    # details dict as a bare float. json.dumps would emit `Infinity`/`NaN`, which
    # is not valid JSON, so to_dict() must render them as null instead.
    f = Finding(
        pillar="Statistical Validity", title="t", status=Status.PASS,
        why="w", how_detected="h", how_to_fix="x",
        details={"cohens_d": bad, "nested": [bad], "deep": {"snr": bad}},
    )
    d = f.to_dict()["details"]
    assert d["cohens_d"] is None
    assert d["nested"] == [None]
    assert d["deep"]["snr"] is None
    # The whole payload must round-trip through a strict JSON parser.
    text = json.dumps(f.to_dict(), allow_nan=False)
    json.loads(text)


# ---------------------------------------------------------------------------
# cluster-aware schema helpers
# ---------------------------------------------------------------------------
from evaltrust.core.schema import EvalData, Example


def _make_clustered_data():
    return EvalData(
        models=["A", "B"],
        examples=[
            Example(id="1", scores={"A": 0.0, "B": 1.0}, group_id="g1"),
            Example(id="2", scores={"A": 1.0, "B": 0.0}, group_id="g1"),
            Example(id="3", scores={"A": 0.0, "B": 1.0}, group_id="g2"),
            Example(id="4", scores={"A": 0.5, "B": 0.5}, group_id="g2"),
        ],
        source_format="test",
    )


def test_has_clusters_true_when_group_id_present():
    data = _make_clustered_data()
    assert data.has_clusters is True


def test_has_clusters_false_when_no_group_id():
    data = EvalData(
        models=["A", "B"],
        examples=[
            Example(id="1", scores={"A": 0.0, "B": 1.0}),
            Example(id="2", scores={"A": 1.0, "B": 0.0}),
        ],
        source_format="test",
    )
    assert data.has_clusters is False


def test_cluster_groups_returns_correct_number_of_clusters():
    data = _make_clustered_data()
    groups = data.cluster_groups("A", "B")
    assert len(groups) == 2


def test_cluster_groups_values_match_differences():
    import numpy as np
    data = _make_clustered_data()
    groups = data.cluster_groups("A", "B")
    all_diffs = np.concatenate(groups)
    expected = data.differences("A", "B")
    assert np.allclose(sorted(all_diffs), sorted(expected))


def test_cluster_groups_singleton_without_group_id():
    """Examples without group_id each form their own cluster."""
    import numpy as np
    data = EvalData(
        models=["A", "B"],
        examples=[
            Example(id="x", scores={"A": 0.0, "B": 1.0}),
            Example(id="y", scores={"A": 1.0, "B": 0.5}),
        ],
        source_format="test",
    )
    groups = data.cluster_groups("A", "B")
    assert len(groups) == 2
    assert all(len(g) == 1 for g in groups)


def test_example_group_id_defaults_to_none():
    ex = Example(id="z", scores={"A": 1.0})
    assert ex.group_id is None
