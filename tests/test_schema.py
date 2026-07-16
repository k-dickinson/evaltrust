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
