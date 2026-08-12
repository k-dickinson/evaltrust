"""Tests for the shared ingestion primitives every adapter builds on.

The universal shape underneath every eval tool's output is a stream of
(example, model, score) records. Get that grouping exactly right once and every
adapter becomes a thin mapping onto it.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

from evaltrust.adapters.common import Record, coerce_score, records_to_evaldata


# ---------------------------------------------------------------------------
# coerce_score: normalise the many ways tools spell a pass/fail or number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (1, 1.0), (0, 0.0), (0.8, 0.8), (True, 1.0), (False, 0.0),
    ("0.8", 0.8), ("1", 1.0),
    ("pass", 1.0), ("PASS", 1.0), ("fail", 0.0), ("Fail", 0.0),
    ("true", 1.0), ("false", 0.0), ("yes", 1.0), ("no", 0.0),
    ("correct", 1.0), ("incorrect", 0.0),
    ("win", 1.0), ("loss", 0.0),
    ("accept", 1.0), ("reject", 0.0),
    ("positive", 1.0), ("negative", 0.0),
    ("good", 1.0), ("bad", 0.0),
    ("won", 1.0), ("lost", 0.0),
    ("85%", 0.85), ("100%", 1.0), ("0%", 0.0),
])
def test_coerce_score_normalises_common_spellings(raw, expected):
    assert coerce_score(raw) == pytest.approx(expected)


def test_coerce_score_rejects_nonsense():
    with pytest.raises(ValueError):
        coerce_score("banana")


def test_coerce_score_rejects_bare_percent():
    with pytest.raises(ValueError):
        coerce_score("%")


def test_coerce_score_rejects_none():
    with pytest.raises(ValueError):
        coerce_score(None)


@pytest.mark.parametrize("raw,reference", [
    ("3/5", Fraction(3, 5)),
    (" 3.5 / 7 ", Fraction("3.5") / Fraction(7)),
    ("4 out of 5", Fraction(4, 5)),
    (" 4\tOUT   OF\t5 ", Fraction(4, 5)),
    ("0/5", Fraction(0, 5)),
    ("6/5", Fraction(6, 5)),
])
def test_coerce_score_parses_fraction_scores_against_fraction_reference(raw, reference):
    assert coerce_score(raw) == pytest.approx(float(reference))


@pytest.mark.parametrize("raw", [
    "5/0", "-1/5", "1/-5", "three/5", "3/five",
    "-1 out of 5", "1 out of -5", "three out of 5", "3 out of five",
    "1e2/5", "3 of 5",
])
def test_coerce_score_rejects_unsafe_or_unratified_fraction_shapes(raw):
    with pytest.raises(ValueError) as exc_info:
        coerce_score(raw)
    assert str(exc_info.value) == f"Cannot interpret {raw!r} as a score"


def test_coerce_score_fraction_extremes_follow_float_semantics_without_crashing():
    huge_decimal = "9" * 400
    assert coerce_score(f"1/{huge_decimal}") == 0.0
    assert math.isinf(coerce_score(f"{huge_decimal}/1"))


# ---------------------------------------------------------------------------
# records_to_evaldata: group flat records into canonical examples
# ---------------------------------------------------------------------------

def test_groups_records_into_examples_and_models():
    records = [
        Record("q1", "A", 1.0), Record("q1", "B", 0.0),
        Record("q2", "A", 0.0), Record("q2", "B", 1.0),
    ]
    data = records_to_evaldata(records, "test")
    assert data.n_examples == 2
    assert set(data.models) == {"A", "B"}
    assert data.examples[0].scores == {"A": 1.0, "B": 0.0}


def test_model_order_follows_first_appearance():
    records = [Record("q1", "B", 1.0), Record("q1", "A", 0.0)]
    assert records_to_evaldata(records, "test").models == ["B", "A"]


def test_repeated_records_without_judges_become_runs():
    records = [
        Record("q1", "A", 1.0), Record("q1", "A", 0.0), Record("q1", "A", 1.0),
        Record("q1", "B", 0.0), Record("q1", "B", 0.0),
    ]
    data = records_to_evaldata(records, "test")
    ex = data.examples[0]
    assert ex.runs == {"A": [1.0, 0.0, 1.0], "B": [0.0, 0.0]}
    # Final score is the mean across runs.
    assert ex.scores["A"] == pytest.approx(2 / 3)
    assert data.has_runs is True


def test_judge_tagged_records_become_judges_and_average_score():
    records = [
        Record("q1", "A", 1.0, judge="gpt"),
        Record("q1", "A", 0.0, judge="claude"),
        Record("q1", "B", 1.0, judge="gpt"),
        Record("q1", "B", 1.0, judge="claude"),
    ]
    data = records_to_evaldata(records, "test")
    ex = data.examples[0]
    assert ex.judges == {"gpt": {"A": 1.0, "B": 1.0}, "claude": {"A": 0.0, "B": 1.0}}
    assert ex.scores["A"] == pytest.approx(0.5)  # mean over judges
    assert data.has_judges is True


def test_empty_records_raise():
    with pytest.raises(ValueError):
        records_to_evaldata([], "test")
