"""Tests for the Statistical Validity audit — EvalLab's flagship check."""

import numpy as np

from evallab.audit.statistical import audit_statistical_validity
from evallab.core.schema import EvalData, Example, Status


def make_data(a_scores, b_scores):
    examples = [
        Example(id=str(i), scores={"A": float(a), "B": float(b)})
        for i, (a, b) in enumerate(zip(a_scores, b_scores))
    ]
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


def by_check(findings, check):
    (f,) = [f for f in findings if f.details.get("check") == check]
    return f


def test_produces_four_checks():
    data = make_data([0] * 50, [1] * 50)
    findings = audit_statistical_validity(data, "A", "B")
    checks = {f.details["check"] for f in findings}
    assert checks == {"significance", "confidence_interval", "effect_size", "power"}


def test_clean_win_passes_every_check():
    # B right 90% of the time, A never right, over 200 paired examples.
    a = [0] * 200
    b = [1] * 180 + [0] * 20
    findings = audit_statistical_validity(make_data(a, b), "A", "B", seed=0)
    assert by_check(findings, "significance").status is Status.PASS
    assert by_check(findings, "confidence_interval").status is Status.PASS
    assert by_check(findings, "effect_size").status is Status.PASS
    assert by_check(findings, "power").status is Status.PASS


def test_no_real_difference_is_not_significant():
    # Perfectly symmetric differences: mean zero, cannot be significant.
    a = [0, 1] * 60
    b = [1, 0] * 60
    findings = audit_statistical_validity(make_data(a, b), "A", "B", seed=0)
    assert by_check(findings, "significance").status is Status.FAIL
    assert by_check(findings, "confidence_interval").status is Status.WARN


def test_confidence_interval_that_excludes_zero_passes():
    a = [0] * 200
    b = [1] * 180 + [0] * 20
    ci = by_check(audit_statistical_validity(make_data(a, b), "A", "B"),
                  "confidence_interval")
    assert ci.details["ci_low"] > 0
    assert ci.status is Status.PASS


def test_small_effect_is_flagged():
    rng = np.random.default_rng(0)
    a = np.zeros(400)
    # Per-example gap of ~0.1 swamped by unit noise: real but tiny effect.
    b = rng.normal(0.1, 1.0, size=400)
    effect = by_check(audit_statistical_validity(make_data(a, b), "A", "B"),
                      "effect_size")
    assert effect.status is Status.WARN
    assert effect.details["magnitude"] in {"negligible", "small"}


def test_underpowered_small_sample_warns_and_recommends_n():
    # A real but modest effect measured on only a handful of examples.
    a = [0, 0, 0, 0, 0, 0]
    b = [1, 0, 1, 0, 1, 0]
    power = by_check(audit_statistical_validity(make_data(a, b), "A", "B"),
                     "power")
    assert power.status is Status.WARN
    assert power.details["required_n"] > power.details["n"]
    assert "example" in power.how_to_fix.lower()


def test_adequate_power_passes():
    a = [0] * 300
    b = [1] * 270 + [0] * 30
    power = by_check(audit_statistical_validity(make_data(a, b), "A", "B"),
                     "power")
    assert power.details["achieved_power"] >= 0.8
    assert power.status is Status.PASS


def test_every_finding_obeys_the_golden_rule():
    findings = audit_statistical_validity(make_data([0] * 40, [1] * 40), "A", "B")
    for f in findings:
        assert f.why.strip()
        assert f.how_detected.strip()
        assert f.how_to_fix.strip()
        assert f.pillar == "Statistical Validity"


def test_is_deterministic():
    data = make_data([0] * 100, [1] * 90 + [0] * 10)
    f1 = audit_statistical_validity(data, "A", "B", seed=1)
    f2 = audit_statistical_validity(data, "A", "B", seed=1)
    assert [f.details for f in f1] == [f.details for f in f2]
