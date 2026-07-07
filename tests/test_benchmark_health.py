"""Tests for the Benchmark Health audit."""

from evallab.audit.benchmark_health import audit_benchmark_health
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


def test_produces_saturation_and_discrimination_checks():
    findings = audit_benchmark_health(make_data([1, 0] * 20, [1, 1] * 20))
    assert {f.details["check"] for f in findings} == {"saturation", "discrimination"}


def test_saturated_benchmark_warns():
    a = [1] * 98 + [0] * 2   # 98%
    b = [1] * 99 + [0] * 1   # 99%
    sat = by_check(audit_benchmark_health(make_data(a, b)), "saturation")
    assert sat.status is Status.WARN


def test_healthy_benchmark_passes_saturation():
    a = [1] * 60 + [0] * 40  # 60%
    b = [1] * 70 + [0] * 30  # 70%
    sat = by_check(audit_benchmark_health(make_data(a, b)), "saturation")
    assert sat.status is Status.PASS


def test_no_variation_flags_discrimination():
    # Every example scored identically for everyone: benchmark has no resolution.
    data = make_data([1] * 50, [1] * 50)
    disc = by_check(audit_benchmark_health(data), "discrimination")
    assert disc.status is Status.WARN


def test_spread_scores_pass_discrimination():
    a = [1] * 60 + [0] * 40
    b = [1] * 70 + [0] * 30
    disc = by_check(audit_benchmark_health(make_data(a, b)), "discrimination")
    assert disc.status is Status.PASS


def test_findings_obey_golden_rule():
    findings = audit_benchmark_health(make_data([1] * 98 + [0] * 2, [1] * 100))
    for f in findings:
        assert f.why.strip() and f.how_detected.strip() and f.how_to_fix.strip()
        assert f.pillar == "Benchmark Health"
