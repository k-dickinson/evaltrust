"""Tests for the Benchmark Health audit."""

from evaltrust.audit.benchmark_health import audit_benchmark_health
from evaltrust.core.schema import EvalData, Example, Status


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


# --- score_ceiling tests (issue #54) ---

def make_rubric_data(a_scores, b_scores):
    """Helper for rubric-scale (e.g. 0-5) data where scores aren't 0/1."""
    examples = [
        Example(id=str(i), scores={"A": float(a), "B": float(b)})
        for i, (a, b) in enumerate(zip(a_scores, b_scores))
    ]
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


def test_configured_ceiling_prevents_false_saturation_warning():
    # On a 0-5 rubric, top_mean=4.0 and observed max=4.2 → 95% of observed → WARN.
    # But relative to the true ceiling of 5.0, that is only 80% → should PASS.
    a = [4.2] + [4.0] * 49  # observed max 4.2, mean 4.004
    b = [4.0] * 50           # mean 4.0
    data = make_rubric_data(a, b)

    sat_no_ceiling = by_check(audit_benchmark_health(data), "saturation")
    assert sat_no_ceiling.status is Status.WARN, (
        "Without a ceiling the observed-max heuristic should fire here"
    )

    sat_with_ceiling = by_check(
        audit_benchmark_health(data, score_ceiling=5.0), "saturation"
    )
    assert sat_with_ceiling.status is Status.PASS, (
        "With score_ceiling=5.0, 4.0/5.0=80% is below the 95% threshold"
    )


def test_configured_ceiling_recorded_in_details():
    a = [4.0] * 50
    b = [3.5] * 50
    data = make_rubric_data(a, b)

    sat = by_check(audit_benchmark_health(data, score_ceiling=5.0), "saturation")
    assert sat.details["ceiling"] == 5.0
    assert sat.details["ceiling_source"] == "configured"
    assert sat.details["observed_max"] == 4.0


def test_no_ceiling_uses_observed_max_and_labels_it():
    a = [0.9, 1.0] * 25
    b = [0.8, 0.9] * 25
    data = make_rubric_data(a, b)

    sat = by_check(audit_benchmark_health(data), "saturation")
    assert sat.details["ceiling_source"] == "observed"
    assert sat.details["ceiling"] == sat.details["observed_max"]
