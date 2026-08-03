"""Tests for the Efficiency audit (issue #165).

The efficiency findings are advisory-only: they appear when token_count or
latency data is present, but they never lower the confidence verdict.  Default
output for quality-only files must be unchanged (no new findings, no verdict
change).
"""

import pytest

from evaltrust.audit.efficiency import audit_efficiency, PILLAR
from evaltrust.audit.runner import run_audit
from evaltrust.core.schema import EvalData, Example, Status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data(a_scores, b_scores, models=("A", "B")):
    """Build a two-model EvalData from parallel score lists."""
    examples = [
        Example(id=str(i), scores={models[0]: float(a), models[1]: float(b)})
        for i, (a, b) in enumerate(zip(a_scores, b_scores))
    ]
    return EvalData(models=list(models), examples=examples,
                    source_format="test", metadata={})


def single_model_data(model, scores):
    """Build a single-model EvalData (for cost/latency parallel datasets)."""
    examples = [
        Example(id=str(i), scores={model: float(s)})
        for i, s in enumerate(scores)
    ]
    return EvalData(models=[model], examples=examples,
                    source_format="test", metadata={})


def two_model_cost_data(a_costs, b_costs, models=("A", "B")):
    """Build a cost/latency EvalData with per-example values for two models."""
    examples = [
        Example(id=str(i), scores={models[0]: float(a), models[1]: float(b)})
        for i, (a, b) in enumerate(zip(a_costs, b_costs))
    ]
    return EvalData(models=list(models), examples=examples,
                    source_format="test", metadata={})


def by_check(findings, check):
    hits = [f for f in findings if f.details.get("check") == check]
    return hits


# ---------------------------------------------------------------------------
# No data → no findings
# ---------------------------------------------------------------------------

def test_no_efficiency_data_returns_empty():
    quality = make_data([0] * 50, [1] * 50)
    findings = audit_efficiency(quality, "A", "B",
                                token_count_data=None, latency_data=None)
    assert findings == []


def test_no_efficiency_data_does_not_appear_in_run_audit():
    quality = make_data([0] * 50, [1] * 50)
    report = run_audit(quality, model_a="A", model_b="B")
    efficiency_findings = [f for f in report.findings if f.pillar == PILLAR]
    assert efficiency_findings == []


# ---------------------------------------------------------------------------
# Token count findings
# ---------------------------------------------------------------------------

def test_token_count_finding_is_advisory_pass():
    quality = make_data([0.8] * 20, [0.9] * 20)
    tokens = two_model_cost_data([100] * 20, [320] * 20)
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    tf = by_check(findings, "efficiency_tokens")
    assert len(tf) == 1
    assert tf[0].status is Status.PASS
    assert tf[0].pillar == PILLAR


def test_token_finding_details_contain_ratio():
    quality = make_data([0.8] * 10, [0.9] * 10)
    tokens = two_model_cost_data([100] * 10, [300] * 10)
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    (tf,) = by_check(findings, "efficiency_tokens")
    details = tf.details
    assert details["mean_tokens_a"] == pytest.approx(100.0)
    assert details["mean_tokens_b"] == pytest.approx(300.0)
    assert details["token_ratio_b_over_a"] == pytest.approx(3.0)


def test_token_finding_shows_quality_delta():
    quality = make_data([0.7] * 10, [0.8] * 10)
    tokens = two_model_cost_data([100] * 10, [300] * 10)
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    (tf,) = by_check(findings, "efficiency_tokens")
    assert tf.details["quality_delta"] == pytest.approx(0.1)
    # The "3x more tokens for a quality gain" tradeoff should be explicit.
    assert "3" in tf.how_detected
    assert "quality" in tf.how_detected.lower()


def test_token_finding_cheaper_and_better_recommends_b():
    quality = make_data([0.7] * 10, [0.9] * 10)
    tokens = two_model_cost_data([300] * 10, [100] * 10)  # B is cheaper
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    (tf,) = by_check(findings, "efficiency_tokens")
    # B is better on quality AND cheaper → prefer B
    assert "cheaper" in tf.how_to_fix.lower() or "prefer" in tf.how_to_fix.lower()


def test_token_finding_worse_and_more_expensive_recommends_a():
    quality = make_data([0.9] * 10, [0.7] * 10)
    tokens = two_model_cost_data([100] * 10, [300] * 10)  # B is expensive and worse
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    (tf,) = by_check(findings, "efficiency_tokens")
    # B is worse AND pricier → prefer A
    assert "A" in tf.how_to_fix


def test_token_finding_equivalent_quality_recommends_cheaper():
    quality = make_data([0.8] * 10, [0.8] * 10)  # same quality
    tokens = two_model_cost_data([100] * 10, [400] * 10)
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    (tf,) = by_check(findings, "efficiency_tokens")
    assert "cheaper" in tf.how_to_fix.lower() or "prefer" in tf.how_to_fix.lower()


# ---------------------------------------------------------------------------
# Latency findings
# ---------------------------------------------------------------------------

def test_latency_finding_is_advisory_pass():
    quality = make_data([0.8] * 20, [0.9] * 20)
    latency = two_model_cost_data([200] * 20, [800] * 20)
    findings = audit_efficiency(quality, "A", "B", latency_data=latency)
    lf = by_check(findings, "efficiency_latency")
    assert len(lf) == 1
    assert lf[0].status is Status.PASS
    assert lf[0].pillar == PILLAR


def test_latency_finding_details():
    quality = make_data([0.8] * 10, [0.9] * 10)
    latency = two_model_cost_data([200] * 10, [600] * 10)
    findings = audit_efficiency(quality, "A", "B", latency_data=latency)
    (lf,) = by_check(findings, "efficiency_latency")
    assert lf.details["mean_latency_a"] == pytest.approx(200.0)
    assert lf.details["mean_latency_b"] == pytest.approx(600.0)
    assert lf.details["latency_ratio_b_over_a"] == pytest.approx(3.0)


def test_latency_finding_faster_and_better_recommends_b():
    quality = make_data([0.7] * 10, [0.9] * 10)
    latency = two_model_cost_data([600] * 10, [200] * 10)  # B is faster
    findings = audit_efficiency(quality, "A", "B", latency_data=latency)
    (lf,) = by_check(findings, "efficiency_latency")
    assert "prefer" in lf.how_to_fix.lower() or "faster" in lf.how_to_fix.lower()


# ---------------------------------------------------------------------------
# Both token and latency together
# ---------------------------------------------------------------------------

def test_both_token_and_latency_findings():
    quality = make_data([0.8] * 20, [0.9] * 20)
    tokens = two_model_cost_data([100] * 20, [300] * 20)
    latency = two_model_cost_data([200] * 20, [600] * 20)
    findings = audit_efficiency(quality, "A", "B",
                                token_count_data=tokens, latency_data=latency)
    assert len(by_check(findings, "efficiency_tokens")) == 1
    assert len(by_check(findings, "efficiency_latency")) == 1
    assert len(findings) == 2


# ---------------------------------------------------------------------------
# Verdict is unchanged by efficiency findings
# ---------------------------------------------------------------------------

def test_efficiency_findings_do_not_lower_verdict():
    """Efficiency findings are PASS and must never be the cause of a lower verdict.

    We verify that every Efficiency finding is Status.PASS (advisory), and that
    removing efficiency findings from the verdict computation does not change the
    reported verdict level — i.e. no Efficiency finding is a driver.
    """
    import numpy as np
    rng = np.random.default_rng(42)
    # Realistic continuous scores: A ~ 0.5, B ~ 0.7, enough n to be conclusive.
    a = list(rng.uniform(0.3, 0.7, 200))
    b = list(rng.uniform(0.5, 0.9, 200))
    quality = make_data(a, b)
    tokens = two_model_cost_data([100] * 200, [300] * 200)
    report = run_audit(quality, model_a="A", model_b="B",
                       token_count_data=tokens)

    # All efficiency findings must be advisory (PASS).
    for f in report.findings:
        if f.pillar == PILLAR:
            assert f.status is Status.PASS

    # No Efficiency finding should be a verdict driver.
    for f in report.verdict.drivers:
        assert f.pillar != PILLAR


def test_efficiency_findings_do_not_alter_quality_only_report():
    """Adding efficiency data must not change any existing finding."""
    quality = make_data([0] * 50, [1] * 50)
    report_no_eff = run_audit(quality, model_a="A", model_b="B")
    tokens = two_model_cost_data([100] * 50, [300] * 50)
    report_with_eff = run_audit(quality, model_a="A", model_b="B",
                                token_count_data=tokens)

    # All pre-existing findings (non-Efficiency pillar) must be unchanged.
    old_checks = {(f.pillar, f.details.get("check")): f.status
                  for f in report_no_eff.findings}
    for f in report_with_eff.findings:
        if f.pillar == PILLAR:
            continue
        key = (f.pillar, f.details.get("check"))
        assert key in old_checks, f"Unexpected new non-Efficiency finding: {key}"
        assert f.status is old_checks[key]


# ---------------------------------------------------------------------------
# Missing model data → silent skip, no crash
# ---------------------------------------------------------------------------

def test_token_data_missing_one_model_returns_no_finding():
    """If cost data only has one of the two models, skip silently."""
    quality = make_data([0.8] * 10, [0.9] * 10)
    # Only model A in the token dataset
    tokens = single_model_data("A", [100] * 10)
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    # NaN for B → should return no token finding
    assert by_check(findings, "efficiency_tokens") == []


def test_latency_data_missing_one_model_returns_no_finding():
    quality = make_data([0.8] * 10, [0.9] * 10)
    latency = single_model_data("B", [500] * 10)
    findings = audit_efficiency(quality, "A", "B", latency_data=latency)
    assert by_check(findings, "efficiency_latency") == []


# ---------------------------------------------------------------------------
# to_dict serialisability
# ---------------------------------------------------------------------------

def test_token_finding_is_json_serialisable():
    import json
    quality = make_data([0.8] * 10, [0.9] * 10)
    tokens = two_model_cost_data([100] * 10, [300] * 10)
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    assert findings
    d = findings[0].to_dict()
    json.dumps(d)  # must not raise


def test_latency_finding_is_json_serialisable():
    import json
    quality = make_data([0.8] * 10, [0.9] * 10)
    latency = two_model_cost_data([200] * 10, [600] * 10)
    findings = audit_efficiency(quality, "A", "B", latency_data=latency)
    assert findings
    d = findings[0].to_dict()
    json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# API-level smoke test (evaltrust.audit with efficiency kwargs)
# ---------------------------------------------------------------------------

def test_public_api_accepts_efficiency_kwargs():
    import evaltrust
    quality = make_data([0] * 50, [1] * 50)
    tokens = two_model_cost_data([100] * 50, [300] * 50)
    latency = two_model_cost_data([200] * 50, [600] * 50)
    report = evaltrust.audit(quality, model_a="A", model_b="B",
                             token_count_data=tokens, latency_data=latency)
    efficiency = [f for f in report.findings if f.pillar == PILLAR]
    assert len(efficiency) == 2  # one token, one latency


def test_public_api_quality_only_unchanged():
    """audit() with no efficiency kwargs must produce no Efficiency findings."""
    import evaltrust
    quality = make_data([0] * 50, [1] * 50)
    report = evaltrust.audit(quality, model_a="A", model_b="B")
    efficiency = [f for f in report.findings if f.pillar == PILLAR]
    assert efficiency == []
