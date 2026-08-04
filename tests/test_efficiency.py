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


# ---------------------------------------------------------------------------
# Bug fixes from CodeRabbit review
# ---------------------------------------------------------------------------

# --- Bug 1 (Critical): NaN quality_delta must not produce findings ----------

def test_nan_quality_delta_skips_all_findings():
    """When quality scores are missing for a model, quality_delta is NaN.
    audit_efficiency must return [] rather than emitting 'nan pt quality drop'
    or storing a non-JSON-serialisable NaN in details['quality_delta'].
    """
    # Build a quality dataset where model B has no scores at all.
    examples = [Example(id=str(i), scores={"A": 0.8}) for i in range(10)]
    quality = EvalData(models=["A", "B"], examples=examples,
                       source_format="test", metadata={})
    tokens = two_model_cost_data([100] * 10, [300] * 10)
    latency = two_model_cost_data([200] * 10, [600] * 10)

    findings = audit_efficiency(quality, "A", "B",
                                token_count_data=tokens, latency_data=latency)
    assert findings == [], (
        "NaN quality_delta must cause audit_efficiency to return [] "
        "rather than emitting malformed findings."
    )


def test_nan_quality_delta_details_not_in_output():
    """Ensure NaN never reaches details dict (not JSON-serialisable)."""
    import json
    import math
    examples = [Example(id=str(i), scores={"A": 0.8}) for i in range(10)]
    quality = EvalData(models=["A", "B"], examples=examples,
                       source_format="test", metadata={})
    tokens = two_model_cost_data([100] * 10, [300] * 10)
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    # No finding at all is correct; but if one slipped through, it must be
    # JSON-serialisable (no NaN).
    for f in findings:
        d = f.to_dict()
        json.dumps(d)  # must not raise
        assert not math.isnan(f.details.get("quality_delta", 0)), (
            "quality_delta must not be NaN in Finding.details"
        )


# --- Bug 2 (Minor): _quality_delta_str sign correctness --------------------

def test_quality_delta_str_drop_is_negative():
    """A quality drop must be shown with a '-' sign, not '+'."""
    from evaltrust.audit.efficiency import _quality_delta_str
    result = _quality_delta_str(-0.1)
    assert result.startswith("-"), (
        f"Expected '-2.0 pt quality drop', got '{result}'"
    )
    assert "drop" in result
    assert "+" not in result


def test_quality_delta_str_gain_is_positive():
    """A quality gain must be shown with a '+' sign."""
    from evaltrust.audit.efficiency import _quality_delta_str
    result = _quality_delta_str(0.1)
    assert result.startswith("+"), (
        f"Expected '+10.0 pt quality gain', got '{result}'"
    )
    assert "gain" in result


def test_quality_delta_str_zero():
    from evaltrust.audit.efficiency import _quality_delta_str
    assert _quality_delta_str(0.0) == "no quality change"


def test_quality_delta_str_no_unicode_minus():
    """Docstring promises ASCII hyphen for the minus sign, not Unicode U+2212."""
    from evaltrust.audit.efficiency import _quality_delta_str
    result = _quality_delta_str(-0.05)
    assert "\u2212" not in result, "Must use ASCII '-', not Unicode minus sign U+2212"


def test_token_how_to_fix_shows_signed_quality_delta():
    """The how_to_fix text in a token finding must show a signed delta,
    e.g. '+10.0 pt quality gain', never '+10.0 pt quality drop'."""
    quality = make_data([0.9] * 10, [0.7] * 10)  # B is worse: delta = -0.2
    tokens = two_model_cost_data([100] * 10, [300] * 10)
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    (tf,) = by_check(findings, "efficiency_tokens")
    # Must say 'drop', not 'gain', and must not start with '+' for a drop
    assert "drop" in tf.how_to_fix.lower()
    assert "+20.0 pt quality drop" not in tf.how_to_fix  # the old broken output


# --- Bug 3 (Critical): token cheaper-branch ratio orientation ---------------

def test_cheaper_token_ratio_reads_greater_than_one():
    """When B is cheaper, the displayed ratio must be > 1x (e.g. '3x fewer'),
    not 0.33x, which would read as a slowdown/increase."""
    quality = make_data([0.8] * 10, [0.8] * 10)
    # A=300, B=100 → B uses 1/3 the tokens → should say '3x fewer', not '0.33x'
    tokens = two_model_cost_data([300] * 10, [100] * 10)
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    (tf,) = by_check(findings, "efficiency_tokens")
    title = tf.title
    # The ratio in the title must be >= 1x (e.g. '3x fewer')
    # Extract the ratio number from the title
    import re
    match = re.search(r"([\d.]+)x", title)
    assert match, f"No ratio found in title: '{title}'"
    ratio_val = float(match.group(1))
    assert ratio_val >= 1.0, (
        f"Cheaper-branch ratio must be >= 1 (e.g. '3x fewer'), "
        f"got {ratio_val}x in: '{title}'"
    )


def test_cheaper_token_title_says_fewer_not_uses():
    """When B is cheaper the title should read 'fewer tokens', not 'uses Nx the tokens'
    with N < 1, which is confusing."""
    quality = make_data([0.8] * 10, [0.8] * 10)
    tokens = two_model_cost_data([300] * 10, [100] * 10)
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    (tf,) = by_check(findings, "efficiency_tokens")
    assert "fewer" in tf.title.lower() or float(
        __import__("re").search(r"([\d.]+)x", tf.title).group(1)
    ) >= 1.0


# --- Bug 4 (Major): latency faster-branch ratio orientation -----------------

def test_faster_latency_ratio_reads_greater_than_one():
    """When B is faster, the displayed ratio must be > 1x (e.g. '3x faster'),
    not 0.33x, which would read as a slowdown."""
    quality = make_data([0.8] * 10, [0.8] * 10)
    # A=600, B=200 → B is 3x faster → should say '3x faster', not '0.33x faster'
    latency = two_model_cost_data([600] * 10, [200] * 10)
    findings = audit_efficiency(quality, "A", "B", latency_data=latency)
    (lf,) = by_check(findings, "efficiency_latency")
    title = lf.title
    import re
    match = re.search(r"([\d.]+)x", title)
    assert match, f"No ratio found in title: '{title}'"
    ratio_val = float(match.group(1))
    assert ratio_val >= 1.0, (
        f"Faster-branch ratio must be >= 1 (e.g. '3x faster'), "
        f"got {ratio_val}x in: '{title}'"
    )


def test_faster_latency_how_to_fix_ratio_greater_than_one():
    """The how_to_fix text for the faster-branch must also use a ratio > 1."""
    quality = make_data([0.7] * 10, [0.9] * 10)  # B better and faster
    latency = two_model_cost_data([600] * 10, [200] * 10)
    findings = audit_efficiency(quality, "A", "B", latency_data=latency)
    (lf,) = by_check(findings, "efficiency_latency")
    import re
    match = re.search(r"([\d.]+)x", lf.how_to_fix)
    assert match, f"No ratio found in how_to_fix: '{lf.how_to_fix}'"
    ratio_val = float(match.group(1))
    assert ratio_val >= 1.0, (
        f"Faster-branch how_to_fix ratio must be >= 1, got {ratio_val}x"
    )


def test_slower_latency_ratio_reads_greater_than_one():
    """When B is slower, the displayed ratio must also be > 1x."""
    quality = make_data([0.8] * 10, [0.8] * 10)
    # A=200, B=600 → B is 3x slower
    latency = two_model_cost_data([200] * 10, [600] * 10)
    findings = audit_efficiency(quality, "A", "B", latency_data=latency)
    (lf,) = by_check(findings, "efficiency_latency")
    import re
    match = re.search(r"([\d.]+)x", lf.title)
    assert match, f"No ratio found in title: '{lf.title}'"
    ratio_val = float(match.group(1))
    assert ratio_val >= 1.0, (
        f"Slower-branch ratio must be >= 1, got {ratio_val}x"
    )


# ---------------------------------------------------------------------------
# Bug fix: latency_unit is carried through to how_detected (Image 1)
# ---------------------------------------------------------------------------

def test_latency_unit_default_is_ms():
    """Default latency_unit is 'ms', so how_detected says ms/example."""
    quality = make_data([0.8] * 10, [0.9] * 10)
    latency = two_model_cost_data([200] * 10, [600] * 10)
    findings = audit_efficiency(quality, "A", "B", latency_data=latency)
    (lf,) = by_check(findings, "efficiency_latency")
    assert "ms/example" in lf.how_detected


def test_latency_unit_seconds_appears_in_how_detected():
    """When latency_unit='s', how_detected must say 's/example', not 'ms/example'."""
    quality = make_data([0.8] * 10, [0.9] * 10)
    latency = two_model_cost_data([0.2] * 10, [0.6] * 10)
    findings = audit_efficiency(quality, "A", "B",
                                latency_data=latency, latency_unit="s")
    (lf,) = by_check(findings, "efficiency_latency")
    assert "s/example" in lf.how_detected
    assert "ms/example" not in lf.how_detected


def test_latency_unit_microseconds():
    """latency_unit='us' flows through correctly."""
    quality = make_data([0.8] * 10, [0.9] * 10)
    latency = two_model_cost_data([200000] * 10, [600000] * 10)
    findings = audit_efficiency(quality, "A", "B",
                                latency_data=latency, latency_unit="us")
    (lf,) = by_check(findings, "efficiency_latency")
    assert "us/example" in lf.how_detected


def test_latency_unit_via_public_api():
    """latency_unit is accepted and threaded through the public audit() call."""
    import evaltrust
    quality = make_data([0.8] * 20, [0.9] * 20)
    latency = two_model_cost_data([0.2] * 20, [0.6] * 20)
    report = evaltrust.audit(quality, model_a="A", model_b="B",
                             latency_data=latency, latency_unit="s")
    lf = [f for f in report.findings if f.details.get("check") == "efficiency_latency"]
    assert len(lf) == 1
    assert "s/example" in lf[0].how_detected
    assert "ms/example" not in lf[0].how_detected


# ---------------------------------------------------------------------------
# Bug fix: equality branches for tied resource means (Image 2)
# ---------------------------------------------------------------------------

def test_equal_token_means_no_directional_claim():
    """When A and B use exactly the same tokens, the title must not say
    'fewer' or 'uses Nx the tokens' -- it should say 'same'."""
    quality = make_data([0.8] * 10, [0.9] * 10)
    tokens = two_model_cost_data([200] * 10, [200] * 10)  # identical means
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    (tf,) = by_check(findings, "efficiency_tokens")
    assert "same" in tf.title.lower(), (
        f"Expected 'same' in title for equal token means, got: '{tf.title}'"
    )
    assert "fewer" not in tf.title.lower()
    # Recommendation should be based on quality only
    assert "quality" in tf.how_to_fix.lower()


def test_equal_token_means_equal_quality_says_either():
    """Both equal tokens AND equal quality -> 'either model is a valid choice'."""
    quality = make_data([0.8] * 10, [0.8] * 10)
    tokens = two_model_cost_data([200] * 10, [200] * 10)
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    (tf,) = by_check(findings, "efficiency_tokens")
    assert "either" in tf.how_to_fix.lower() or "valid" in tf.how_to_fix.lower()


def test_equal_latency_means_no_directional_claim():
    """When A and B have the same latency, the title must not call either slower."""
    quality = make_data([0.8] * 10, [0.9] * 10)
    latency = two_model_cost_data([300] * 10, [300] * 10)
    findings = audit_efficiency(quality, "A", "B", latency_data=latency)
    (lf,) = by_check(findings, "efficiency_latency")
    assert "same" in lf.title.lower(), (
        f"Expected 'same' in title for equal latency means, got: '{lf.title}'"
    )
    assert "slower" not in lf.title.lower()
    assert "faster" not in lf.title.lower()


def test_equal_latency_equal_quality_says_either():
    """Both equal latency AND equal quality -> 'either model is a valid choice'."""
    quality = make_data([0.8] * 10, [0.8] * 10)
    latency = two_model_cost_data([300] * 10, [300] * 10)
    findings = audit_efficiency(quality, "A", "B", latency_data=latency)
    (lf,) = by_check(findings, "efficiency_latency")
    assert "either" in lf.how_to_fix.lower() or "valid" in lf.how_to_fix.lower()


def test_equal_latency_b_better_quality_recommends_b():
    """Equal latency but B is better on quality -> prefer B."""
    quality = make_data([0.7] * 10, [0.9] * 10)
    latency = two_model_cost_data([300] * 10, [300] * 10)
    findings = audit_efficiency(quality, "A", "B", latency_data=latency)
    (lf,) = by_check(findings, "efficiency_latency")
    assert "B" in lf.how_to_fix or "prefer" in lf.how_to_fix.lower()


def test_equal_token_b_worse_quality_recommends_a():
    """Equal tokens but B is worse on quality -> prefer A."""
    quality = make_data([0.9] * 10, [0.7] * 10)
    tokens = two_model_cost_data([200] * 10, [200] * 10)
    findings = audit_efficiency(quality, "A", "B", token_count_data=tokens)
    (tf,) = by_check(findings, "efficiency_tokens")
    assert "A" in tf.how_to_fix
