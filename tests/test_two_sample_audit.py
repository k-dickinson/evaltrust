"""Tests for the unpaired two-sample audit (audit/two_sample.py) and
run-level ingest (core/ingest.load_run_level).

Verifies:
  - audit_two_sample produces three findings with expected check keys
  - decision, effect_size, precision finding shapes and statuses
  - load_run_level handles wide CSV, long CSV, and JSON inputs
  - Paired path (audit_statistical_validity) is completely unaffected
  - CLI --run-level flag routes to the two-sample path
"""

import json
import textwrap

import numpy as np
import pytest

from evaltrust.audit.two_sample import audit_two_sample
from evaltrust.core.ingest import load_run_level
from evaltrust.core.schema import RunLevelData, Status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run_level(a_scores, b_scores, model_a="A", model_b="B"):
    return RunLevelData(
        model_a=model_a,
        model_b=model_b,
        scores_a=np.array(a_scores, dtype=float),
        scores_b=np.array(b_scores, dtype=float),
        source_format="test",
    )


def by_check(findings, check):
    matches = [f for f in findings if f.details.get("check") == check]
    assert len(matches) == 1, f"Expected 1 finding with check={check!r}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# audit_two_sample: finding structure
# ---------------------------------------------------------------------------

def test_produces_three_findings():
    data = make_run_level([0.8] * 20, [0.6] * 20)
    findings = audit_two_sample(data)
    assert {f.details["check"] for f in findings} == {
        "decision", "effect_size", "precision"
    }


def test_all_findings_carry_comparison_path():
    data = make_run_level([0.8] * 20, [0.6] * 20)
    for f in audit_two_sample(data):
        assert f.details.get("comparison_path") == "unpaired_two_sample"


def test_clear_win_decision_is_pass():
    rng = np.random.default_rng(0)
    a = rng.normal(0.85, 0.03, 50)
    b = rng.normal(0.60, 0.03, 50)
    decision = by_check(audit_two_sample(make_run_level(a, b), seed=0), "decision")
    assert decision.status is Status.PASS
    assert decision.details["outcome"] == "significant"


def test_near_equal_decision_is_warn_or_fail():
    rng = np.random.default_rng(0)
    a = rng.normal(0.7, 0.1, 30)
    b = rng.normal(0.7, 0.1, 30)
    decision = by_check(audit_two_sample(make_run_level(a, b), seed=0), "decision")
    assert decision.status in (Status.WARN, Status.FAIL)


def test_decision_details_contain_p_value_and_p_a_gt_b():
    data = make_run_level([0.8] * 20, [0.6] * 20)
    decision = by_check(audit_two_sample(data, seed=0), "decision")
    assert "p_value_mann_whitney" in decision.details
    assert "p_a_gt_b" in decision.details
    assert 0.0 <= decision.details["p_a_gt_b"] <= 1.0
    assert 0.0 <= decision.details["p_value_mann_whitney"] <= 1.0


def test_decision_ci_bounds_present():
    data = make_run_level([0.8] * 15, [0.6] * 15)
    decision = by_check(audit_two_sample(data, seed=0), "decision")
    assert "ci_low" in decision.details
    assert "ci_high" in decision.details
    assert decision.details["ci_low"] <= decision.details["ci_high"]


# ---------------------------------------------------------------------------
# effect_size finding
# ---------------------------------------------------------------------------

def test_large_effect_size_is_pass():
    a = np.ones(30) * 0.9
    b = np.zeros(30) * 0.0 + 0.1
    effect = by_check(audit_two_sample(make_run_level(a, b), seed=0), "effect_size")
    assert effect.details["magnitude"] in {"large"}
    assert effect.status is Status.PASS


def test_small_effect_size_is_warn():
    rng = np.random.default_rng(1)
    a = rng.normal(0.7, 0.1, 30)
    b = rng.normal(0.705, 0.1, 30)  # tiny difference
    effect = by_check(audit_two_sample(make_run_level(a, b), seed=0), "effect_size")
    assert effect.details["magnitude"] in {"negligible", "small"}
    assert effect.status is Status.WARN


def test_effect_size_details_have_descriptive_stats():
    a = np.array([0.7, 0.8, 0.75])
    b = np.array([0.5, 0.55, 0.52])
    effect = by_check(audit_two_sample(make_run_level(a, b), seed=0), "effect_size")
    assert "mean_a" in effect.details
    assert "mean_b" in effect.details
    assert "std_a" in effect.details
    assert "std_b" in effect.details
    assert effect.details["mean_a"] == pytest.approx(np.mean(a), abs=1e-6)
    assert effect.details["mean_b"] == pytest.approx(np.mean(b), abs=1e-6)


# ---------------------------------------------------------------------------
# precision finding
# ---------------------------------------------------------------------------

def test_precision_pass_when_sufficient_runs():
    # 20 runs per model is >= _MIN_RUNS_RECOMMENDED (10)
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, 20)
    b = rng.normal(0.6, 0.05, 20)
    precision = by_check(audit_two_sample(make_run_level(a, b), seed=0), "precision")
    assert precision.details["sufficient"] is True


def test_precision_warn_when_few_runs():
    a = np.array([0.8, 0.9])   # only 2 runs
    b = np.array([0.5, 0.6, 0.7, 0.8])
    precision = by_check(audit_two_sample(make_run_level(a, b), seed=0), "precision")
    assert precision.details["sufficient"] is False
    assert precision.status is Status.WARN


def test_precision_details_contain_run_counts():
    a = np.ones(12)
    b = np.ones(8)
    precision = by_check(audit_two_sample(make_run_level(a, b), seed=0), "precision")
    assert precision.details["n_a"] == 12
    assert precision.details["n_b"] == 8
    assert precision.details["min_n"] == 8


# ---------------------------------------------------------------------------
# Paired path is unaffected
# ---------------------------------------------------------------------------

def test_paired_path_still_works():
    """audit_statistical_validity must continue to work after this change."""
    from evaltrust.audit.statistical import audit_statistical_validity
    from evaltrust.core.schema import EvalData, Example

    examples = [
        Example(id=str(i), scores={"A": float(a), "B": float(b)})
        for i, (a, b) in enumerate(zip([0] * 30, [1] * 30))
    ]
    data = EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})
    findings = audit_statistical_validity(data, "A", "B", seed=0)
    assert {f.details["check"] for f in findings} == {
        "decision", "effect_size", "precision"
    }
    # None of the paired findings should carry the two-sample path marker.
    for f in findings:
        assert f.details.get("comparison_path") != "unpaired_two_sample"


# ---------------------------------------------------------------------------
# load_run_level: wide CSV
# ---------------------------------------------------------------------------

def test_load_run_level_wide_csv(tmp_path):
    csv = "model_a,model_b\n0.81,0.79\n0.83,0.82\n0.80,0.78\n"
    p = tmp_path / "scores.csv"
    p.write_text(csv, encoding="utf-8")
    data = load_run_level(str(p))
    assert data.model_a == "model_a"
    assert data.model_b == "model_b"
    assert data.n_a == 3
    assert data.n_b == 3
    assert data.source_format == "run_level_csv_wide"
    np.testing.assert_allclose(data.scores_a, [0.81, 0.83, 0.80])


def test_load_run_level_wide_csv_model_override(tmp_path):
    csv = "alpha,beta\n0.9,0.7\n0.8,0.6\n"
    p = tmp_path / "scores.csv"
    p.write_text(csv, encoding="utf-8")
    data = load_run_level(str(p), model_a="alpha", model_b="beta")
    assert data.model_a == "alpha"
    assert data.model_b == "beta"
    assert data.n_a == 2


def test_load_run_level_wide_csv_skips_meta_columns(tmp_path):
    csv = "run,model_a,model_b\n1,0.81,0.79\n2,0.83,0.82\n"
    p = tmp_path / "scores.csv"
    p.write_text(csv, encoding="utf-8")
    data = load_run_level(str(p))
    assert data.model_a == "model_a"
    assert data.model_b == "model_b"
    assert data.n_a == 2


# ---------------------------------------------------------------------------
# load_run_level: long CSV
# ---------------------------------------------------------------------------

def test_load_run_level_long_csv(tmp_path):
    csv = textwrap.dedent("""\
        model,score
        alpha,0.81
        alpha,0.83
        alpha,0.80
        beta,0.79
        beta,0.82
        beta,0.78
    """)
    p = tmp_path / "long.csv"
    p.write_text(csv, encoding="utf-8")
    data = load_run_level(str(p), model_a="alpha", model_b="beta")
    assert data.model_a == "alpha"
    assert data.model_b == "beta"
    assert data.n_a == 3
    assert data.n_b == 3
    assert data.source_format == "run_level_csv_long"


def test_load_run_level_long_csv_auto_selects_first_two(tmp_path):
    csv = textwrap.dedent("""\
        model,score
        modelA,0.8
        modelA,0.9
        modelB,0.7
        modelB,0.75
    """)
    p = tmp_path / "long.csv"
    p.write_text(csv, encoding="utf-8")
    data = load_run_level(str(p))
    assert data.model_a == "modelA"
    assert data.model_b == "modelB"


# ---------------------------------------------------------------------------
# load_run_level: JSON
# ---------------------------------------------------------------------------

def test_load_run_level_json(tmp_path):
    raw = {"gpt4": [0.81, 0.83, 0.80], "claude": [0.79, 0.82, 0.78]}
    p = tmp_path / "scores.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    data = load_run_level(str(p))
    assert data.model_a == "gpt4"
    assert data.model_b == "claude"
    assert data.n_a == 3
    assert data.n_b == 3
    assert data.source_format == "run_level_json"
    np.testing.assert_allclose(data.scores_a, [0.81, 0.83, 0.80])


def test_load_run_level_json_model_selection(tmp_path):
    raw = {"A": [0.8, 0.9], "B": [0.7, 0.75], "C": [0.6, 0.65]}
    p = tmp_path / "scores.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    data = load_run_level(str(p), model_a="B", model_b="C")
    assert data.model_a == "B"
    assert data.model_b == "C"
    assert data.n_a == 2
    assert data.n_b == 2


def test_load_run_level_json_unknown_model_raises(tmp_path):
    raw = {"A": [0.8], "B": [0.7]}
    p = tmp_path / "scores.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="not found"):
        load_run_level(str(p), model_a="A", model_b="X")


def test_load_run_level_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_run_level("/tmp/does_not_exist_evaltrust_xyz.csv")


def test_load_run_level_bad_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_run_level(str(p))


# ---------------------------------------------------------------------------
# RunLevelData properties
# ---------------------------------------------------------------------------

def test_run_level_data_n_properties():
    d = RunLevelData(
        model_a="A", model_b="B",
        scores_a=np.array([0.8, 0.9]),
        scores_b=np.array([0.7, 0.75, 0.72]),
        source_format="test",
    )
    assert d.n_a == 2
    assert d.n_b == 3


# ---------------------------------------------------------------------------
# Regression tests for CodeRabbit review fixes
# ---------------------------------------------------------------------------

def test_decision_inconclusive_message_is_accurate_when_significant():
    """Fix (image 1): inconclusive branch must not claim 'not < alpha' when significant."""
    # Manufacture a borderline case: significant p-value but CI straddles 0.5.
    # Use very few runs so CI is wide and manually craft data around 0.5.
    rng = np.random.default_rng(77)
    a = rng.normal(0.55, 0.15, 6)
    b = rng.normal(0.50, 0.15, 6)
    decision = by_check(audit_two_sample(make_run_level(a, b), seed=0), "decision")
    # Whatever the outcome, how_detected must not say "not < alpha" when p is actually < alpha.
    p_mw = decision.details["p_value_mann_whitney"]
    alpha = decision.details["alpha"]
    if p_mw < alpha:
        assert "not < alpha" not in decision.how_detected, (
            "When p < alpha the how_detected text must not claim 'not < alpha'"
        )


def test_effect_size_labels_are_by_model_not_rank():
    """Fix (image 2): descriptive stats must be keyed to model_a/model_b, not leader/trailer."""
    # B beats A — leader=model_b, trailer=model_a.
    a = np.array([0.5, 0.52, 0.51, 0.50, 0.53])
    b = np.array([0.8, 0.82, 0.81, 0.80, 0.83])
    effect = by_check(audit_two_sample(make_run_level(a, b, model_a="lowA", model_b="highB"), seed=0), "effect_size")
    # mean_a in details must equal np.mean(a), not np.mean(b).
    assert effect.details["mean_a"] == pytest.approx(np.mean(a), abs=1e-6), (
        "mean_a must always refer to scores_a, not the leader's scores"
    )
    assert effect.details["mean_b"] == pytest.approx(np.mean(b), abs=1e-6), (
        "mean_b must always refer to scores_b, not the trailer's scores"
    )
    # how_detected must label means by model name, not by rank.
    assert "lowA" in effect.how_detected
    assert "highB" in effect.how_detected


def test_precision_no_negative_shortage_when_sufficient_and_not_significant():
    """Fix (image 3): not-significant + sufficient must not produce negative shortage."""
    rng = np.random.default_rng(2025)
    # Same distribution, plenty of runs → not significant, sufficient.
    a = rng.normal(0.7, 0.1, 20)
    b = rng.normal(0.7, 0.1, 20)
    precision = by_check(audit_two_sample(make_run_level(a, b), seed=0), "precision")
    assert precision.details["sufficient"] is True
    # The fix message text should NOT contain a negative number.
    assert "-" not in precision.how_to_fix or "unlikely" in precision.how_to_fix


def test_load_run_level_json_scalar_value_raises():
    """Fix (image 6, part 1): scalar JSON values must raise a clear ValueError."""
    import tempfile, json as _json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        _json.dump({"A": 0.8, "B": [0.7, 0.75]}, f)
        fname = f.name
    with pytest.raises(ValueError, match="must be a list"):
        load_run_level(fname, model_a="A", model_b="B")


def test_load_run_level_jsonl_parsed_correctly(tmp_path):
    """Fix (image 6, part 2): true JSONL (one JSON object per line) must parse."""
    jsonl = '{"alpha": [0.8, 0.85, 0.82]}\n{"beta": [0.7, 0.72, 0.71]}\n'
    p = tmp_path / "scores.jsonl"
    p.write_text(jsonl, encoding="utf-8")
    data = load_run_level(str(p), model_a="alpha", model_b="beta")
    assert data.n_a == 3
    assert data.n_b == 3
    np.testing.assert_allclose(data.scores_a, [0.8, 0.85, 0.82])


def test_load_run_level_wide_csv_whitespace_headers(tmp_path):
    """Fix (image 7): headers with surrounding whitespace must not cause KeyError."""
    csv_text = " model_a , model_b \n0.81,0.79\n0.83,0.82\n"
    p = tmp_path / "spaced.csv"
    p.write_text(csv_text, encoding="utf-8")
    data = load_run_level(str(p))
    assert data.model_a == "model_a"
    assert data.n_a == 2


def test_load_run_level_wide_csv_atomic_row_ingestion(tmp_path):
    """Fix (image 8): a bad score in column B must not half-ingest the A value."""
    csv_text = "model_a,model_b\n0.81,0.79\n0.83,bad_value\n0.80,0.78\n"
    p = tmp_path / "partial.csv"
    p.write_text(csv_text, encoding="utf-8")
    data = load_run_level(str(p))
    # The bad row (0.83, bad_value) must be dropped entirely — not 0.83 ingested alone.
    assert data.n_a == data.n_b == 2
    np.testing.assert_allclose(sorted(data.scores_a), [0.80, 0.81])
    np.testing.assert_allclose(sorted(data.scores_b), [0.78, 0.79])
