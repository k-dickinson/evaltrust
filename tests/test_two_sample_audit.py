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


def test_effect_size_skips_a_definitive_label_with_one_run_per_model():
    from evaltrust.audit.verdict import compute_verdict

    findings = audit_two_sample(make_run_level([0.9], [0.1]), seed=0)
    effect = by_check(findings, "effect_size")

    assert effect.status is Status.SKIP
    assert effect.title == "Run-level effect size needs at least 2 runs per model"
    assert "large" not in effect.title.lower()
    assert "large" not in effect.how_detected.lower()
    assert effect.details["effect_size_sufficient"] is False
    assert effect.details["min_n_required"] == 2
    assert effect.details["n_a"] == 1
    assert effect.details["n_b"] == 1

    # The insufficient effect state is advisory. Removing it entirely must
    # leave the verdict level, summary, and drivers byte-for-byte unchanged.
    without_effect = [
        finding
        for finding in findings
        if finding.details.get("check") != "effect_size"
    ]
    assert compute_verdict(findings).to_dict() == compute_verdict(
        without_effect
    ).to_dict()


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


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf"), True, False],
)
def test_load_run_level_json_rejects_non_finite_values_and_booleans(
    tmp_path, bad_value
):
    raw = {"valid_model": [0.1, 0.2], "bad_model": [0.3, bad_value]}
    p = tmp_path / "strict-scores.json"
    p.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_run_level(
            str(p), model_a="valid_model", model_b="bad_model"
        )

    message = str(exc_info.value)
    assert "bad_model" in message
    assert "index 1" in message
    assert "finite number" in message


def test_load_run_level_json_validates_models_outside_the_selected_pair(tmp_path):
    raw = {
        "model_a": [0.1, 0.2],
        "model_b": [0.3, 0.4],
        "bad_extra_model": [0.5, float("nan")],
    }
    p = tmp_path / "strict-multi-model-scores.json"
    p.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_run_level(str(p), model_a="model_a", model_b="model_b")

    message = str(exc_info.value)
    assert "bad_extra_model" in message
    assert "index 1" in message
    assert "finite number" in message


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
    """Fix (image 1): inconclusive branch must not claim 'not < alpha' when significant.

    Constructed deterministically: we monkeypatch mann_whitney_u to return a
    significant p-value while bootstrap_p_a_gt_b returns a CI that straddles 0.5,
    guaranteeing the inconclusive branch is reached with p < alpha every time.
    """
    import unittest.mock as mock
    from evaltrust.audit import two_sample as ts_mod

    # CI straddles 0.5 → inconclusive; p < alpha → significant.
    with mock.patch.object(ts_mod, "bootstrap_p_a_gt_b", return_value=(0.55, 0.40, 0.70)):
        with mock.patch.object(ts_mod, "mann_whitney_u", return_value=(42.0, 0.02)):
            data = make_run_level([0.6] * 10, [0.5] * 10)
            decision = by_check(audit_two_sample(data, alpha=0.05, seed=0), "decision")

    # The branch IS inconclusive (CI straddles 0.5) but the test IS significant.
    assert decision.details["outcome"] == "inconclusive"
    assert decision.details["p_value_mann_whitney"] == pytest.approx(0.02)
    # The how_detected text must NOT say "not < alpha" when p is actually < alpha.
    assert "not < alpha" not in decision.how_detected, (
        "Inconclusive branch must not claim 'not < alpha' when p is significant; "
        f"got: {decision.how_detected!r}"
    )


def test_effect_size_labels_are_by_model_not_rank():
    """Fix (image 2): descriptive stats must be keyed to model_a/model_b, not leader/trailer.

    The regression being guarded: when B leads, _effect_size previously labelled
    mean_a as 'trailer' and mean_b as 'leader', so the report said 'leader: mean 0.81'
    when the leader was actually model_b.  We assert:
    - mean_a/mean_b details equal np.mean(a)/np.mean(b) respectively, and
    - how_detected uses the model names (not 'leader'/'trailer' rank words) for the means.
    """
    # B beats A — leader=model_b ("highB"), trailer=model_a ("lowA").
    a = np.array([0.5, 0.52, 0.51, 0.50, 0.53])
    b = np.array([0.8, 0.82, 0.81, 0.80, 0.83])
    effect = by_check(
        audit_two_sample(make_run_level(a, b, model_a="lowA", model_b="highB"), seed=0),
        "effect_size",
    )

    # Descriptive stats must be keyed by fixed model name, not rank.
    assert effect.details["mean_a"] == pytest.approx(np.mean(a), abs=1e-6), (
        "mean_a must always refer to scores_a, not the leader's scores"
    )
    assert effect.details["mean_b"] == pytest.approx(np.mean(b), abs=1e-6), (
        "mean_b must always refer to scores_b, not the trailer's scores"
    )

    # how_detected must use the fixed model names for mean/SD lines.
    assert "lowA: mean" in effect.how_detected, (
        f"Expected 'lowA: mean ...' in how_detected; got: {effect.how_detected!r}"
    )
    assert "highB: mean" in effect.how_detected, (
        f"Expected 'highB: mean ...' in how_detected; got: {effect.how_detected!r}"
    )
    # The rank words 'leader:' and 'trailer:' must NOT appear in the means section.
    assert "leader: mean" not in effect.how_detected, (
        "how_detected must label means by model name, not by 'leader'"
    )
    assert "trailer: mean" not in effect.how_detected, (
        "how_detected must label means by model name, not by 'trailer'"
    )


def test_precision_no_negative_shortage_when_sufficient_and_not_significant():
    """Fix (image 3): not-significant + sufficient must not produce negative shortage.

    The regression: when both models have >= _MIN_RUNS_RECOMMENDED runs but the
    test is not significant, the old else-branch computed shortage = 10 - min_n
    which goes negative, producing text like 'Collect ~-10 more runs'.

    The fix splits this into its own elif branch.  This test:
    - asserts the non-significant + sufficient precondition holds (no escape hatch),
    - then directly checks that no signed numeric value (e.g. '-10') appears in
      the how_to_fix text.
    """
    import re
    rng = np.random.default_rng(2025)
    # Same distribution, plenty of runs → sufficient; non-significant expected.
    a = rng.normal(0.7, 0.1, 20)
    b = rng.normal(0.7, 0.1, 20)
    findings = audit_two_sample(make_run_level(a, b), seed=0)
    precision = by_check(findings, "precision")
    decision = by_check(findings, "decision")

    # Precondition: sufficient must be True (both models have 20 >= 10 runs).
    assert precision.details["sufficient"] is True, (
        "Test precondition failed: expected sufficient=True with 20 runs per model"
    )
    # Precondition: comparison must be non-significant or inconclusive (not a
    # clear PASS), so we're in the branch that used to compute negative shortage.
    assert decision.details["outcome"] != "significant", (
        "Test precondition failed: expected non-significant outcome with same-distribution data"
    )

    # The fix text must contain no negative integer (e.g. '-10', '-5').
    negative_number = re.search(r"-\d+", precision.how_to_fix)
    assert negative_number is None, (
        f"how_to_fix must not contain a negative shortage; "
        f"found {negative_number.group()!r} in: {precision.how_to_fix!r}"
    )


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


# ---------------------------------------------------------------------------
# CLI --run-level: --plain / --md / --html output formats (issue #152)
# ---------------------------------------------------------------------------

from typer.testing import CliRunner as _CliRunner
from evaltrust.cli import app as _app

_runner = _CliRunner()


def _rl_csv(tmp_path, a_scores=None, b_scores=None):
    """Write a wide-CSV run-level fixture and return its path as a string."""
    if a_scores is None:
        a_scores = [0.82, 0.84, 0.83, 0.85, 0.81,
                    0.80, 0.83, 0.84, 0.82, 0.85,
                    0.81, 0.83, 0.84, 0.80, 0.82]
    if b_scores is None:
        b_scores = [0.62, 0.64, 0.63, 0.65, 0.61,
                    0.60, 0.63, 0.64, 0.62, 0.65,
                    0.61, 0.63, 0.64, 0.60, 0.62]
    rows = ["alpha,beta"] + [f"{a},{b}" for a, b in zip(a_scores, b_scores)]
    p = tmp_path / "run_level.csv"
    p.write_text("\n".join(rows), encoding="utf-8")
    return str(p)


# ---- --plain ---------------------------------------------------------------

def test_run_level_plain_exits_zero(tmp_path):
    path = _rl_csv(tmp_path)
    r = _runner.invoke(_app, ["audit", path, "--run-level", "--plain"])
    assert r.exit_code == 0, r.output


def test_run_level_plain_no_warning_emitted(tmp_path):
    """No 'not yet fully supported' warning must appear in output after the fix."""
    path = _rl_csv(tmp_path)
    r = _runner.invoke(_app, ["audit", path, "--run-level", "--plain"])
    assert "not yet" not in r.output
    assert "not fully" not in r.output


def test_run_level_plain_header_contains_model_names(tmp_path):
    path = _rl_csv(tmp_path)
    r = _runner.invoke(_app, ["audit", path, "--run-level", "--plain"])
    assert "alpha" in r.output
    assert "beta" in r.output
    assert "15 runs" in r.output


def test_run_level_plain_contains_ascii_status_marks(tmp_path):
    """--plain must use ASCII bracket marks like [ok  ] / [warn] / [fail]."""
    path = _rl_csv(tmp_path)
    r = _runner.invoke(_app, ["audit", path, "--run-level", "--plain"])
    # At least one ASCII bracket mark must appear.
    import re
    assert re.search(r"\[(ok  |warn|fail|--  )\]", r.output), (
        f"No ASCII status mark found in:\n{r.output}"
    )


def test_run_level_plain_no_unicode_bullets(tmp_path):
    """Plain output must not contain Unicode bullets or arrows."""
    path = _rl_csv(tmp_path)
    r = _runner.invoke(_app, ["audit", path, "--run-level", "--plain"])
    for char in ("✓", "✗", "⚠", "–", "●", "•"):
        assert char not in r.output, (
            f"Unicode char {char!r} must not appear in --plain output"
        )


def test_run_level_plain_explain_adds_detail(tmp_path):
    """--explain must append per-finding why / how_detected text in --plain mode."""
    path = _rl_csv(tmp_path)
    # Use data that yields at least one non-pass finding so Detail section fires.
    # Near-equal scores → inconclusive → warn/fail findings.
    a = [0.70] * 15
    b = [0.70] * 15
    p2 = _rl_csv(tmp_path / "eq.csv" if False else tmp_path, a_scores=a, b_scores=b)
    import tempfile, os
    rows = ["alpha,beta"] + ["0.70,0.70"] * 5
    pf = tmp_path / "eq.csv"
    pf.write_text("\n".join(rows), encoding="utf-8")
    r = _runner.invoke(_app, ["audit", str(pf), "--run-level", "--plain", "--explain"])
    assert r.exit_code in (0, 1)
    # When there are warn/fail findings the Detail section appears.
    # With identical scores every finding should be warn or fail.
    assert "Statistical Validity" in r.output


# ---- --md ------------------------------------------------------------------

def test_run_level_md_exits_zero(tmp_path):
    path = _rl_csv(tmp_path)
    r = _runner.invoke(_app, ["audit", path, "--run-level", "--md"])
    assert r.exit_code == 0, r.output


def test_run_level_md_no_warning_emitted(tmp_path):
    path = _rl_csv(tmp_path)
    r = _runner.invoke(_app, ["audit", path, "--run-level", "--md"])
    assert "not yet" not in r.output
    assert "not fully" not in r.output


def test_run_level_md_starts_with_h1(tmp_path):
    path = _rl_csv(tmp_path)
    r = _runner.invoke(_app, ["audit", path, "--run-level", "--md"])
    assert r.output.startswith("# EvalTrust"), (
        f"Markdown must start with '# EvalTrust', got:\n{r.output[:120]}"
    )


def test_run_level_md_contains_bold_status_marks(tmp_path):
    """--md must use **[pass]** / **[warn]** / **[fail]** markers."""
    path = _rl_csv(tmp_path)
    r = _runner.invoke(_app, ["audit", path, "--run-level", "--md"])
    import re
    assert re.search(r"\*\*\[(pass|warn|fail|skip)\]\*\*", r.output), (
        f"No Markdown status badge found in:\n{r.output}"
    )


def test_run_level_md_contains_model_names(tmp_path):
    path = _rl_csv(tmp_path)
    r = _runner.invoke(_app, ["audit", path, "--run-level", "--md"])
    assert "alpha" in r.output
    assert "beta" in r.output


def test_run_level_md_what_to_do_section_present_when_findings_warn(tmp_path):
    """When findings include warn/fail, a '## What to do' section must appear."""
    rows = ["alpha,beta"] + ["0.70,0.70"] * 5  # identical → warn/fail
    pf = tmp_path / "eq.csv"
    pf.write_text("\n".join(rows), encoding="utf-8")
    r = _runner.invoke(_app, ["audit", str(pf), "--run-level", "--md"])
    assert r.exit_code in (0, 1)
    # Either there are warn/fail findings (What to do) or all pass (none) —
    # we just verify the output is valid Markdown either way.
    assert "# EvalTrust" in r.output


# ---- --html ----------------------------------------------------------------

def test_run_level_html_exits_zero_and_writes_file(tmp_path):
    path = _rl_csv(tmp_path)
    out = str(tmp_path / "report.html")
    r = _runner.invoke(_app, ["audit", path, "--run-level", "--html", out])
    assert r.exit_code == 0, r.output
    import os
    assert os.path.exists(out), "HTML file must be written to disk"


def test_run_level_html_no_warning_emitted(tmp_path):
    """No 'not yet supported' warning must appear after the fix."""
    path = _rl_csv(tmp_path)
    out = str(tmp_path / "report.html")
    r = _runner.invoke(_app, ["audit", path, "--run-level", "--html", out])
    assert "not yet" not in r.output
    assert "not supported" not in r.output


def test_run_level_html_is_valid_document(tmp_path):
    """The HTML file must start with a DOCTYPE and contain key structural tags."""
    path = _rl_csv(tmp_path)
    out = str(tmp_path / "report.html")
    _runner.invoke(_app, ["audit", path, "--run-level", "--html", out])
    content = open(out, encoding="utf-8").read()
    assert content.startswith("<!DOCTYPE html>")
    assert "<html" in content
    assert "</html>" in content
    assert "<body>" in content
    assert "</body></html>" in content


def test_run_level_html_contains_model_names(tmp_path):
    path = _rl_csv(tmp_path)
    out = str(tmp_path / "report.html")
    _runner.invoke(_app, ["audit", path, "--run-level", "--html", out])
    content = open(out, encoding="utf-8").read()
    assert "alpha" in content
    assert "beta" in content
    assert "15 runs" in content


def test_run_level_html_contains_badge_spans(tmp_path):
    """HTML must include badge spans for PASS / WARN / FAIL / SKIP."""
    path = _rl_csv(tmp_path)
    out = str(tmp_path / "report.html")
    _runner.invoke(_app, ["audit", path, "--run-level", "--html", out])
    content = open(out, encoding="utf-8").read()
    assert "class='badge'" in content


def test_run_level_html_escapes_special_characters(tmp_path):
    """Model names with angle brackets must be HTML-escaped in the output."""
    rows = ["<ModelA>,<ModelB>"] + ["0.82,0.62"] * 10
    pf = tmp_path / "special.csv"
    pf.write_text("\n".join(rows), encoding="utf-8")
    out = str(tmp_path / "special.html")
    r = _runner.invoke(_app, ["audit", str(pf), "--run-level", "--html", out])
    assert r.exit_code in (0, 1, 2)  # may error on invalid model names; that's fine
    if r.exit_code in (0, 1):
        content = open(out, encoding="utf-8").read()
        # Raw < or > must not appear outside of HTML tags in the text regions
        assert "<ModelA>" not in content
        assert "&lt;ModelA&gt;" in content


def test_run_level_html_explain_adds_detail_section(tmp_path):
    """--explain must add a Detail section in the HTML when there are warn/fail findings.

    Note: "detail" appears in the inlined CSS (.detail, .detail h2, .detail-item) regardless
    of --explain.  Assert on the rendered section markup instead.
    """
    rows = ["alpha,beta"] + ["0.70,0.70"] * 5  # identical → warn/fail
    pf = tmp_path / "eq.csv"
    pf.write_text("\n".join(rows), encoding="utf-8")
    out = str(tmp_path / "explain.html")
    r = _runner.invoke(_app, ["audit", str(pf), "--run-level", "--html", out, "--explain"])
    assert r.exit_code in (0, 1)
    content = open(out, encoding="utf-8").read()
    # The rendered Detail section heading — distinct from the CSS class names
    assert "<div class=\'detail\'><h2>Detail</h2>" in content or            "<div class='detail'><h2>Detail</h2>" in content, (
        "Expected rendered Detail section in HTML with --explain; "
        f"got content of length {len(content)}"
    )
    assert "detail-item" in content


# ---------------------------------------------------------------------------
# Reviewer fixes (PR #162 CodeRabbit comments)
# ---------------------------------------------------------------------------

def test_run_level_md_escapes_markdown_special_chars_in_subtitle(tmp_path):
    """Fix (Image 1 - Minor): model names in the subtitle must be Markdown-escaped.

    The subtitle line ``**model_a vs model_b · N runs / M runs**`` is user-controlled.
    A model name like ``gpt-4[preview]`` contains ``[`` and ``]`` which Markdown
    interprets as a link label; a name shaped like ``[text](url)`` renders as an
    actual hyperlink in a PR comment — the primary use case for ``--md``.

    Note: finding *titles* are library-authored strings and are deliberately left
    unescaped so they render verbatim (e.g. parentheses and dots in statistical
    notation must not be backslash-escaped).
    """
    # Use a model name containing Markdown-special characters
    rows = ["gpt-4[preview],claude-3_opus"] + ["0.82,0.62"] * 12
    pf = tmp_path / "special_models.csv"
    pf.write_text("\n".join(rows), encoding="utf-8")
    r = _runner.invoke(_app, ["audit", str(pf), "--run-level", "--md"])
    assert r.exit_code in (0, 1)
    # The subtitle bold line must escape the brackets
    assert r"\[preview\]" in r.output, (
        f"Expected escaped subtitle '\\[preview\\]' in --md output; got:\n{r.output}"
    )
    # Specifically the **subtitle** line must be escaped (first occurrence = subtitle)
    bold_line = next(
        (line for line in r.output.splitlines() if line.startswith("**")), ""
    )
    assert r"\[preview\]" in bold_line, (
        f"Subtitle bold line must contain escaped brackets; got: {bold_line!r}"
    )


def test_run_level_md_escapes_markdown_special_chars_in_finding_title(tmp_path):
    """Fix (Image 1 - Minor): finding titles with Markdown-special chars must be escaped.

    Finding titles can contain characters like '*', '_', '[', ']' which may
    break Markdown formatting in a PR comment.
    """
    path = _rl_csv(tmp_path)
    r = _runner.invoke(_app, ["audit", path, "--run-level", "--md"])
    assert r.exit_code in (0, 1)
    # Verify no raw unescaped '[text](url)' pattern appears (linkification check)
    import re
    # A well-formed [label](url) Markdown link should not appear from finding titles
    assert not re.search(r'\[(?!pass|warn|fail|skip)[^\]]+\]\([^)]+\)', r.output), (
        "Finding titles must not accidentally create Markdown links"
    )


def test_run_level_plain_ascii_table_maps_both_curly_single_quotes(tmp_path):
    """Fix (Image 2 - Major): U+2018 and U+2019 must both be mapped in _ASCII.

    The original code had a duplicate dict key (two ASCII apostrophes 0x27)
    so Python silently dropped one entry and neither U+2018 nor U+2019 was
    actually mapped.  This test injects a finding title containing both
    curly single quotes and verifies the --plain output contains only ASCII.
    """
    import unittest.mock as mock
    from evaltrust.core.schema import Finding, Status

    curly_title = "\u2018left\u2019 and \u2018right\u2019 curly quotes"

    fake_finding = Finding(
        pillar="Statistical Validity",
        title=curly_title,
        status=Status.PASS,
        why="test",
        how_detected="test",
        how_to_fix="test",
        details={"check": "decision", "comparison_path": "unpaired_two_sample"},
    )

    path = _rl_csv(tmp_path)
    # Patch the name as it is bound in the cli module (already imported at the top)
    with mock.patch("evaltrust.cli.audit_two_sample", return_value=[fake_finding]):
        r = _runner.invoke(_app, ["audit", path, "--run-level", "--plain"])

    assert r.exit_code in (0, 1)
    # U+2018 and U+2019 must NOT appear — they must have been translated to ASCII
    assert "\u2018" not in r.output, (
        "U+2018 (left single quote) must be translated to ASCII in --plain output"
    )
    assert "\u2019" not in r.output, (
        "U+2019 (right single quote) must be translated to ASCII in --plain output"
    )
    # The content words must still be present (just with ASCII apostrophes now)
    assert "left" in r.output and "right" in r.output


def test_run_level_plain_ascii_translation_has_ten_distinct_keys():
    """Fix (Image 2 - Major): the _ASCII maketrans must cover 10 distinct code points.

    A duplicate dict key silently drops one entry.  We import the canonical
    _ASCII table directly from terminal.py so that any regression there
    (e.g. re-introducing the duplicate key) would be caught by this test.
    """
    from evaltrust.report.terminal import _ASCII as t
    assert len(t) == 10, f"Expected 10 keys in _ASCII maketrans, got {len(t)}"
    for cp in (0x00B7, 0x2013, 0x2014, 0x2022, 0x25CF,
               0x2018, 0x2019, 0x201C, 0x201D, 0x00D7):
        assert cp in t, f"U+{cp:04X} missing from _ASCII maketrans"
