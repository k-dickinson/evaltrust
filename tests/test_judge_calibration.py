"""Tests for judge calibration: when the file contains a human/gold judge, measure
how well each AI judge agrees with it (treated as ground truth)."""

import pytest
from scipy.stats import spearmanr

from evaltrust.audit.judge_calibration import audit_judge_calibration
from evaltrust.core.schema import EvalData, Example, Status


def make(judge_examples):
    examples = []
    for i, judges in enumerate(judge_examples):
        first = next(iter(judges.values()))
        examples.append(Example(id=str(i), scores=dict(first), judges=judges))
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


def by_check(findings, check="judge_calibration"):
    return [f for f in findings if f.details.get("check") == check]


def test_no_finding_without_judges():
    data = EvalData(models=["A", "B"],
                    examples=[Example("q1", {"A": 1, "B": 0})],
                    source_format="t", metadata={})
    assert audit_judge_calibration(data, "A", "B") == []


def test_no_finding_without_a_gold_judge():
    ex = [{"gpt": {"A": 1, "B": 0}, "claude": {"A": 1, "B": 0}} for _ in range(10)]
    assert audit_judge_calibration(make(ex), "A", "B") == []


def test_well_calibrated_judge_passes():
    ex = [{"gpt": {"A": 1, "B": 0}, "gold": {"A": 1, "B": 0}} for _ in range(20)]
    f = by_check(audit_judge_calibration(make(ex), "A", "B"))[0]
    assert f.status is Status.PASS
    assert f.details["accuracies"]["gpt"] == 1.0


def test_poorly_calibrated_judge_warns_and_is_named():
    ex = []
    for i in range(20):
        gold = {"A": 1, "B": 0}
        claude = gold if i % 2 == 0 else {"A": 0, "B": 1}   # disagrees half the time
        ex.append({"gold": gold, "gpt": {"A": 1, "B": 0}, "claude": claude})
    f = by_check(audit_judge_calibration(make(ex), "A", "B"))[0]
    assert f.status is Status.WARN
    assert f.details["worst_judge"] == "claude"
    assert f.pillar == "Judge Reliability"


def test_reference_judge_can_be_named_explicitly():
    ex = [{"gpt": {"A": 1, "B": 0}, "human_expert": {"A": 1, "B": 0}} for _ in range(10)]
    # 'human_expert' isn't in the default alias set, so pass it explicitly.
    findings = audit_judge_calibration(make(ex), "A", "B",
                                       reference_judge="human_expert")
    assert by_check(findings)[0].details["reference"] == "human_expert"


# --- continuous scores: correlation instead of exact match (issue #27) --------


def _continuous(pairs, judge="gpt", ref="gold"):
    """pairs: list of (ref_A, ref_B, judge_A, judge_B) on a continuous scale."""
    ex = [{ref: {"A": ra, "B": rb}, judge: {"A": ja, "B": jb}}
          for (ra, rb, ja, jb) in pairs]
    return make(ex)


def _expected_rho(pairs):
    ref_vals, judge_vals = [], []
    for ra, rb, ja, jb in pairs:            # collected example-by-example, A then B
        ref_vals += [ra, rb]
        judge_vals += [ja, jb]
    return float(spearmanr(judge_vals, ref_vals).statistic)


def test_binary_calibration_is_exact_match_and_byte_identical():
    # Binary judge scores must still take the exact-match path unchanged.
    ex = [{"gpt": {"A": 1, "B": 0}, "gold": {"A": 1, "B": 0}} for _ in range(20)]
    f = by_check(audit_judge_calibration(make(ex), "A", "B"))[0]
    assert f.status is Status.PASS
    assert f.title == "Judges track gold"
    assert f.how_detected == "Agreement with gold (treated as ground truth): gpt 100%."
    assert f.details["accuracies"] == {"gpt": 1.0}
    assert f.details["worst_accuracy"] == 1.0
    assert "metric" not in f.details          # binary details are unchanged
    assert "correlations" not in f.details


def test_continuous_scores_switch_to_spearman_validated_against_scipy():
    pairs = [(5, 1, 4, 2), (4, 2, 5, 1), (3, 3, 3, 2), (2, 4, 2, 4), (1, 5, 1, 5)]
    f = by_check(audit_judge_calibration(_continuous(pairs), "A", "B"))[0]
    assert f.details["metric"] == "spearman"
    assert f.details["correlations"]["gpt"] == pytest.approx(_expected_rho(pairs))
    assert "Spearman" in f.how_detected            # names the metric, not "%"
    assert "accuracies" not in f.details


def test_continuous_rank_perfect_but_offset_judge_passes():
    # Judge ranks items exactly like gold but every score is shifted +2. Spearman
    # sees a perfect monotonic relationship -> rho = 1.0 -> PASS. Intentional: for
    # A-vs-B comparison, consistent ranking is what matters; a constant offset
    # changes no ranking. (A tool needing absolute calibration would want a
    # different metric -- out of scope here.)
    pairs = [(5, 1, 7, 3), (4, 2, 6, 4), (3, 1, 5, 3), (2, 4, 4, 6), (1, 5, 3, 7)]
    f = by_check(audit_judge_calibration(_continuous(pairs), "A", "B"))[0]
    assert f.status is Status.PASS
    assert f.details["correlations"]["gpt"] == pytest.approx(1.0)


def test_continuous_ties_match_scipy():
    pairs = [(3, 3, 2, 2), (2, 2, 2, 2), (1, 3, 1, 3), (3, 1, 3, 1)]
    f = by_check(audit_judge_calibration(_continuous(pairs), "A", "B"))[0]
    assert f.details["correlations"]["gpt"] == pytest.approx(_expected_rho(pairs))


def test_continuous_too_few_points_skips_not_crashes():
    ex = [{"gold": {"A": 5}, "gpt": {"A": 3}}]      # one comparable point, non-binary
    f = by_check(audit_judge_calibration(make(ex), "A", "B"))[0]
    assert f.status is Status.SKIP
    assert f.details["metric"] == "spearman"


def test_continuous_threshold_is_a_correlation_floor_not_an_agreement_rate():
    pairs = [(5, 1, 1, 2), (4, 2, 3, 1), (3, 3, 2, 5), (2, 4, 4, 3), (1, 5, 5, 4)]
    data = _continuous(pairs)
    rho = by_check(audit_judge_calibration(data, "A", "B"))[0].details["correlations"]["gpt"]
    # The correlation path is governed by `correlation_threshold` (a rho floor).
    strict = by_check(
        audit_judge_calibration(data, "A", "B", correlation_threshold=rho + 0.05))[0]
    loose = by_check(
        audit_judge_calibration(data, "A", "B", correlation_threshold=rho - 0.05))[0]
    assert strict.status is Status.WARN
    assert loose.status is Status.PASS


def test_agreement_and_correlation_thresholds_are_independent_across_paths():
    # A fraction-agreed floor and a rank-correlation floor are different bars, so
    # they must move the two paths independently. Binary judge (agreement 0.75):
    # only `threshold` moves it; `correlation_threshold` is inert.
    ex = []
    for i in range(20):
        gold = {"A": 1, "B": 0}
        gpt = gold if i % 4 != 0 else {"A": 0, "B": 1}    # agrees 75% of the time
        ex.append({"gold": gold, "gpt": gpt})
    binary = make(ex)
    assert by_check(audit_judge_calibration(binary, "A", "B", threshold=0.7,
                    correlation_threshold=0.99))[0].status is Status.PASS
    assert by_check(audit_judge_calibration(binary, "A", "B", threshold=0.8,
                    correlation_threshold=0.01))[0].status is Status.WARN

    # Continuous judge: only `correlation_threshold` moves it; `threshold` (the
    # agreement floor) is inert on this path. The floor is set relative to the
    # scipy reference rho, not the audit's own number.
    pairs = [(5, 1, 1, 2), (4, 2, 3, 1), (3, 3, 2, 5), (2, 4, 4, 3), (1, 5, 5, 4)]
    cont = _continuous(pairs)
    rho = _expected_rho(pairs)              # scipy.stats.spearmanr reference
    assert by_check(audit_judge_calibration(cont, "A", "B", threshold=0.99,
                    correlation_threshold=rho - 0.05))[0].status is Status.PASS
    assert by_check(audit_judge_calibration(cont, "A", "B", threshold=0.01,
                    correlation_threshold=rho + 0.05))[0].status is Status.WARN


def test_threshold_still_governs_correlation_when_correlation_threshold_is_unset():
    # Backwards compatibility: before the split, one `threshold` governed BOTH
    # paths. A caller that passes only `threshold` on continuous data must still
    # get it as the rho floor (correlation_threshold defaults to "fall back to
    # threshold"), so no existing caller silently changes behaviour. Floor is set
    # against the scipy reference rho.
    pairs = [(5, 1, 1, 2), (4, 2, 3, 1), (3, 3, 2, 5), (2, 4, 4, 3), (1, 5, 5, 4)]
    data = _continuous(pairs)
    rho = _expected_rho(pairs)
    assert by_check(
        audit_judge_calibration(data, "A", "B", threshold=rho + 0.05))[0].status is Status.WARN
    assert by_check(
        audit_judge_calibration(data, "A", "B", threshold=rho - 0.05))[0].status is Status.PASS


def test_defaults_keep_both_floors_at_0_8_so_output_is_unchanged():
    # With correlation_threshold unset it falls back to threshold (default 0.8) —
    # exactly the single-knob behaviour main used — so the default call is
    # byte-identical to passing 0.8, and a continuous judge is still judged at
    # rho >= 0.8 (main's number, from scipy).
    pairs = [(5, 1, 1, 2), (4, 2, 3, 1), (3, 3, 2, 5), (2, 4, 4, 3), (1, 5, 5, 4)]
    cont = _continuous(pairs)
    default = by_check(audit_judge_calibration(cont, "A", "B"))[0]
    explicit = by_check(audit_judge_calibration(cont, "A", "B", threshold=0.8))[0]
    assert default.to_dict() == explicit.to_dict()
    assert (default.status is Status.PASS) == (_expected_rho(pairs) >= 0.8)


def test_config_correlation_threshold_flows_through_run_audit():
    # End-to-end: the config's judge_correlation_threshold reaches the calibration
    # check through run_audit -> _comparison. Floor set against the scipy rho.
    from evaltrust.audit.runner import run_audit
    from evaltrust.config import AuditConfig
    pairs = [(5, 1, 1, 2), (4, 2, 3, 1), (3, 3, 2, 5), (2, 4, 4, 3), (1, 5, 5, 4)]
    data = _continuous(pairs)
    rho = _expected_rho(pairs)
    strict = run_audit(data, model_a="A", model_b="B",
                       config=AuditConfig(judge_correlation_threshold=rho + 0.05))
    loose = run_audit(data, model_a="A", model_b="B",
                      config=AuditConfig(judge_correlation_threshold=rho - 0.05))
    assert by_check(strict.findings)[0].status is Status.WARN
    assert by_check(loose.findings)[0].status is Status.PASS


# --- edge cases that must not regress the original / must degrade gracefully ---


def test_string_binary_scores_take_exact_match_path_without_crashing():
    # Non-numeric scores the original == comparison handled must still work and
    # not hit float() in the metric-detection predicate.
    ex = [{"gpt": {"A": "pass", "B": "fail"}, "gold": {"A": "pass", "B": "fail"}}
          for _ in range(5)]
    f = by_check(audit_judge_calibration(make(ex), "A", "B"))[0]
    assert f.status is Status.PASS
    assert "accuracies" in f.details and "correlations" not in f.details


def test_no_ai_judges_returns_empty_even_for_continuous_reference():
    # Only the reference judge is present (continuous scores). There is nothing to
    # calibrate, so the result is silent [], as on main -- not a SKIP finding.
    ex = [{"gold": {"A": 5, "B": 1}} for _ in range(5)]
    assert audit_judge_calibration(make(ex), "A", "B") == []


def test_continuous_constant_scores_skip_with_an_accurate_reason():
    # Plenty of pairs, but the judge never varies -> Spearman is undefined. The
    # SKIP must say so, not falsely claim "fewer than two pairs".
    ex = [{"gpt": {"A": 5, "B": 5}, "gold": {"A": 3, "B": 3}} for _ in range(5)]
    f = by_check(audit_judge_calibration(make(ex), "A", "B"))[0]
    assert f.status is Status.SKIP
    assert "vary" in f.how_detected
    assert "Fewer than two" not in f.how_detected


def test_continuous_no_comparable_models_returns_empty_like_main():
    # AI judge scored a different model than the reference (no overlap on A/B):
    # nothing to calibrate -> silent [], not a SKIP, matching main's behaviour.
    ex = [{"gold": {"A": 5, "B": 3}, "gpt": {"C": 4}} for _ in range(3)]
    assert audit_judge_calibration(make(ex), "A", "B") == []


# --- single-model mode (#21) -------------------------------------------------

def test_single_model_calibration_against_gold():
    # One model, a gold judge and an AI judge that mostly agrees with it.
    ex = [{"gold": {"A": float(i % 2)},
           "gpt": {"A": float(i % 2) if i < 8 else float((i + 1) % 2)}}
          for i in range(10)]
    findings = audit_judge_calibration(make(ex), "A")   # single model
    # Assert the calibration actually computed (not just a SKIP finding):
    # gpt matches gold on 8 of 10 items -> 80% agreement against the reference.
    [finding] = [f for f in findings
                 if f.details.get("check") == "judge_calibration"]
    assert finding.details["reference"] == "gold"
    assert abs(finding.details["accuracies"]["gpt"] - 0.8) < 1e-12
