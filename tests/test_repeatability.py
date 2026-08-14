"""Tests for the Repeatability audit (uses repeated-run data when present)."""

from evaltrust.audit.repeatability import audit_repeatability
from evaltrust.core.schema import EvalData, Example, Status


def make_data(runs_a, runs_b, n=40):
    """n identical examples, each carrying the given per-run score lists."""
    examples = [
        Example(id=str(i), scores={"A": runs_a[0], "B": runs_b[0]},
                runs={"A": list(runs_a), "B": list(runs_b)})
        for i in range(n)
    ]
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


def no_runs_data():
    examples = [Example(id=str(i), scores={"A": 0.0, "B": 1.0}) for i in range(10)]
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


def by_check(findings, check):
    (f,) = [f for f in findings if f.details.get("check") == check]
    return f


def test_skips_when_no_repeated_runs():
    findings = audit_repeatability(no_runs_data(), "A", "B")
    assert len(findings) == 1
    assert findings[0].status is Status.SKIP
    assert "run" in findings[0].how_to_fix.lower()


def test_stable_reruns_pass():
    # B beats A by the same margin on every rerun.
    findings = audit_repeatability(make_data([0, 0, 0], [1, 1, 1]), "A", "B")
    assert by_check(findings, "rerun_stability").status is Status.PASS
    assert by_check(findings, "measurement_variance").status is Status.PASS


def test_ranking_that_flips_across_reruns_is_not_pass():
    # Run 0 favours B, run 1 favours A: the winner is a coin flip.
    findings = audit_repeatability(make_data([0.0, 0.9], [0.5, 0.4]), "A", "B")
    assert by_check(findings, "rerun_stability").status in {Status.WARN, Status.FAIL}


def test_noisy_but_consistent_direction_flags_variance_only():
    # Gap stays positive but swings a lot run to run.
    findings = audit_repeatability(make_data([0, 0, 0, 0], [0.05, 0.9, 0.05, 0.9]),
                                   "A", "B")
    assert by_check(findings, "rerun_stability").status is Status.PASS
    assert by_check(findings, "measurement_variance").status is Status.WARN


def test_findings_obey_golden_rule():
    findings = audit_repeatability(make_data([0, 0], [1, 1]), "A", "B")
    for f in findings:
        assert f.why.strip() and f.how_detected.strip() and f.how_to_fix.strip()
        assert f.pillar == "Repeatability"


def test_exact_tie_every_run_does_not_name_a_false_winner():
    # Both models score identically on every run: they are perfectly tied.
    # The finding must not claim either model "wins consistently".
    findings = audit_repeatability(make_data([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                   "A", "B")
    stab = by_check(findings, "rerun_stability")
    assert stab.status is Status.PASS
    assert stab.details["mean_gap"] == 0.0
    text = (stab.title + stab.how_detected + stab.how_to_fix).lower()
    assert "wins consistently" not in text
    assert "tie" in text or "indistinguishable" in text


def test_near_zero_mean_gap_from_cancellation_is_treated_as_a_tie():
    # Per-run gaps [0.1, 0.2, -0.3] sum to a floating-point residue (~1.85e-17),
    # not exactly 0.0. A single example keeps the residue (averaging across many
    # would cancel it). The models are tied and no winner must be named.
    findings = audit_repeatability(
        make_data([0.0, 0.0, 0.0], [0.1, 0.2, -0.3], n=1), "A", "B"
    )
    stab = by_check(findings, "rerun_stability")
    assert stab.details["mean_gap"] != 0.0          # confirms the cancellation path
    assert abs(stab.details["mean_gap"]) < 1e-12
    assert "the winner was" not in stab.how_detected  # no model named as winner
    assert "neither model led" in stab.how_detected
    assert "wins consistently" not in (
        stab.title + stab.how_detected + stab.how_to_fix
    ).lower()


def test_tie_with_alternating_direction_is_not_pass_and_names_no_winner():
    # Gap is +0.5, -0.5 across two runs: mean is exactly 0 but the direction
    # flips, so this is noise, not a stable tie.
    findings = audit_repeatability(make_data([0.0, 0.5], [0.5, 0.0]), "A", "B")
    stab = by_check(findings, "rerun_stability")
    assert stab.status in {Status.WARN, Status.FAIL}
    assert stab.details["mean_gap"] == 0.0
    assert "wins consistently" not in (
        stab.title + stab.how_detected + stab.how_to_fix
    ).lower()
