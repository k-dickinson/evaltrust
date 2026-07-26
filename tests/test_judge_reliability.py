"""Tests for the Judge Reliability audit (uses multi-judge data when present)."""

from evaltrust.audit.judge_reliability import audit_judge_reliability
from evaltrust.core.schema import EvalData, Example, Status


def make_data(judge_examples):
    """judge_examples: list of {judge: {model: score}} dicts, one per example."""
    examples = []
    for i, judges in enumerate(judge_examples):
        first = next(iter(judges.values()))
        examples.append(Example(id=str(i), scores=dict(first), judges=judges))
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


def by_check(findings, check):
    (f,) = [f for f in findings if f.details.get("check") == check]
    return f


def no_judges_data():
    examples = [Example(id=str(i), scores={"A": 0.0, "B": 1.0}) for i in range(10)]
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


def one_judge_data():
    ex = [{"gpt": {"A": 0.0, "B": 1.0}} for _ in range(10)]
    return make_data(ex)


def test_skips_when_no_judges():
    findings = audit_judge_reliability(no_judges_data(), "A", "B")
    assert len(findings) == 1 and findings[0].status is Status.SKIP


def test_skips_when_only_one_judge():
    findings = audit_judge_reliability(one_judge_data(), "A", "B")
    assert findings[0].status is Status.SKIP
    assert "judge" in findings[0].how_to_fix.lower()


def test_agreeing_judges_pass_consensus_and_agreement():
    ex = [{"gpt": {"A": 0.0, "B": 1.0}, "claude": {"A": 0.0, "B": 1.0}}
          for _ in range(20)]
    findings = audit_judge_reliability(make_data(ex), "A", "B")
    assert by_check(findings, "judge_consensus").status is Status.PASS
    assert by_check(findings, "inter_judge_agreement").status is Status.PASS


def test_judges_disagreeing_on_winner_fail_consensus():
    ex = [{"gpt": {"A": 0.0, "B": 1.0}, "claude": {"A": 1.0, "B": 0.0}}
          for _ in range(20)]
    consensus = by_check(audit_judge_reliability(make_data(ex), "A", "B"),
                         "judge_consensus")
    assert consensus.status is Status.FAIL


def test_low_agreement_warns_and_names_outlier():
    ex = []
    for i in range(20):
        if i % 2 == 0:
            gemini = {"A": 0.0, "B": 1.0}   # agrees
        else:
            gemini = {"A": 1.0, "B": 0.0}   # disagrees
        ex.append({
            "gpt": {"A": 0.0, "B": 1.0},
            "claude": {"A": 0.0, "B": 1.0},
            "gemini": gemini,
        })
    agreement = by_check(audit_judge_reliability(make_data(ex), "A", "B"),
                         "inter_judge_agreement")
    assert agreement.status is Status.WARN
    assert agreement.details["outlier_judge"] == "gemini"


def test_findings_obey_golden_rule():
    ex = [{"gpt": {"A": 0.0, "B": 1.0}, "claude": {"A": 0.0, "B": 1.0}}
          for _ in range(10)]
    for f in audit_judge_reliability(make_data(ex), "A", "B"):
        assert f.why.strip() and f.how_detected.strip() and f.how_to_fix.strip()
        assert f.pillar == "Judge Reliability"


def test_consensus_skips_judge_that_scored_only_one_model():
    # gpt scored both; claude scored only A — 1 surviving judge is not consensus
    ex = []
    for _ in range(10):
        ex.append({
            "gpt":    {"A": 0.0, "B": 1.0},
            "claude": {"A": 0.5},           # B is missing entirely
        })
    findings = audit_judge_reliability(make_data(ex), "A", "B")
    consensus = by_check(findings, "judge_consensus")
    # Must not crash, must not silently default via nan
    # With len(winners) < 2 guard, one surviving judge → SKIP
    assert consensus.status is Status.SKIP
    # claude should appear in the skipped note, not as a winner
    assert "claude" not in consensus.details["per_judge_winner"]
    assert "claude" in consensus.how_detected


def test_consensus_skips_when_all_judges_scored_only_one_model():
    # both judges only scored model A — winners dict will be empty
    ex = [{"gpt": {"A": 0.5}, "claude": {"A": 0.7}} for _ in range(10)]
    findings = audit_judge_reliability(make_data(ex), "A", "B")
    consensus = by_check(findings, "judge_consensus")
    assert consensus.status is Status.SKIP
    assert "gpt" in consensus.how_detected
    assert "claude" in consensus.how_detected


def test_consensus_all_skipped_details_includes_skipped_judges():
    # skipped_judges must appear in details for --json consumers
    ex = [{"gpt": {"A": 0.5}, "claude": {"A": 0.7}} for _ in range(10)]
    findings = audit_judge_reliability(make_data(ex), "A", "B")
    consensus = by_check(findings, "judge_consensus")
    assert "skipped_judges" in consensus.details
    assert set(consensus.details["skipped_judges"]) == {"gpt", "claude"}


def test_consensus_partial_skip_details_includes_skipped_judges():
    # gpt scored both; claude scored only A — partial skip
    # skipped_judges must appear in details of the normal Finding too
    ex = [{"gpt": {"A": 0.0, "B": 1.0}, "claude": {"A": 0.5}} for _ in range(10)]
    findings = audit_judge_reliability(make_data(ex), "A", "B")
    consensus = by_check(findings, "judge_consensus")
    assert "skipped_judges" in consensus.details
    assert consensus.details["skipped_judges"] == ["claude"]


def test_consensus_skips_when_only_one_judge_scored_both_models():
    # gpt scored both; claude scored only A — only 1 winner, not consensus
    ex = [{"gpt": {"A": 0.0, "B": 1.0}, "claude": {"A": 0.5}} for _ in range(10)]
    findings = audit_judge_reliability(make_data(ex), "A", "B")
    consensus = by_check(findings, "judge_consensus")
    assert consensus.status is Status.SKIP
    assert "gpt" in consensus.how_detected
    assert "claude" in consensus.how_detected
    assert consensus.details["skipped_judges"] == ["claude"]


# --- single-model mode (#21) -------------------------------------------------

def test_single_model_reliability_reports_agreement_and_no_consensus():
    # Two judges score one model across items. Inter-judge agreement is
    # assessable; consensus (who wins A vs B) is comparison-only, so it's absent.
    ex = [{"gpt": {"A": float(i % 2)}, "claude": {"A": float(i % 2)}}
          for i in range(10)]
    findings = audit_judge_reliability(make_data(ex), "A")   # model_b omitted
    checks = {f.details.get("check") for f in findings}
    assert "inter_judge_agreement" in checks
    assert "judge_consensus" not in checks


def test_single_model_reliability_skips_without_a_second_judge():
    ex = [{"gpt": {"A": float(i % 2)}} for i in range(10)]
    findings = audit_judge_reliability(make_data(ex), "A")
    assert all(f.status is Status.SKIP for f in findings)


def test_single_model_audit_runs_judge_checks_end_to_end():
    from evaltrust.audit.runner import run_audit
    exs = [Example(id=str(i), scores={"A": float(i % 2)},
                   judges={"gpt": {"A": float(i % 2)},
                           "claude": {"A": float(i % 2)}})
           for i in range(20)]
    data = EvalData(models=["A"], examples=exs, source_format="test", metadata={})
    report = run_audit(data)   # one model -> single path
    checks = {f.details.get("check") for f in report.findings}
    assert "inter_judge_agreement" in checks
