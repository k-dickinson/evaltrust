"""Tests for the Judge Reliability audit (uses multi-judge data when present)."""

from evallab.audit.judge_reliability import audit_judge_reliability
from evallab.core.schema import EvalData, Example, Status


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
