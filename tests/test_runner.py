"""Tests for the audit runner that wires every check together."""

from evaltrust.audit.runner import run_audit
from evaltrust.audit.verdict import VerdictLevel
from evaltrust.core.schema import EvalData, Example


def make_data(scores_by_model, n):
    """scores_by_model: {model: list-of-scores length n}."""
    examples = [
        Example(id=str(i), scores={m: float(s[i]) for m, s in scores_by_model.items()})
        for i in range(n)
    ]
    return EvalData(models=list(scores_by_model), examples=examples,
                    source_format="test", metadata={})


def test_report_covers_all_four_pillars():
    data = make_data({"A": [0] * 50, "B": [1] * 50}, 50)
    report = run_audit(data, seed=0)
    pillars = {f.pillar for f in report.findings}
    assert pillars == {
        "Statistical Validity", "Benchmark Health",
        "Repeatability", "Judge Reliability",
    }


def test_report_records_the_two_models_compared():
    data = make_data({"A": [0] * 20, "B": [1] * 20}, 20)
    report = run_audit(data)
    assert {report.model_a, report.model_b} == {"A", "B"}


def test_clean_win_yields_high_confidence():
    data = make_data({"A": [0] * 200, "B": [1] * 180 + [0] * 20}, 200)
    assert run_audit(data, seed=0).verdict.level is VerdictLevel.HIGH


def test_pure_noise_yields_low_confidence():
    data = make_data({"A": [0, 1] * 60, "B": [1, 0] * 60}, 120)
    assert run_audit(data, seed=0).verdict.level is VerdictLevel.LOW


def test_more_than_two_models_compares_the_two_strongest():
    data = make_data({
        "weak": [0] * 100,
        "best": [1] * 100,
        "mid": [1] * 50 + [0] * 50,
    }, 100)
    report = run_audit(data)
    assert {report.model_a, report.model_b} == {"best", "mid"}


def test_report_records_all_available_models():
    data = make_data({"weak": [0] * 30, "best": [1] * 30, "mid": [1] * 15 + [0] * 15}, 30)
    report = run_audit(data)
    assert set(report.models_available) == {"weak", "best", "mid"}
    assert report.to_dict()["models_available"] == report.models_available


def test_two_model_file_reports_both_as_available():
    data = make_data({"A": [0] * 20, "B": [1] * 20}, 20)
    assert set(run_audit(data).models_available) == {"A", "B"}


def test_explicit_models_are_respected():
    data = make_data({"A": [0] * 30, "B": [1] * 30, "C": [1] * 30}, 30)
    report = run_audit(data, model_a="A", model_b="C")
    assert {report.model_a, report.model_b} == {"A", "C"}


def test_is_deterministic():
    data = make_data({"A": [0] * 80, "B": [1] * 70 + [0] * 10}, 80)
    r1 = run_audit(data, seed=3)
    r2 = run_audit(data, seed=3)
    assert [f.details for f in r1.findings] == [f.details for f in r2.findings]
    assert r1.verdict.level is r2.verdict.level


def _decision(report):
    return next(f for f in report.findings if f.details.get("check") == "decision")


def test_significant_override_forwards_to_the_statistical_decision():
    # run_audit exposes the keyword-only `significant` override and forwards it to
    # the two-model comparison's decision. Default (None) leaves the audit to
    # decide; a clear win is significant.
    data = make_data({"A": [0] * 200, "B": [1] * 180 + [0] * 20}, 200)
    assert _decision(run_audit(data, seed=0)).details["outcome"] == "significant"
    # Forcing significant=False overrides the decision despite the clear win.
    assert _decision(
        run_audit(data, seed=0, significant=False)).details["outcome"] != "significant"
