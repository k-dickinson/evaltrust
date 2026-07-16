"""Tests for the audit runner that wires every check together."""

import json

import pytest

from evaltrust.audit.runner import run_audit
from evaltrust.audit.verdict import VerdictLevel
from evaltrust.config import AuditConfig
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


def test_all_pairs_is_additive_and_default_output_is_byte_identical():
    data = make_data({
        "weak": [0] * 40,
        "best": [1] * 40,
        "mid": [1] * 20 + [0] * 20,
    }, 40)
    off = run_audit(data, config=AuditConfig(n_resamples=99, seed=7))
    on = run_audit(
        data,
        config=AuditConfig(all_pairs=True, n_resamples=99, seed=7),
    )

    assert not any(
        finding.details.get("check") == "all_pairs" for finding in off.findings)
    assert any(
        finding.details.get("check") == "all_pairs" for finding in on.findings)
    assert off.verdict.to_dict() == on.verdict.to_dict()

    off_payload = off.to_dict()
    on_payload = on.to_dict()
    on_payload["findings"] = [
        finding for finding in on_payload["findings"]
        if finding["details"].get("check") != "all_pairs"
    ]
    off_bytes = json.dumps(
        off_payload, sort_keys=True, separators=(",", ":")).encode()
    on_bytes = json.dumps(
        on_payload, sort_keys=True, separators=(",", ":")).encode()
    assert off_bytes == on_bytes


def test_explicit_primary_pair_stays_primary_with_all_pairs_enabled():
    data = make_data({
        "A": [0] * 30,
        "B": [1] * 30,
        "C": [1] * 15 + [0] * 15,
    }, 30)
    report = run_audit(
        data,
        model_a="A",
        model_b="C",
        config=AuditConfig(all_pairs=True, n_resamples=99),
    )
    finding = next(
        f for f in report.findings if f.details.get("check") == "all_pairs")

    assert (report.model_a, report.model_b) == ("A", "C")
    assert finding.details["n_pairs_total"] == 3


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


def test_missing_explicit_model_error_lists_available_models():
    data = make_data({"A": [0] * 30, "B": [1] * 30}, 30)

    with pytest.raises(ValueError, match="Available models: 'A', 'B'"):
        run_audit(data, model_a="typo", model_b="B")


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
