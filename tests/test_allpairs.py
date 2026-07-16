"""Tests for the optional all-pairs model comparison."""

from __future__ import annotations

import importlib
import json

import pytest
from statsmodels.stats.multitest import multipletests

from evaltrust.audit.allpairs import _pair_pvalue, audit_all_pairs
from evaltrust.audit.runner import run_audit
from evaltrust.audit.statistical import audit_statistical_validity
from evaltrust.audit.suite import audit_suite
from evaltrust.config import AuditConfig
from evaltrust.core.schema import EvalData, Example, Status
from evaltrust.report.html import render_html
from evaltrust.report.terminal import render_markdown, render_plain, render_report


def make_data(rows, models=None):
    examples = [
        Example(
            id=str(i),
            scores={model: float(score) for model, score in row.items()
                    if score is not None},
        )
        for i, row in enumerate(rows)
    ]
    if models is None:
        models = list(dict.fromkeys(model for row in rows for model in row))
    return EvalData(models=list(models), examples=examples, source_format="test")


def all_pairs_finding(findings):
    return next(f for f in findings if f.details.get("check") == "all_pairs")


def assessed_pairs(finding):
    return [pair for pair in finding.details["pairs"] if pair["assessed"]]


def test_pair_pvalue_matches_the_existing_statistical_path():
    cases = [
        make_data([{"A": 0, "B": 1}] * 10 + [{"A": 1, "B": 0}] * 2),
        make_data([
            {"A": 0.10, "B": 0.42},
            {"A": 0.35, "B": 0.61},
            {"A": 0.70, "B": 0.63},
            {"A": 0.20, "B": 0.55},
            {"A": 0.45, "B": 0.80},
        ]),
    ]
    for data in cases:
        findings = audit_statistical_validity(
            data, "A", "B", n_resamples=199, seed=7)
        decision = next(
            f for f in findings if f.details.get("check") == "decision")
        assert _pair_pvalue(
            data, "A", "B", n_resamples=199, seed=7
        ) == decision.details["p_value"]


def test_bonferroni_and_holm_match_statsmodels_reference():
    rows = (
        [{"A": 0, "B": 1, "C": 0}] * 10
        + [{"A": 0, "B": 0, "C": 1}] * 2
        + [{"A": 0, "B": 1, "C": 1}] * 20
    )
    data = make_data(rows, models=["A", "B", "C"])
    raw = audit_all_pairs(
        data, AuditConfig(correction="none", n_resamples=99, seed=3))[0]
    raw_pairs = assessed_pairs(raw)
    raw_p = [pair["p_value"] for pair in raw_pairs]

    assert [(pair["model_a"], pair["model_b"]) for pair in raw_pairs] == [
        ("B", "C"), ("B", "A"), ("C", "A")]

    for method in ("bonferroni", "holm"):
        finding = audit_all_pairs(
            data, AuditConfig(correction=method, n_resamples=99, seed=3))[0]
        pairs = assessed_pairs(finding)
        ref_reject, ref_adjusted, _, _ = multipletests(
            raw_p, alpha=0.05, method=method)

        assert finding.details["k"] == 3
        assert finding.details["n_pairs_total"] == 3
        assert [pair["adjusted_p"] for pair in pairs] == pytest.approx(
            ref_adjusted.tolist())
        assert [pair["reject"] for pair in pairs] == ref_reject.tolist()

        if method == "bonferroni":
            for pair in pairs:
                assert pair["adjusted_p"] == min(3 * pair["p_value"], 1.0)
                assert pair["reject"] is (pair["p_value"] < 0.05 / 3)


def test_holm_handles_tied_pvalues_and_reports_the_result_in_the_title():
    rows = [
        {"leader": 1, "near_a": i % 2, "near_b": i % 2}
        for i in range(40)
    ]
    finding = audit_all_pairs(
        make_data(rows), AuditConfig(correction="holm", n_resamples=99))[0]
    pairs = assessed_pairs(finding)
    raw_p = [pair["p_value"] for pair in pairs]
    ref_reject, ref_adjusted, _, _ = multipletests(
        raw_p, alpha=0.05, method="holm")

    assert raw_p[0] == raw_p[1]
    assert [pair["adjusted_p"] for pair in pairs] == pytest.approx(
        ref_adjusted.tolist())
    assert [pair["reject"] for pair in pairs] == ref_reject.tolist()
    assert finding.status is Status.PASS
    assert "2 of 3 pairs separable" in finding.title


def test_all_identical_models_report_nothing_separable():
    data = make_data([{"A": 1, "B": 1, "C": 1}] * 20)
    finding = audit_all_pairs(data, AuditConfig(n_resamples=99))[0]
    pairs = assessed_pairs(finding)

    assert finding.status is Status.PASS
    assert finding.details["k"] == 3
    assert finding.details["n_separable"] == 0
    assert all(pair["p_value"] == 1.0 for pair in pairs)
    assert all(pair["adjusted_p"] == 1.0 for pair in pairs)
    assert not any(pair["reject"] for pair in pairs)
    assert "0 of 3 pairs separable" in finding.title


def test_two_models_are_an_uncorrected_single_pair_family():
    data = make_data([{"A": 0, "B": 1}] * 20)
    finding = audit_all_pairs(data, AuditConfig(n_resamples=99))[0]
    (pair,) = assessed_pairs(finding)

    assert finding.status is Status.PASS
    assert finding.details["k"] == 1
    assert finding.details["n_pairs_total"] == 1
    assert pair["adjusted_p"] == pair["p_value"]


def test_family_counts_only_tested_pairs_and_reports_skipped_pairs():
    rows = ([{"A": 0.1, "B": 0.9}] * 20 + [{"C": 0.5}] * 20)
    finding = audit_all_pairs(
        make_data(rows, models=["C", "B", "A"]),
        AuditConfig(n_resamples=99, seed=5),
    )[0]
    tested = assessed_pairs(finding)
    skipped = [pair for pair in finding.details["pairs"] if not pair["assessed"]]

    assert finding.details["n_pairs_total"] == 3
    assert finding.details["k"] == 1
    assert len(tested) == 1
    assert len(skipped) == 2
    assert all(pair["reason"] == "no_paired_scores" for pair in skipped)
    assert tested[0]["adjusted_p"] == tested[0]["p_value"]


def test_declared_model_with_no_scores_is_reported_without_entering_the_family():
    data = make_data([{"A": 0, "B": 1}] * 20, models=["A", "empty", "B"])
    finding = audit_all_pairs(data, AuditConfig(n_resamples=99))[0]

    skipped = [pair for pair in finding.details["pairs"] if not pair["assessed"]]
    assert finding.details["n_pairs_total"] == 3
    assert finding.details["k"] == 1
    assert len(skipped) == 2
    assert all("empty" in (pair["model_a"], pair["model_b"]) for pair in skipped)


def test_no_testable_pair_returns_skip():
    rows = [{"A": 1}, {"B": 1}, {"C": 1}] * 4
    finding = audit_all_pairs(
        make_data(rows, models=["A", "B", "C"]), AuditConfig(n_resamples=99))[0]

    assert finding.status is Status.SKIP
    assert finding.details["assessed"] is False
    assert finding.details["k"] == 0
    assert finding.details["n_pairs_total"] == 3
    assert all(not pair["assessed"] for pair in finding.details["pairs"])


def test_pair_order_and_permutation_seeds_are_deterministic(monkeypatch):
    rows = [
        {"zeta": 0.4 + i / 1000, "leader": 0.9 + i / 1000,
         "alpha": 0.4 + i / 1000}
        for i in range(10)
    ]
    data = make_data(rows, models=["zeta", "leader", "alpha"])
    calls = []

    def fake_pvalue(data, model_a, model_b, n_resamples, seed):
        calls.append((model_a, model_b, n_resamples, seed))
        return 0.01 * (seed - 10)

    module = importlib.import_module("evaltrust.audit.allpairs")
    monkeypatch.setattr(module, "_pair_pvalue", fake_pvalue)
    finding = audit_all_pairs(
        data, AuditConfig(n_resamples=17, seed=11, correction="holm"))[0]

    expected_pairs = [
        ("leader", "alpha"),
        ("leader", "zeta"),
        ("alpha", "zeta"),
    ]
    assert [(a, b) for a, b, _, _ in calls] == expected_pairs
    assert [seed for _, _, _, seed in calls] == [11, 12, 13]
    assert [(pair["model_a"], pair["model_b"])
            for pair in finding.details["pairs"]] == expected_pairs


def test_seeded_all_pairs_run_is_deterministic():
    rows = [
        {"A": i / 50, "B": i / 50 + 0.2, "C": i / 50 + (i % 3) / 20}
        for i in range(20)
    ]
    cfg = AuditConfig(n_resamples=99, seed=13, correction="holm")
    first = audit_all_pairs(make_data(rows), cfg)[0]
    second = audit_all_pairs(make_data(rows), cfg)[0]
    assert first.to_dict() == second.to_dict()


def test_invalid_correction_raises_on_the_opted_in_path():
    data = make_data([{"A": 0, "B": 1, "C": 0}] * 10)
    with pytest.raises(ValueError, match="correction must be one of"):
        audit_all_pairs(data, AuditConfig(correction="bogus", n_resamples=99))


def test_details_are_plain_json_scalars():
    data = make_data([{"A": 0, "B": 1, "C": 0}] * 20)
    payload = audit_all_pairs(data, AuditConfig(n_resamples=99))[0].to_dict()
    json.dumps(payload, allow_nan=False)
    details = payload["details"]
    pair = next(pair for pair in details["pairs"] if pair["assessed"])

    assert type(details["k"]) is int
    assert type(details["n_pairs_total"]) is int
    assert type(details["alpha"]) is float
    assert type(pair["p_value"]) is float
    assert type(pair["adjusted_p"]) is float
    assert type(pair["reject"]) is bool
    assert type(pair["assessed"]) is bool


def test_result_title_renders_in_every_human_format():
    data = make_data([{"A": 0, "B": 1, "C": 0}] * 20)
    report = run_audit(
        data, config=AuditConfig(all_pairs=True, n_resamples=99))
    title = all_pairs_finding(report.findings).title

    assert title in render_report(report, width=200)
    assert title in render_plain(report)
    assert title in render_markdown(report)
    assert title in render_html(report)


def test_suite_does_not_expand_into_a_pair_by_metric_grid():
    data = make_data([{"A": 0, "B": 1, "C": 0}] * 20)
    report = audit_suite(
        {"accuracy": data, "quality": data},
        config=AuditConfig(all_pairs=True, n_resamples=99),
    )

    for metric_report in report.reports.values():
        assert not any(
            finding.details.get("check") == "all_pairs"
            for finding in metric_report.findings
        )
