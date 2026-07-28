"""Tests for the optional all-example paired win-rate finding."""

from __future__ import annotations

import json

import numpy as np

from evaltrust.audit.runner import run_audit
from evaltrust.audit.win_rate import audit_win_rate
from evaltrust.config import AuditConfig
from evaltrust.core.schema import EvalData, Example, Preference, Status
from evaltrust.report.html import render_html
from evaltrust.report.terminal import render_markdown, render_plain, render_report


REQUIRED_EVENT_KEYS = {
    "paired_win_rate_a",
    "win_rate_interval_low",
    "win_rate_interval_high",
    "n_wins_a",
    "n_ties",
    "n_wins_b",
    "tie_policy",
    "estimand",
    "method",
}
COLLISION_KEYS = {"p_a_gt_b", "probability_a_better"}


def _data(differences, *, clusters=None):
    examples = []
    for index, difference in enumerate(differences):
        group_id = None if clusters is None else clusters[index]
        examples.append(
            Example(
                id=str(index),
                scores={"A": 0.0, "B": float(difference)},
                group_id=group_id,
            )
        )
    return EvalData(
        models=["A", "B"],
        examples=examples,
        source_format="test",
        metadata={},
    )


def _finding(report):
    return next(
        finding
        for finding in report.findings
        if finding.details.get("check") == "paired_win_rate"
    )


def test_orientation_uses_b_minus_a_and_is_hard_locked_both_ways():
    a_higher = _data([-1.0] * 8)
    b_higher = _data([1.0] * 8)

    assert np.all(a_higher.differences("A", "B") < 0.0)
    assert np.all(b_higher.differences("A", "B") > 0.0)
    assert audit_win_rate(a_higher, "A", "B")[0].details["paired_win_rate_a"] == 1.0
    assert audit_win_rate(b_higher, "A", "B")[0].details["paired_win_rate_a"] == 0.0


def test_assessed_title_names_event_counts_ties_and_bootstrap_interval():
    finding = audit_win_rate(_data([-1.0, -1.0, 0.0, 1.0]), "A", "B")[0]

    assert finding.status is Status.PASS
    assert "A scores higher than B on 62.5% of examples" in finding.title
    assert "ties half credit" in finding.title
    assert "4 examples" in finding.title
    assert "95% bootstrap interval" in finding.title
    assert REQUIRED_EVENT_KEYS <= finding.details.keys()
    assert not COLLISION_KEYS & finding.details.keys()
    assert finding.details["tie_policy"] == "half_credit"
    assert finding.details["estimand"] == "all_example_paired_win_rate_a"
    assert finding.details["method"] == "half-tie-percentile-bootstrap-v1"
    assert (
        finding.details["n_wins_a"],
        finding.details["n_ties"],
        finding.details["n_wins_b"],
    ) == (2, 1, 1)


def test_title_is_visible_in_every_human_report_surface():
    report = run_audit(
        _data([-1.0] * 20),
        model_a="A",
        model_b="B",
        config=AuditConfig(win_rate=True),
    )
    title = _finding(report).title

    for rendered in (
        render_report(report, width=240),
        render_plain(report),
        render_markdown(report),
        render_html(report),
    ):
        assert " ".join(title.split()) in " ".join(rendered.split())


def test_preference_only_returns_counted_explained_skip():
    data = EvalData(
        models=["A", "B"],
        examples=[
            Example(
                id="1",
                scores={},
                preferences={"judge": Preference.TIE},
            )
        ],
        source_format="test",
    )

    report = run_audit(
        data,
        model_a="A",
        model_b="B",
        config=AuditConfig(win_rate=True),
    )
    finding = _finding(report)

    assert finding.status is Status.SKIP
    assert finding.details["assessed"] is False
    assert finding.details["reason"] == "preference_only"
    assert finding.details["n_examples"] == 0
    assert "0 examples" in finding.title
    assert "preference-only" in finding.how_detected.lower()
    assert finding.why.strip()
    assert finding.how_to_fix.strip()


def test_zero_paired_scores_returns_skip_instead_of_crashing():
    data = EvalData(
        models=["A", "B"],
        examples=[
            Example(id="a", scores={"A": 1.0}),
            Example(id="b", scores={"B": 1.0}),
        ],
        source_format="test",
    )

    finding = audit_win_rate(data, "A", "B")[0]

    assert finding.status is Status.SKIP
    assert finding.details["reason"] == "no_paired_scores"
    assert finding.details["n_examples"] == 0
    assert "bootstrap interval unavailable" in finding.title


def test_nonfinite_scores_skip_instead_of_crashing():
    finding = audit_win_rate(_data([-1.0, np.nan]), "A", "B")[0]

    assert finding.status is Status.SKIP
    assert finding.details["assessed"] is False
    assert finding.details["reason"] == "calculation_error"
    assert "calculation_error" in finding.details


def test_unhashable_group_id_skips_instead_of_crashing():
    data = _data(
        [-1.0, 1.0],
        clusters=[["malformed"], "valid"],  # type: ignore[list-item]
    )

    finding = audit_win_rate(data, "A", "B")[0]

    assert finding.status is Status.SKIP
    assert finding.details["assessed"] is False
    assert finding.details["reason"] == "calculation_error"
    assert finding.details["clustered"] is True
    assert "hashable" in finding.details["calculation_error"]


def test_group_ids_enable_cluster_bootstrap_without_changing_point_estimate():
    differences = [-1.0] * 8 + [1.0] * 2
    clustered = audit_win_rate(
        _data(differences, clusters=["majority"] * 8 + ["minority"] * 2),
        "A",
        "B",
        seed=8,
    )[0]
    plain = audit_win_rate(_data(differences), "A", "B", seed=8)[0]

    assert clustered.details["clustered"] is True
    assert clustered.details["n_resampling_units"] == 2
    assert plain.details["clustered"] is False
    assert clustered.details["paired_win_rate_a"] == plain.details["paired_win_rate_a"]
    assert (
        clustered.details["win_rate_interval_low"]
        < plain.details["win_rate_interval_low"]
    )


def test_optional_finding_is_additive_and_off_mode_is_identical():
    data = _data([-1.0] * 20)
    default = run_audit(data, model_a="A", model_b="B")
    explicit_off = run_audit(
        data,
        model_a="A",
        model_b="B",
        config=AuditConfig(win_rate=False),
    )
    enabled = run_audit(
        data,
        model_a="A",
        model_b="B",
        config=AuditConfig(win_rate=True),
    )

    assert default.to_dict() == explicit_off.to_dict()
    enabled_payload = enabled.to_dict()
    enabled_payload["findings"] = [
        finding
        for finding in enabled_payload["findings"]
        if finding["details"].get("check") != "paired_win_rate"
    ]
    off_bytes = json.dumps(
        default.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    on_minus_new_bytes = json.dumps(
        enabled_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    assert on_minus_new_bytes == off_bytes


def test_optional_finding_never_changes_an_existing_fail_verdict():
    data = _data([-1.0, 1.0] * 60)
    off = run_audit(
        data,
        model_a="A",
        model_b="B",
        config=AuditConfig(win_rate=False),
    )
    on = run_audit(
        data,
        model_a="A",
        model_b="B",
        config=AuditConfig(win_rate=True),
    )

    assert any(finding.status is Status.FAIL for finding in off.findings)
    assert off.verdict == on.verdict
    assert off.verdict.drivers == on.verdict.drivers
    assert _finding(on).status is Status.PASS


def test_pass_and_skip_findings_follow_the_golden_rule():
    passed = audit_win_rate(_data([-1.0]), "A", "B")[0]
    skipped = audit_win_rate(
        EvalData(models=["A", "B"], examples=[], source_format="test"),
        "A",
        "B",
    )[0]

    for finding in (passed, skipped):
        assert finding.why.strip()
        assert finding.how_detected.strip()
        assert finding.how_to_fix.strip()
