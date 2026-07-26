"""Audit-layer and runner integration tests for rank stability."""

from __future__ import annotations

import json

import numpy as np
import pytest

from evaltrust.audit.rank_stability import audit_rank_stability
from evaltrust.audit.runner import run_audit
from evaltrust.audit.suite import audit_suite
from evaltrust.config import AuditConfig
from evaltrust.core.schema import EvalData, Example, Status
from evaltrust.report.html import render_html
from evaltrust.report.terminal import render_markdown, render_plain, render_report


def make_data(scores_by_model: dict[str, list[float]], n: int | None = None):
    models = list(scores_by_model)
    n = n if n is not None else len(next(iter(scores_by_model.values())))
    examples = []
    for i in range(n):
        scores = {
            model: float(scores_by_model[model][i])
            for model in models
            if i < len(scores_by_model[model])
            and scores_by_model[model][i] is not None
        }
        examples.append(Example(id=str(i), scores=scores))
    return EvalData(models=models, examples=examples, source_format="test")


def by_check(findings, check):
    matches = [f for f in findings if f.details.get("check") == check]
    assert len(matches) == 1, f"expected one {check}, got {len(matches)}"
    return matches[0]


def test_strict_dominance_all_positions_stable():
    data = make_data({
        "A": [3.0] * 20,
        "B": [2.0] * 20,
        "C": [1.0] * 20,
    })
    finding = audit_rank_stability(
        data, AuditConfig(all_pairs=True, n_resamples=200, seed=0)
    )[0]
    assert finding.status is Status.PASS
    assert finding.details["assessed"] is True
    assert finding.details["stable_positions"] == [1, 2, 3]
    assert finding.details["full_order_retention"] == 1.0
    assert finding.details["top1_retention"] == 1.0
    assert "3 of 3 positions stable" in finding.title
    assert "full order retained 100.0%" in finding.title


def test_identical_models_zero_stable_positions():
    data = make_data({
        "A": [1.0] * 15,
        "B": [1.0] * 15,
        "C": [1.0] * 15,
    })
    finding = audit_rank_stability(
        data, AuditConfig(alpha=0.05, n_resamples=300, seed=1)
    )[0]
    assert finding.status is Status.PASS
    assert finding.details["stable_positions"] == []
    assert finding.details["full_order_retention"] == 0.0
    assert finding.details["tie_resamples"] == 300
    for model, row in finding.details["rank_occupancy"].items():
        assert row == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_two_models_skip():
    data = make_data({"A": [1.0] * 10, "B": [0.0] * 10})
    finding = audit_rank_stability(data, AuditConfig())[0]
    assert finding.status is Status.SKIP
    assert finding.details["reason"] == "fewer_than_three_models"
    assert finding.details["assessed"] is False


def test_zero_score_model_skips():
    data = EvalData(
        models=["A", "B", "C"],
        examples=[
            Example(id="0", scores={"A": 1.0, "B": 0.5}),
            Example(id="1", scores={"A": 1.0, "B": 0.4}),
        ],
        source_format="test",
    )
    finding = audit_rank_stability(data, AuditConfig())[0]
    assert finding.status is Status.SKIP
    assert finding.details["reason"] == "zero_score_model"
    assert finding.details["score_counts"]["C"] == 0


def test_preference_only_skips():
    data = EvalData(
        models=["A", "B", "C"],
        examples=[
            Example(
                id="0",
                scores={},
                preferences={"j": "A"},
            ),
        ],
        source_format="test",
    )
    finding = audit_rank_stability(data, AuditConfig())[0]
    assert finding.status is Status.SKIP
    assert finding.details["reason"] == "preference_only"


def test_single_cluster_skips():
    examples = [
        Example(id=str(i), scores={"A": 3.0, "B": 2.0, "C": 1.0}, group_id="only")
        for i in range(10)
    ]
    data = EvalData(models=["A", "B", "C"], examples=examples, source_format="test")
    finding = audit_rank_stability(data, AuditConfig(n_resamples=50))[0]
    assert finding.status is Status.SKIP
    assert finding.details["reason"] == "fewer_than_two_units"
    assert finding.details["clustered"] is True


def test_clustered_path_sets_details_and_differs_from_example_level():
    examples = []
    for i in range(8):
        examples.append(Example(
            id=f"a{i}", scores={"A": 3.0, "B": 2.0, "C": 0.0}, group_id="g0",
        ))
    for i in range(2):
        examples.append(Example(
            id=f"b{i}", scores={"A": 0.0, "B": 0.5, "C": 3.0}, group_id="g1",
        ))
    data = EvalData(models=["A", "B", "C"], examples=examples, source_format="test")
    clustered = audit_rank_stability(
        data, AuditConfig(n_resamples=1_000, seed=3)
    )[0]
    unclustered_data = EvalData(
        models=["A", "B", "C"],
        examples=[
            Example(id=ex.id, scores=dict(ex.scores)) for ex in examples
        ],
        source_format="test",
    )
    plain = audit_rank_stability(
        unclustered_data, AuditConfig(n_resamples=1_000, seed=3)
    )[0]
    assert clustered.details["clustered"] is True
    assert plain.details["clustered"] is False
    assert plain.details["full_order_retention"] > 0.5
    assert clustered.details["full_order_retention"] < plain.details[
        "full_order_retention"
    ]


def test_never_warn_or_fail_on_instability():
    data = make_data({
        "A": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "B": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "C": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    })
    finding = audit_rank_stability(
        data, AuditConfig(n_resamples=500, seed=0)
    )[0]
    assert finding.status is Status.PASS
    assert finding.status not in (Status.WARN, Status.FAIL)


def test_skip_also_never_warn_or_fail():
    finding = audit_rank_stability(
        make_data({"A": [1.0], "B": [0.0]}), AuditConfig()
    )[0]
    assert finding.status is Status.SKIP


def test_determinism_of_details():
    data = make_data({
        "A": [3, 3, 2, 2, 1],
        "B": [2, 2, 2, 1, 1],
        "C": [1, 1, 1, 1, 0],
    })
    cfg = AuditConfig(n_resamples=400, seed=17)
    a = audit_rank_stability(data, cfg)[0].details
    b = audit_rank_stability(data, cfg)[0].details
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_score_counts_rendered_and_missingness_ok():
    data = EvalData(
        models=["A", "B", "C"],
        examples=[
            Example(id="0", scores={"A": 1.0, "B": 0.5, "C": 0.0}),
            Example(id="1", scores={"A": 1.0, "B": 0.4}),  # C missing
            Example(id="2", scores={"A": 0.8, "B": 0.6, "C": 0.2}),
            Example(id="3", scores={"A": 0.9, "B": 0.5}),  # C missing
        ],
        source_format="test",
    )
    finding = audit_rank_stability(
        data, AuditConfig(n_resamples=200, seed=0)
    )[0]
    assert finding.status is Status.PASS
    assert finding.details["score_counts"] == {"A": 4, "B": 4, "C": 2}


def test_default_off_byte_identity_and_verdict_neutrality():
    data = make_data({
        "weak": [0] * 40,
        "best": [1] * 40,
        "mid": [1] * 20 + [0] * 20,
    })
    off = run_audit(data, config=AuditConfig(n_resamples=99, seed=7))
    on = run_audit(
        data,
        config=AuditConfig(all_pairs=True, n_resamples=99, seed=7),
    )

    assert not any(
        f.details.get("check") == "rank_stability" for f in off.findings
    )
    assert any(
        f.details.get("check") == "rank_stability" for f in on.findings
    )
    assert off.verdict.to_dict() == on.verdict.to_dict()

    # Strip both all_pairs and rank_stability findings; remainder matches off.
    off_payload = off.to_dict()
    on_payload = on.to_dict()
    on_payload["findings"] = [
        f for f in on_payload["findings"]
        if f["details"].get("check") not in {"all_pairs", "rank_stability"}
    ]
    off_bytes = json.dumps(
        off_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    on_bytes = json.dumps(
        on_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    assert off_bytes == on_bytes


def test_flag_on_under_three_models_rank_stability_skips_without_changing_pairs():
    data = make_data({"A": [0] * 20, "B": [1] * 20})
    off = run_audit(data, config=AuditConfig(n_resamples=50, seed=2))
    on = run_audit(
        data, config=AuditConfig(all_pairs=True, n_resamples=50, seed=2)
    )
    rs = by_check(on.findings, "rank_stability")
    assert rs.status is Status.SKIP
    pairs = by_check(on.findings, "all_pairs")
    assert pairs.status is Status.PASS
    # Verdict unchanged.
    assert off.verdict.to_dict() == on.verdict.to_dict()

    # The all_pairs finding bytes must match a path that only runs all_pairs
    # (no mutation of shared state by the rank-stability SKIP).
    from evaltrust.audit.allpairs import audit_all_pairs
    isolated = audit_all_pairs(
        data, AuditConfig(all_pairs=True, n_resamples=50, seed=2)
    )[0]
    assert json.dumps(pairs.to_dict(), sort_keys=True) == json.dumps(
        isolated.to_dict(), sort_keys=True
    )


def test_title_renders_in_every_human_format():
    data = make_data({
        "A": [3.0] * 15,
        "B": [2.0] * 15,
        "C": [1.0] * 15,
    })
    report = run_audit(
        data, config=AuditConfig(all_pairs=True, n_resamples=100, seed=0)
    )
    title = by_check(report.findings, "rank_stability").title
    assert "Rank stability:" in title
    for rendered in (
        render_report(report, width=200),
        render_plain(report),
        render_markdown(report),
        render_html(report),
    ):
        # Plain/md/html keep the full title; rich may wrap long lines so
        # require the stable lead fragment in every format.
        assert "Rank stability:" in rendered
        assert "positions stable" in rendered


def test_suite_suppresses_rank_stability_via_all_pairs_flag():
    data = make_data({
        "A": [3.0] * 12,
        "B": [2.0] * 12,
        "C": [1.0] * 12,
    })
    suite = audit_suite(
        {"m1": data, "m2": data},
        config=AuditConfig(all_pairs=True, n_resamples=50, seed=0),
    )
    for report in suite.reports.values():
        assert not any(
            f.details.get("check") in {"all_pairs", "rank_stability"}
            for f in report.findings
        )


def test_golden_rule_fields_present():
    passed = audit_rank_stability(
        make_data({"A": [3] * 10, "B": [2] * 10, "C": [1] * 10}),
        AuditConfig(n_resamples=50),
    )[0]
    skipped = audit_rank_stability(
        make_data({"A": [1], "B": [0]}), AuditConfig()
    )[0]
    for finding in (passed, skipped):
        assert finding.why.strip()
        assert finding.how_detected.strip()
        assert finding.how_to_fix.strip()
    assert "bootstrap" in passed.how_detected.lower()
    assert "tied" in passed.how_to_fix.lower() or "unstable" in passed.how_to_fix.lower()


def test_details_are_json_scalars():
    finding = audit_rank_stability(
        make_data({"A": [3] * 8, "B": [2] * 8, "C": [1] * 8}),
        AuditConfig(n_resamples=80, seed=0),
    )[0]
    payload = json.dumps(finding.to_dict())
    reloaded = json.loads(payload)
    assert reloaded["details"]["assessed"] is True
    assert isinstance(reloaded["details"]["full_order_retention"], float)
    assert isinstance(reloaded["details"]["stable_positions"], list)
