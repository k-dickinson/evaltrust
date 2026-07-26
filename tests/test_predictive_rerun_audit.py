"""Tests for predictive-rerun eligibility, findings, and runner wiring."""

from __future__ import annotations

import json

import numpy as np
import pytest

from evaltrust.audit import predictive_rerun as predictive_audit
from evaltrust.audit.predictive_rerun import audit_predictive_rerun
from evaltrust.audit.runner import run_audit
from evaltrust.config import AuditConfig
from evaltrust.core.schema import EvalData, Example, Status
from evaltrust.stats.predictive_rerun import predictive_rerun_normal_theory


DETAIL_KEYS = {
    "n_total",
    "n_predictive",
    "reason_counts",
    "future_runs",
    "method",
    "normal_theory_probability_a_better",
    "normal_theory_range_low",
    "normal_theory_range_high",
    "central_mass",
    "degenerate_reason",
    "point_estimate_b_minus_a",
    "prediction_variance",
    "degrees_of_freedom",
}


def _data(examples: list[Example]) -> EvalData:
    return EvalData(
        models=["A", "B"],
        examples=examples,
        source_format="test",
        metadata={},
    )


def _valid_example(
    example_id: str,
    runs_a: list[float],
    runs_b: list[float],
) -> Example:
    return Example(
        id=example_id,
        scores={
            "A": float(np.mean(runs_a)),
            "B": float(np.mean(runs_b)),
        },
        runs={"A": runs_a, "B": runs_b},
    )


def _finding(data: EvalData, *, future_runs: int = 3):
    findings = audit_predictive_rerun(
        data,
        "A",
        "B",
        future_runs=future_runs,
    )
    assert len(findings) == 1
    return findings[0]


def test_reason_taxonomy_is_first_match_and_never_double_counts():
    examples = [
        Example("missing-a", {"A": 0.0, "B": 0.0}, runs=None),
        Example(
            "missing-b-before-nonfinite-a",
            {"A": 0.0, "B": 0.0},
            runs={"A": [np.nan]},
        ),
        Example(
            "nonfinite-a-before-nonfinite-b",
            {"A": 0.0, "B": 0.0},
            runs={"A": [np.nan], "B": [np.inf]},
        ),
        Example(
            "nonfinite-b-before-disagreement",
            {"A": 99.0, "B": 0.0},
            runs={"A": [0.0, 1.0], "B": [np.inf, 0.0]},
        ),
        Example(
            "one-disagreement-reason",
            {"A": 99.0, "B": 99.0},
            runs={"A": [0.0, 1.0], "B": [0.0, 1.0]},
        ),
        _valid_example("valid", [0.0, 1.0], [0.1, 1.1]),
    ]

    finding = _finding(_data(examples))

    assert finding.details["reason_counts"] == {
        "missing_runs_a": 1,
        "missing_runs_b": 1,
        "nonfinite_a": 1,
        "nonfinite_b": 1,
        "score_run_mean_disagreement": 1,
        "singleton_in_mixed": 0,
    }
    assert finding.details["n_total"] == 6
    assert finding.details["n_predictive"] == 1
    assert sum(finding.details["reason_counts"].values()) + 1 == 6


def test_mixed_singleton_batch_is_demoted_before_primitive_call(monkeypatch):
    data = _data(
        [
            _valid_example("singleton", [0.0], [1.0]),
            _valid_example("identified", [0.0, 1.0], [0.2, 1.2]),
        ]
    )
    calls = []

    def reject_mixed(runs_a, runs_b, *, future_runs, central_mass=0.95):
        counts = [len(stream) for stream in (*runs_a, *runs_b)]
        assert all(count == 1 for count in counts) or all(
            count > 1 for count in counts
        )
        calls.append(counts)
        return predictive_rerun_normal_theory(
            runs_a,
            runs_b,
            future_runs=future_runs,
            central_mass=central_mass,
        )

    monkeypatch.setattr(
        predictive_audit,
        "predictive_rerun_normal_theory",
        reject_mixed,
    )

    finding = _finding(data)

    assert calls == [[2, 2]]
    assert finding.details["n_predictive"] == 1
    assert finding.details["reason_counts"]["singleton_in_mixed"] == 1


def test_all_singleton_batch_is_a_valid_degenerate_finding():
    finding = _finding(
        _data(
            [
                _valid_example("one", [0.7], [0.4]),
                _valid_example("two", [0.8], [0.6]),
            ]
        ),
        future_runs=4,
    )

    assert finding.status is Status.PASS
    assert finding.details["normal_theory_probability_a_better"] is None
    assert finding.details["prediction_variance"] is None
    assert finding.details["degrees_of_freedom"] is None
    assert finding.details["degenerate_reason"] == "single_run"
    assert finding.details["normal_theory_range_low"] == pytest.approx(-0.25)
    assert finding.details["normal_theory_range_high"] == pytest.approx(-0.25)
    assert "degenerate_reason=single_run" in finding.title
    assert "predictive range (normal-theory approximation)" in finding.title
    assert "n_predictive=2/2" in finding.title


def test_empty_predictive_set_returns_one_complete_skip_finding():
    finding = _finding(
        _data(
            [
                Example("one", {"A": 0.0, "B": 1.0}),
                Example("two", {"A": 0.0, "B": 1.0}, runs={"A": [0.0]}),
            ]
        )
    )

    assert finding.status is Status.SKIP
    assert finding.details["n_total"] == 2
    assert finding.details["n_predictive"] == 0
    assert DETAIL_KEYS <= finding.details.keys()
    assert finding.details["normal_theory_probability_a_better"] is None
    assert finding.details["normal_theory_range_low"] is None
    assert finding.details["normal_theory_range_high"] is None
    assert "n_predictive=0/2" in finding.title


def test_assessed_title_and_details_expose_probability_counts_and_model_limit():
    finding = _finding(
        _data(
            [
                _valid_example("one", [0.6, 0.8, 1.0], [0.2, 0.5, 0.7]),
                _valid_example("two", [0.5, 0.7], [0.4, 0.6, 0.8]),
            ]
        ),
        future_runs=5,
    )

    assert finding.status is Status.PASS
    assert finding.title.startswith(
        "P(A better on a rerun of these examples at 5 future runs)"
    )
    assert "n_predictive=2/2" in finding.title
    assert "predictive range (normal-theory approximation)" in finding.title
    assert "A=A, B=B" in finding.title
    assert DETAIL_KEYS <= finding.details.keys()
    assert finding.details["normal_theory_probability_a_better"] is not None
    assert finding.details["degenerate_reason"] is None
    assert "~0.933" in finding.details["coverage_limitation"]
    assert "~0.937" in finding.details["coverage_limitation"]
    assert "~0.933" in finding.how_detected
    assert "~0.937" in finding.how_detected
    assert finding.why.strip()
    assert finding.how_to_fix.strip()


def test_predictive_finding_never_changes_a_low_verdict():
    examples = []
    for index in range(120):
        score_a = float(index % 2)
        score_b = float((index + 1) % 2)
        examples.append(
            Example(
                id=str(index),
                scores={"A": score_a, "B": score_b},
                runs={"A": [score_a], "B": [score_b]},
            )
        )
    data = _data(examples)

    off = run_audit(data, config=AuditConfig(n_resamples=99, seed=7))
    on = run_audit(
        data,
        config=AuditConfig(
            n_resamples=99,
            seed=7,
            run_aware=True,
            run_aware_future_runs=3,
        ),
    )

    assert off.verdict == on.verdict
    assert off.verdict.drivers == on.verdict.drivers
    assert any(
        finding.status is Status.FAIL for finding in off.findings
    )
    predictive = next(
        finding
        for finding in on.findings
        if finding.details.get("check") == "predictive_rerun"
    )
    assert predictive.status in {Status.PASS, Status.SKIP}


def test_run_aware_is_additive_and_repeatability_is_identical():
    data = _data(
        [
            _valid_example(str(index), [0.0, 0.2], [0.8, 1.0])
            for index in range(40)
        ]
    )
    off = run_audit(data, config=AuditConfig(n_resamples=99, seed=7))
    on = run_audit(
        data,
        config=AuditConfig(
            n_resamples=99,
            seed=7,
            run_aware=True,
            run_aware_future_runs=3,
        ),
    )

    off_repeatability = [
        finding.to_dict()
        for finding in off.findings
        if finding.pillar == "Repeatability"
    ]
    on_repeatability = [
        finding.to_dict()
        for finding in on.findings
        if finding.pillar == "Repeatability"
    ]
    assert off_repeatability == on_repeatability

    on_payload = on.to_dict()
    on_payload["findings"] = [
        finding
        for finding in on_payload["findings"]
        if finding["details"].get("check") != "predictive_rerun"
    ]
    off_bytes = json.dumps(
        off.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    on_bytes = json.dumps(
        on_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    assert off_bytes == on_bytes
