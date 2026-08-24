"""Tests for the pass@k audit check and its wiring (issue #164)."""

from __future__ import annotations

import numpy as np

from evaltrust.audit.pass_at_k import CHECK, audit_pass_at_k
from evaltrust.audit.runner import run_audit
from evaltrust.config import AuditConfig
from evaltrust.core.schema import EvalData, Example, Status


def _runs_data(tasks, k_samples=10):
    """Build EvalData where runs carry per-sample 0/1 outcomes.

    ``tasks`` is a list of ``(c_a, c_b)`` correct counts; each model gets
    ``k_samples`` runs per task with ``c`` ones and the rest zeros.
    """
    examples = []
    for i, (c_a, c_b) in enumerate(tasks):
        runs_a = [1.0] * c_a + [0.0] * (k_samples - c_a)
        runs_b = [1.0] * c_b + [0.0] * (k_samples - c_b)
        examples.append(
            Example(
                id=str(i),
                scores={"A": c_a / k_samples, "B": c_b / k_samples},
                runs={"A": runs_a, "B": runs_b},
            )
        )
    return EvalData(models=["A", "B"], examples=examples, source_format="test")


def _finding(findings):
    return next(f for f in findings if f.details.get("check") == CHECK)


class TestCheck:
    def test_emits_finding_with_estimates(self):
        data = _runs_data([(3, 6), (4, 7), (2, 5), (5, 8), (1, 4)])
        findings = audit_pass_at_k(data, "A", "B", k=2)
        f = _finding(findings)
        assert f.details["assessed"] is True
        assert f.details["k"] == 2
        assert f.details["n_tasks"] == 5
        assert f.details["pass_at_k_b"] > f.details["pass_at_k_a"]

    def test_clear_gap_passes(self):
        data = _runs_data([(1, 9)] * 8)
        f = _finding(audit_pass_at_k(data, "A", "B", k=2, n_resamples=4000))
        assert f.status is Status.PASS
        assert f.details["delta_low"] > 0.0

    def test_no_gap_warns(self):
        data = _runs_data([(5, 5)] * 8)
        f = _finding(audit_pass_at_k(data, "A", "B", k=2, n_resamples=4000))
        assert f.status is Status.WARN

    def test_skips_when_no_runs(self):
        examples = [Example(id=str(i), scores={"A": 0.5, "B": 0.6}) for i in range(4)]
        data = EvalData(models=["A", "B"], examples=examples, source_format="test")
        f = _finding(audit_pass_at_k(data, "A", "B", k=2))
        assert f.status is Status.SKIP
        assert f.details["assessed"] is False

    def test_skips_when_all_tasks_below_k(self):
        data = _runs_data([(1, 1), (0, 1)], k_samples=3)
        f = _finding(audit_pass_at_k(data, "A", "B", k=5))
        assert f.status is Status.SKIP
        assert f.details["n_skipped"] == 2

    def test_counts_skipped_below_k(self):
        # First two tasks have 10 runs, last has 3; at k=5 the last is skipped.
        data = EvalData(
            models=["A", "B"],
            examples=[
                Example(id="0", scores={"A": 0.3, "B": 0.6},
                        runs={"A": [1.0] * 3 + [0.0] * 7, "B": [1.0] * 6 + [0.0] * 4}),
                Example(id="1", scores={"A": 0.4, "B": 0.7},
                        runs={"A": [1.0] * 4 + [0.0] * 6, "B": [1.0] * 7 + [0.0] * 3}),
                Example(id="2", scores={"A": 0.3, "B": 0.6},
                        runs={"A": [1.0, 0.0, 0.0], "B": [1.0, 1.0, 0.0]}),
            ],
            source_format="test",
        )
        f = _finding(audit_pass_at_k(data, "A", "B", k=5))
        assert f.details["n_tasks"] == 2
        assert f.details["n_skipped"] == 1

    def test_nonbinary_runs_counted_as_fail_and_reported(self):
        data = EvalData(
            models=["A", "B"],
            examples=[
                Example(id="0", scores={"A": 0.5, "B": 0.5},
                        runs={"A": [1.0, 0.5, 1.0, 0.0], "B": [1.0, 1.0, 0.0, 0.0]}),
            ],
            source_format="test",
        )
        f = _finding(audit_pass_at_k(data, "A", "B", k=2))
        # A's 0.5 sample is treated as a fail: c_a = 2 (only the two 1.0s).
        assert f.details["n_nonbinary"] == 1


class TestWiring:
    def test_runner_activates_only_with_k_and_runs(self):
        data = _runs_data([(3, 6), (4, 7), (2, 5), (5, 8)])
        # No k -> no pass@k finding.
        report = run_audit(data, "A", "B", config=AuditConfig())
        assert all(f.details.get("check") != CHECK for f in report.findings)
        # k set -> pass@k finding present.
        report = run_audit(data, "A", "B", config=AuditConfig(k=2))
        assert any(f.details.get("check") == CHECK for f in report.findings)

    def test_config_rejects_bad_k(self):
        for bad in (0, -1, True):
            try:
                AuditConfig(k=bad)
            except ValueError:
                continue
            raise AssertionError(f"AuditConfig(k={bad!r}) should have raised")

    def test_config_hashable_with_k(self):
        assert hash(AuditConfig(k=5)) != hash(AuditConfig(k=10))

    def test_api_audit_k_kwarg(self):
        import evaltrust

        data = _runs_data([(1, 9)] * 6)
        report = evaltrust.audit(data, model_a="A", model_b="B", k=2)
        assert any(f.details.get("check") == CHECK for f in report.findings)
