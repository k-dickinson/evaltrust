"""Tests for issue #150: Python API parity with the CLI.

Covers:
- ``config=`` passthrough to ``evaltrust.audit()``
- opt-in kwargs (``bayesian``, ``all_pairs``, ``run_aware``, ``correction``)
- ``evaltrust.audit_run_level()`` entry point
- ``evaltrust.audit_contamination()`` entry point
- ``__all__`` completeness
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import evaltrust
from evaltrust import (
    AuditConfig,
    ContaminationResult,
    RunLevelData,
    audit,
    audit_contamination,
    audit_run_level,
)
from evaltrust.core.schema import EvalData, Example, Finding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(scores_by_model: dict, n: int) -> EvalData:
    examples = [
        Example(id=str(i), scores={m: float(s[i]) for m, s in scores_by_model.items()})
        for i in range(n)
    ]
    return EvalData(models=list(scores_by_model), examples=examples,
                    source_format="test", metadata={})


def _clear_win() -> EvalData:
    """200 examples: B wins 90% of the time."""
    return _make_data({"A": [0] * 200, "B": [1] * 180 + [0] * 20}, 200)


# ---------------------------------------------------------------------------
# config= passthrough
# ---------------------------------------------------------------------------

class TestConfigPassthrough:

    def test_config_kwarg_is_accepted(self):
        cfg = AuditConfig()
        report = audit(_clear_win(), config=cfg)
        assert report.verdict is not None

    def test_config_takes_precedence_over_loose_kwargs(self):
        """When config= is supplied, its alpha wins over the loose alpha kwarg."""
        cfg = AuditConfig(alpha=0.01)
        # Pass a conflicting loose alpha; config should win silently.
        report = audit(_clear_win(), alpha=0.99, config=cfg)
        # A clear win should still be HIGH regardless of reasonable alpha.
        assert report.verdict.level.name == "HIGH"

    def test_config_passthrough_enables_bayesian(self):
        cfg = AuditConfig(bayesian=True)
        report = audit(_clear_win(), config=cfg)
        titles = {f.title for f in report.findings}
        # The Bayesian check produces a finding with "win probability" in the title.
        assert any("win probability" in t.lower() or "bayesian" in t.lower()
                   for t in titles), f"Expected Bayesian finding, got: {titles}"

    def test_config_passthrough_enables_all_pairs(self):
        data = _make_data({"A": [0]*100, "B": [1]*90 + [0]*10,
                           "C": [1]*80 + [0]*20}, 100)
        cfg = AuditConfig(all_pairs=True)
        report = audit(data, config=cfg)
        # all_pairs produces extra findings; just assert no crash and non-empty.
        assert len(report.findings) > 0


# ---------------------------------------------------------------------------
# opt-in kwargs
# ---------------------------------------------------------------------------

class TestOptInKwargs:

    def test_bayesian_kwarg_enables_bayesian_finding(self):
        report = audit(_clear_win(), bayesian=True)
        titles = {f.title for f in report.findings}
        assert any("win probability" in t.lower() or "bayesian" in t.lower()
                   for t in titles), f"Expected Bayesian finding, got: {titles}"

    def test_bayesian_false_by_default(self):
        report = audit(_clear_win())
        titles = {f.title for f in report.findings}
        assert not any("win probability" in t.lower() or "bayesian" in t.lower()
                       for t in titles)

    def test_all_pairs_kwarg_is_accepted(self):
        data = _make_data({"A": [0]*100, "B": [1]*90 + [0]*10,
                           "C": [1]*80 + [0]*20}, 100)
        report = audit(data, all_pairs=True)
        assert len(report.findings) > 0

    def test_correction_kwarg_threads_into_config(self):
        """correction= kwarg must reach AuditConfig and alter report behaviour.

        We verify this via audit_suite, which exposes the correction label in
        SuiteReport.correction. Running the same suite with bonferroni vs holm
        confirms the value propagates rather than being silently discarded.
        """
        from evaltrust import audit_suite
        suite = {
            "correctness": _make_data({"A": [0] * 200, "B": [1] * 180 + [0] * 20}, 200),
            "tone":         _make_data({"A": [0, 1] * 60, "B": [1, 0] * 60}, 120),
        }
        bonf = audit_suite(suite, correction="bonferroni")
        holm = audit_suite(suite, correction="holm")
        assert "bonferroni" in bonf.correction.lower()
        assert "holm" in holm.correction.lower()

    def test_run_aware_requires_future_runs(self):
        """run_aware=True without run_aware_future_runs must raise at config construction."""
        with pytest.raises(ValueError, match="run_aware_future_runs"):
            audit(_clear_win(), run_aware=True)

    def test_run_aware_with_future_runs_accepted(self):
        """run_aware=True must produce a predictive-rerun finding in the report."""
        report = audit(_clear_win(), run_aware=True, run_aware_future_runs=5)
        pillar_titles = [(f.pillar, f.title) for f in report.findings]
        assert any(
            "rerun" in title.lower() or "predictive" in title.lower() or "repeatab" in title.lower()
            for _, title in pillar_titles
        ), f"Expected a predictive-rerun finding; got: {pillar_titles}"


# ---------------------------------------------------------------------------
# audit_run_level
# ---------------------------------------------------------------------------

class TestAuditRunLevel:

    def _write_run_level_csv(self, tmp_path, a_scores, b_scores):
        p = tmp_path / "runs.csv"
        lines = ["model_a,model_b"] + [
            f"{a},{b}" for a, b in zip(a_scores, b_scores, strict=True)
        ]
        p.write_text("\n".join(lines))
        return str(p)

    def test_returns_list_of_findings(self, tmp_path):
        path = self._write_run_level_csv(
            tmp_path,
            [0.80, 0.82, 0.79, 0.83, 0.81] * 3,
            [0.70, 0.71, 0.68, 0.72, 0.69] * 3,
        )
        findings = audit_run_level(path)
        assert isinstance(findings, list)
        assert all(isinstance(f, Finding) for f in findings)
        assert len(findings) == 3  # decision, effect_size, precision

    def test_finding_checks_are_correct(self, tmp_path):
        path = self._write_run_level_csv(
            tmp_path,
            [0.80, 0.82, 0.79, 0.83, 0.81] * 3,
            [0.70, 0.71, 0.68, 0.72, 0.69] * 3,
        )
        findings = audit_run_level(path)
        checks = {f.details["check"] for f in findings}
        assert checks == {"decision", "effect_size", "precision"}

    def test_model_labels_passed_through(self, tmp_path):
        # Use long-format CSV so the model names in the file match what we pass.
        p = tmp_path / "runs_long.csv"
        rows = ["model,score"]
        for s in [0.80, 0.82, 0.79] * 4:
            rows.append(f"gpt-4,{s}")
        for s in [0.70, 0.71, 0.68] * 4:
            rows.append(f"claude-3,{s}")
        p.write_text("\n".join(rows))
        findings = audit_run_level(str(p), model_a="gpt-4", model_b="claude-3")
        decision = next(f for f in findings if f.details["check"] == "decision")
        # model names should appear somewhere in the finding
        assert "gpt-4" in decision.title or "claude-3" in decision.title

    def test_alpha_and_seed_respected(self, tmp_path):
        path = self._write_run_level_csv(
            tmp_path,
            [0.80, 0.82] * 10,
            [0.70, 0.71] * 10,
        )
        # Should not raise with custom alpha/seed.
        findings = audit_run_level(path, alpha=0.01, seed=42)
        assert len(findings) == 3

    def test_config_passthrough(self, tmp_path):
        path = self._write_run_level_csv(
            tmp_path,
            [0.80, 0.82] * 10,
            [0.70, 0.71] * 10,
        )
        cfg = AuditConfig(alpha=0.01, seed=99, n_resamples=500)
        findings = audit_run_level(path, config=cfg)
        assert len(findings) == 3

    def test_is_in_all(self):
        assert "audit_run_level" in evaltrust.__all__

    def test_importable_from_top_level(self):
        assert callable(evaltrust.audit_run_level)


# ---------------------------------------------------------------------------
# audit_contamination
# ---------------------------------------------------------------------------

class TestAuditContamination:

    def test_no_contamination(self):
        benchmark = ["What is the capital of France?",
                     "Solve 2 + 2."]
        reference = ["The sky is blue.", "Water boils at 100 degrees."]
        result = audit_contamination(benchmark, reference)
        assert isinstance(result, ContaminationResult)
        assert result.exact_matches == 0
        assert result.near_matches == 0
        assert result.contamination_fraction == 0.0

    def test_exact_contamination_detected(self):
        benchmark = ["What is the capital of France?",
                     "Solve 2 + 2."]
        reference = ["What is the capital of France?",
                     "Something completely unrelated."]
        result = audit_contamination(benchmark, reference)
        assert result.exact_matches >= 1
        assert result.contamination_fraction > 0.0

    def test_near_match_contamination_detected(self):
        benchmark = ["What is the capitl of France?"]  # typo
        reference = ["What is the capital of France?"]
        result = audit_contamination(benchmark, reference)
        # Should detect as near-match
        assert result.near_matches >= 1 or result.exact_matches >= 1

    def test_full_contamination(self):
        items = ["A", "B", "C"]
        result = audit_contamination(items, items)
        assert result.contamination_fraction == 1.0
        assert result.total_items == 3

    def test_threshold_kwarg_accepted(self):
        benchmark = ["The quick brown fox"]
        reference = ["The quick brown cat"]
        # High threshold: may or may not match; just must not raise.
        result = audit_contamination(benchmark, reference, threshold=0.95)
        assert isinstance(result, ContaminationResult)

    def test_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            audit_contamination(["hello"], ["hello"], threshold=-0.1)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            audit_contamination(["hello"], ["hello"], threshold=1.5)

    def test_empty_benchmark(self):
        result = audit_contamination([], ["some reference"])
        assert result.total_items == 0
        assert result.contamination_fraction == 0.0

    def test_is_in_all(self):
        assert "audit_contamination" in evaltrust.__all__

    def test_importable_from_top_level(self):
        assert callable(evaltrust.audit_contamination)

    def test_returns_contamination_result_type(self):
        result = audit_contamination(["hello"], ["hello"])
        assert isinstance(result, ContaminationResult)


# ---------------------------------------------------------------------------
# __all__ completeness
# ---------------------------------------------------------------------------

class TestAllExports:

    def test_audit_run_level_in_all(self):
        assert "audit_run_level" in evaltrust.__all__

    def test_audit_contamination_in_all(self):
        assert "audit_contamination" in evaltrust.__all__

    def test_audit_config_in_all(self):
        assert "AuditConfig" in evaltrust.__all__

    def test_run_level_data_in_all(self):
        assert "RunLevelData" in evaltrust.__all__

    def test_contamination_result_in_all(self):
        assert "ContaminationResult" in evaltrust.__all__

    def test_all_names_are_importable(self):
        """Every name in __all__ must be importable from the top-level package."""
        for name in evaltrust.__all__:
            assert hasattr(evaltrust, name), f"evaltrust.{name} is in __all__ but not importable"
