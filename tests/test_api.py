"""Tests for the public Python API and machine-readable serialization.

Industry teams need to call EvalTrust from code and pipe its output into CI and
experiment trackers, not just read a terminal box. This is that surface.
"""

import json

import evaltrust
from evaltrust import audit, audit_suite
from evaltrust.audit.runner import AuditReport
from evaltrust.audit.suite import SuiteReport
from evaltrust.core.schema import EvalData, Example, Finding, Status


def test_package_exposes_a_version_string():
    assert isinstance(evaltrust.__version__, str)
    assert evaltrust.__version__


def make_data(scores_by_model, n):
    examples = [
        Example(id=str(i), scores={m: float(s[i]) for m, s in scores_by_model.items()})
        for i in range(n)
    ]
    return EvalData(models=list(scores_by_model), examples=examples,
                    source_format="test", metadata={})


def test_audit_accepts_an_evaldata_object():
    # 90% vs 0% — a clear win without saturating the benchmark.
    report = audit(make_data({"A": [0] * 200, "B": [1] * 180 + [0] * 20}, 200))
    assert isinstance(report, AuditReport)
    assert report.verdict.level.name == "HIGH"


def test_audit_accepts_a_file_path(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"models": ["A", "B"],
                             "examples": [{"id": "q1", "scores": {"A": 1, "B": 0}}]}))
    report = audit(str(p))
    assert {report.model_a, report.model_b} == {"A", "B"}


def test_audit_accepts_two_paths_for_single_model_files(tmp_path):
    for name, model, rate in [("a.json", "gpt", 0.6), ("b.json", "claude", 0.9)]:
        (tmp_path / name).write_text(json.dumps({"models": [model], "examples": [
            {"id": str(i), "scores": {model: 1 if i < int(100 * rate) else 0}}
            for i in range(100)]}))
    report = audit([str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert {report.model_a, report.model_b} == {"gpt", "claude"}


def test_finding_to_dict_has_golden_rule_fields():
    f = Finding(pillar="P", title="t", status=Status.WARN,
                why="w", how_detected="h", how_to_fix="x", details={"p_value": 0.2})
    d = f.to_dict()
    assert d["status"] == "WARN"
    assert d["pillar"] == "P" and d["why"] == "w"
    assert d["details"]["p_value"] == 0.2


def test_report_to_dict_is_json_serializable():
    report = audit(make_data({"A": [0] * 60, "B": [1] * 55 + [0] * 5}, 60))
    d = report.to_dict()
    text = json.dumps(d)  # must not raise
    round_tripped = json.loads(text)
    assert round_tripped["verdict"]["level"] in {"HIGH", "MODERATE", "LOW"}
    assert round_tripped["models"] == [report.model_a, report.model_b]
    assert len(round_tripped["findings"]) == len(report.findings)
    assert {f["pillar"] for f in round_tripped["findings"]} >= {"Statistical Validity"}


def test_verdict_to_dict_lists_driver_titles():
    report = audit(make_data({"A": [0, 1] * 60, "B": [1, 0] * 60}, 120))
    d = report.to_dict()
    assert d["verdict"]["level"] == "LOW"
    assert isinstance(d["verdict"]["drivers"], list)
    assert all(isinstance(x, str) for x in d["verdict"]["drivers"])


def test_audit_suite_from_a_mapping():
    suite = {
        "correctness": make_data({"A": [0] * 200, "B": [1] * 180 + [0] * 20}, 200),
        "tone": make_data({"A": [0, 1] * 60, "B": [1, 0] * 60}, 120),
    }
    report = audit_suite(suite)
    assert isinstance(report, SuiteReport)
    assert set(report.reports.keys()) == {"correctness", "tone"}


def test_audit_suite_from_a_file(tmp_path):
    p = tmp_path / "s.json"
    rows = []
    for i in range(60):
        rows += [{"id": str(i), "model": "A", "metric": "m1", "score": 0},
                 {"id": str(i), "model": "B", "metric": "m1", "score": 1},
                 {"id": str(i), "model": "A", "metric": "m2", "score": 1},
                 {"id": str(i), "model": "B", "metric": "m2", "score": 1}]
    p.write_text(json.dumps(rows))
    report = audit_suite(str(p))
    assert set(report.reports.keys()) == {"m1", "m2"}


def test_audit_suite_correction_is_threaded_through():
    suite = {
        "correctness": make_data({"A": [0] * 200, "B": [1] * 180 + [0] * 20}, 200),
        "tone": make_data({"A": [0, 1] * 60, "B": [1, 0] * 60}, 120),
    }
    bonf = audit_suite(suite)                     # default
    holm = audit_suite(suite, correction="holm")
    assert "bonferroni" in bonf.correction.lower()
    assert "holm" in holm.correction.lower()


def test_threshold_is_ignored_for_two_model_comparison(tmp_path):
    # Create two single-model files
    for name, model, rate in [("a.json", "gpt", 0.5), ("b.json", "claude", 0.5)]:
        (tmp_path / name).write_text(json.dumps({"models": [model], "examples": [
            {"id": str(i), "scores": {model: 1 if i < int(100 * rate) else 0}}
            for i in range(100)]}))

    # Two-file comparison should give the same result with or without threshold
    report_without_threshold = audit([str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    report_with_threshold = audit([str(tmp_path / "a.json"), str(tmp_path / "b.json")],
                                   threshold=0.9)

    # Results should be identical (threshold is ignored for comparisons)
    assert report_without_threshold.verdict.level == report_with_threshold.verdict.level
    assert len(report_without_threshold.findings) == len(report_with_threshold.findings)


def test_threshold_is_used_for_single_model_audit(tmp_path):
    # Create a single-model file with 70% score
    p = tmp_path / "model.json"
    p.write_text(json.dumps({"models": ["A"], "examples": [
        {"id": str(i), "scores": {"A": 1 if i < 70 else 0}}
        for i in range(100)]}))

    # With threshold=0.5, model exceeds target (70% > 50%)
    report_low_threshold = audit(str(p), threshold=0.5)
    low_threshold_findings = [f for f in report_low_threshold.findings
                              if "target" in f.title.lower() or "threshold" in f.title.lower()]
    assert low_threshold_findings, "Should have threshold-related finding"
    assert all(f.status.name in ("PASS", "OK") for f in low_threshold_findings), \
        f"Model should pass with low threshold, got: {[f.status.name for f in low_threshold_findings]}"

    # With threshold=0.9, model falls short (70% < 90%)
    report_high_threshold = audit(str(p), threshold=0.9)
    high_threshold_findings = [f for f in report_high_threshold.findings
                               if "target" in f.title.lower() or "threshold" in f.title.lower()]
    assert high_threshold_findings, "Should have threshold-related finding"
    assert any(f.status.name in ("FAIL", "WARN") for f in high_threshold_findings), \
        f"Model should fail with high threshold, got: {[f.status.name for f in high_threshold_findings]}"

    # Verify the reports actually differ
    assert report_low_threshold.verdict.level != report_high_threshold.verdict.level or \
           len(low_threshold_findings) != len(high_threshold_findings), \
        "Reports should differ based on threshold"
