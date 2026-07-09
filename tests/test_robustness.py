"""Real-world data is messy: missing cells, unreadable scores, non-binary rubric
scores. EvalTrust should skip and report bad rows, not crash on them."""

import json

from evaltrust.audit.runner import run_audit
from evaltrust.audit.verdict import VerdictLevel
from evaltrust.core.ingest import load
from evaltrust.core.schema import Status


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_long_csv_with_bad_score_cell_does_not_crash(tmp_path):
    csv_text = ("id,model,score\n"
                "q1,A,1\nq1,B,0\n"
                "q2,A,x\nq2,B,1\n"      # q2/A score is unreadable
                "q3,A,0\nq3,B,1\n")
    data = load(write(tmp_path, "long.csv", csv_text))
    assert set(data.models) == {"A", "B"}
    assert data.metadata.get("skipped_rows", 0) >= 1


def test_missing_score_cell_is_skipped(tmp_path):
    csv_text = "id,model,score\nq1,A,1\nq1,B,0\nq2,A,\nq2,B,1\n"
    data = load(write(tmp_path, "m.csv", csv_text))
    assert data.metadata.get("skipped_rows", 0) >= 1


def test_generic_json_with_bad_score_is_skipped(tmp_path):
    rows = [{"id": "q1", "model": "A", "score": 1},
            {"id": "q1", "model": "B", "score": 0},
            {"id": "q2", "model": "A", "score": "oops"},
            {"id": "q2", "model": "B", "score": 1}]
    data = load(write(tmp_path, "r.json", json.dumps(rows)))
    assert data.metadata.get("skipped_rows", 0) >= 1


def test_audit_reports_skipped_rows_as_a_finding(tmp_path):
    csv_text = "id,model,score\nq1,A,1\nq1,B,0\nq2,A,bad\nq2,B,1\nq3,A,0\nq3,B,1\n"
    data = load(write(tmp_path, "x.csv", csv_text))
    report = run_audit(data)
    dq = [f for f in report.findings if f.pillar == "Data Quality"]
    assert dq and dq[0].status is Status.WARN


def test_clean_file_has_no_data_quality_finding(tmp_path):
    csv_text = "id,model,score\nq1,A,1\nq1,B,0\nq2,A,0\nq2,B,1\n"
    data = load(write(tmp_path, "clean.csv", csv_text))
    report = run_audit(data)
    assert not [f for f in report.findings if f.pillar == "Data Quality"]


def test_non_binary_rubric_scores_audit_without_crashing(tmp_path):
    # 1-5 rubric scores (continuous path, not McNemar).
    rows = ["id,model,score"]
    for i in range(60):
        rows.append(f"q{i},A,{1 + i % 4}")
        rows.append(f"q{i},B,{2 + i % 4}")
    data = load(write(tmp_path, "rubric.csv", "\n".join(rows) + "\n"))
    report = run_audit(data)
    assert report.verdict.level in (VerdictLevel.HIGH, VerdictLevel.MODERATE,
                                    VerdictLevel.LOW)


def test_model_with_no_comparable_scores_errors_cleanly(tmp_path):
    import pytest
    from evaltrust.audit.runner import run_audit
    # B is declared but has no per-example scores.
    data = load(write(tmp_path, "one.json",
                      '{"models":["A","B"],"examples":['
                      '{"id":"q1","scores":{"A":1}},{"id":"q2","scores":{"A":0}}]}'))
    with pytest.raises(ValueError):
        run_audit(data, model_a="A", model_b="B")


def test_unmatched_pairing_examples_reported_as_a_finding():
    from evaltrust.core.schema import EvalData, Example

    examples = [Example(id=str(i), scores={"A": float(i % 2), "B": float((i + 1) % 2)})
                for i in range(40)]
    data = EvalData(models=["A", "B"], examples=examples, source_format="t+t",
                    metadata={"unmatched_examples": 25})
    report = run_audit(data)
    dq = [f for f in report.findings
          if f.details.get("check") == "pairing_coverage"]
    assert dq and dq[0].status is Status.WARN
    assert "25" in dq[0].title


def test_fully_paired_files_have_no_pairing_finding():
    from evaltrust.core.schema import EvalData, Example

    examples = [Example(id=str(i), scores={"A": float(i % 2), "B": float((i + 1) % 2)})
                for i in range(40)]
    data = EvalData(models=["A", "B"], examples=examples, source_format="t+t",
                    metadata={"unmatched_examples": 0, "skipped_rows": 0})
    report = run_audit(data)
    assert not [f for f in report.findings
                if f.details.get("check") == "pairing_coverage"]
