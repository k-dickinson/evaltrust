"""Tests for benchmark contamination audit."""

import json
import textwrap

from typer.testing import CliRunner

from evaltrust.audit.contamination import (
    ContaminationResult,
    _normalize_text,
    _find_exact_matches,
    _find_near_matches,
    run_contamination_audit,
)
from evaltrust.cli import app


# ---------------------------------------------------------------------------
# Unit tests — core logic
# ---------------------------------------------------------------------------

def test_normalize_text():
    assert _normalize_text("Hello, World!") == "hello world"
    assert _normalize_text("  Extra   spaces  \n here ") == "extra spaces here"
    assert _normalize_text("NoPunctuation") == "nopunctuation"
    assert _normalize_text("123!@#") == "123"
    assert _normalize_text("") == ""


def test_find_exact_matches():
    benchmark = ["This is a test.", "Another test!", "Unique string."]
    reference = ["this is a test", "completely different", "another   test"]

    matches = _find_exact_matches(benchmark, reference)
    assert matches == {0, 1}


def test_find_near_matches():
    benchmark = ["This is a test.", "Another test!", "Unique string."]
    # "anothr test" is a near match (typo) for "Another test!"
    reference = ["completely different", "anothr test", "this is a test"]

    # Assuming exact match found index 0 ("This is a test.")
    exact_matches = {0}
    near_matches = _find_near_matches(benchmark, reference, exact_matches)

    # It should skip index 0, and flag index 1 as a near match
    assert near_matches == {1}


def test_run_contamination_audit_partial():
    benchmark = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "What is 2+2?",
        "Where is the Eiffel Tower?"
    ]
    reference = [
        "what is the capital of france",  # Exact match for index 0
        "random text about space",        # No match for index 1
        "what is 2+2",                    # Exact match for index 2
        "where is the eiffel towr"        # Near match (typo) for index 3
    ]

    result = run_contamination_audit(benchmark, reference)

    assert result.exact_matches == 2
    assert result.near_matches == 1
    assert result.total_items == 4
    assert result.contamination_fraction == 3 / 4


def test_run_contamination_audit_empty():
    result = run_contamination_audit([], [])

    assert result.exact_matches == 0
    assert result.total_items == 0
    assert result.contamination_fraction == 0.0


def test_run_contamination_audit_no_overlap():
    benchmark = ["A", "B", "C"]
    reference = ["X", "Y", "Z"]

    result = run_contamination_audit(benchmark, reference)

    assert result.exact_matches == 0
    assert result.contamination_fraction == 0.0


# ---------------------------------------------------------------------------
# CLI-level tests
# ---------------------------------------------------------------------------

_runner = CliRunner()


def _write(tmp_path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_contamination_json_output(tmp_path):
    """--json emits valid JSON with the expected keys."""
    bench = _write(tmp_path, "bench.jsonl",
                   '{"prompt": "What is 2+2?"}\n{"prompt": "Capital of France?"}\n')
    ref = _write(tmp_path, "ref.jsonl",
                 '{"prompt": "what is 2+2"}\n{"prompt": "something else"}\n')

    result = _runner.invoke(app, ["contamination", bench, ref, "--json"])
    assert result.exit_code in (0, 1), result.output
    payload = json.loads(result.stdout)
    assert "total_items" in payload
    assert "exact_matches" in payload
    assert "near_matches" in payload
    assert "contamination_fraction" in payload
    assert payload["total_items"] == 2
    assert payload["exact_matches"] == 1


def test_contamination_json_no_human_text(tmp_path):
    """--json must not mix human-readable text into stdout."""
    bench = _write(tmp_path, "bench.jsonl", '{"prompt": "hello"}\n')
    ref = _write(tmp_path, "ref.jsonl", '{"prompt": "hello"}\n')

    result = _runner.invoke(app, ["contamination", bench, ref, "--json"])
    # stdout must be parseable as JSON — header/table must not appear there
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)


def test_contamination_fail_over_triggers(tmp_path):
    """--fail-over 0.0 exits 1 whenever any contamination is found."""
    bench = _write(tmp_path, "bench.jsonl", '{"prompt": "same text"}\n')
    ref = _write(tmp_path, "ref.jsonl", '{"prompt": "same text"}\n')

    result = _runner.invoke(app, ["contamination", bench, ref, "--fail-over", "0.0"])
    assert result.exit_code == 1


def test_contamination_fail_over_not_triggered(tmp_path):
    """--fail-over 1.0 never exits 1 (100 % threshold is never reached)."""
    bench = _write(tmp_path, "bench.jsonl",
                   '{"prompt": "completely unique A"}\n{"prompt": "completely unique B"}\n')
    ref = _write(tmp_path, "ref.jsonl", '{"prompt": "totally different"}\n')

    result = _runner.invoke(app, ["contamination", bench, ref, "--fail-over", "1.0"])
    assert result.exit_code == 0


def test_contamination_default_threshold_fires_at_15pct(tmp_path):
    """Default threshold is 15 %: 1 out of 6 (~16.7 %) triggers exit 1."""
    lines_bench = "\n".join(f'{{"prompt": "item {i}"}}' for i in range(6)) + "\n"
    # Only the first item matches — 1/6 ≈ 16.7 % > 15 %
    ref_content = '{"prompt": "item 0"}\n'
    bench = _write(tmp_path, "bench.jsonl", lines_bench)
    ref = _write(tmp_path, "ref.jsonl", ref_content)

    result = _runner.invoke(app, ["contamination", bench, ref])
    assert result.exit_code == 1


def test_contamination_skips_missing_column_rows(tmp_path):
    """Rows missing the target column are skipped with a warning, not an abort."""
    bench_content = textwrap.dedent("""\
        {"prompt": "item A"}
        {"other": "no prompt here"}
        {"prompt": "item B"}
    """)
    ref_content = '{"prompt": "item A"}\n'
    bench = _write(tmp_path, "bench.jsonl", bench_content)
    ref = _write(tmp_path, "ref.jsonl", ref_content)

    result = _runner.invoke(app, ["contamination", bench, ref, "--json"])
    # Should not crash — exit 0 or 1 depending on fraction, never 2
    assert result.exit_code in (0, 1), result.output
    payload = json.loads(result.stdout)
    # Only 2 rows were usable (the missing-column row is skipped)
    assert payload["total_items"] == 2
    # Warning appears in output (typer's CliRunner merges streams)
    assert "skipped" in (result.stderr or result.output).lower()


def test_contamination_all_rows_missing_column_exits_2(tmp_path):
    """If every row is missing the column the command exits 2 (usage error)."""
    bench = _write(tmp_path, "bench.jsonl", '{"other": "no prompt"}\n')
    ref = _write(tmp_path, "ref.jsonl", '{"prompt": "x"}\n')

    result = _runner.invoke(app, ["contamination", bench, ref])
    assert result.exit_code == 2
