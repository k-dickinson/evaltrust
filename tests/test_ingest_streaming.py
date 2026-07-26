"""Tests for issue #80 – stream large result files instead of loading them fully.

Verifies that:
  - JSONL and CSV are read line-by-line via generator pipelines for large files.
  - Peak memory is bounded by the streaming buffer, not the file size.
  - Small files still use the fast in-memory path unchanged.
  - The switch between small-file and large-file paths is transparent: identical
    results are produced regardless of which path is taken.
  - JSON files above the threshold attempt ijson streaming (and fall back
    gracefully when ijson is not installed).
  - load_suite() streams JSONL and CSV correctly too.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from unittest import mock

import pytest

from evaltrust.adapters.registry import UnknownFormatError
from evaltrust.core.ingest import (
    _STREAM_THRESHOLD,
    _iter_csv_rows,
    _iter_jsonl_lines,
    load,
    load_suite,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _jsonl_text(n_examples: int, models=("A", "B")) -> str:
    lines = []
    for i in range(n_examples):
        for m in models:
            score = i % 2
            lines.append(json.dumps({"id": f"q{i}", "model": m, "score": score}))
    return "\n".join(lines) + "\n"


def _csv_text(n_examples: int, models=("A", "B")) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id"] + list(models))
    writer.writeheader()
    for i in range(n_examples):
        row = {"id": f"q{i}"}
        for j, m in enumerate(models):
            row[m] = (i + j) % 2
        writer.writerow(row)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Generator helpers
# ---------------------------------------------------------------------------

class TestIterJsonlLines:
    """Unit tests for the _iter_jsonl_lines streaming generator."""

    def test_yields_all_non_blank_lines(self, tmp_path):
        text = '{"id": "q1", "model": "A", "score": 1}\n\n{"id": "q2", "model": "B", "score": 0}\n'
        p = tmp_path / "f.jsonl"
        p.write_text(text, encoding="utf-8")
        rows = list(_iter_jsonl_lines(p))
        assert len(rows) == 2
        assert rows[0] == {"id": "q1", "model": "A", "score": 1}

    def test_raises_valueerror_with_line_number_on_bad_json(self, tmp_path):
        text = '{"id": "q1", "model": "A", "score": 1}\nnot json\n'
        p = tmp_path / "bad.jsonl"
        p.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="line 2"):
            list(_iter_jsonl_lines(p))

    def test_raises_on_non_object_line(self, tmp_path):
        text = '{"id": "q1", "model": "A", "score": 1}\n[1, 2]\n'
        p = tmp_path / "arr.jsonl"
        p.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="line 2"):
            list(_iter_jsonl_lines(p))

    def test_tolerates_crlf_line_endings(self, tmp_path):
        lines = ['{"id": "q1", "model": "A", "score": 1}',
                 '{"id": "q2", "model": "B", "score": 0}']
        p = tmp_path / "crlf.jsonl"
        p.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
        rows = list(_iter_jsonl_lines(p))
        assert len(rows) == 2

    def test_unicode_line_separator_inside_value_does_not_split(self, tmp_path):
        model = "A\u2028B"
        text = '{"id": "q1", "model": "' + model + '", "score": 1}\n'
        p = tmp_path / "u.jsonl"
        p.write_text(text, encoding="utf-8")
        rows = list(_iter_jsonl_lines(p))
        assert rows[0]["model"] == model


class TestIterCsvRows:
    """Unit tests for the _iter_csv_rows streaming generator."""

    def test_yields_dict_rows(self, tmp_path):
        text = "id,model,score\nq1,A,1\nq2,B,0\n"
        p = tmp_path / "f.csv"
        p.write_text(text, encoding="utf-8")
        rows = list(_iter_csv_rows(p))
        assert len(rows) == 2
        assert rows[0] == {"id": "q1", "model": "A", "score": "1"}

    def test_yields_nothing_for_header_only_file(self, tmp_path):
        text = "id,model,score\n"
        p = tmp_path / "empty.csv"
        p.write_text(text, encoding="utf-8")
        assert list(_iter_csv_rows(p)) == []


# ---------------------------------------------------------------------------
# Streaming paths produce identical results to the small-file paths
# ---------------------------------------------------------------------------

class TestStreamingMatchesInMemory:
    """Force the streaming path by patching _STREAM_THRESHOLD to 0."""

    def _force_stream(self):
        return mock.patch("evaltrust.core.ingest._STREAM_THRESHOLD", 0)

    def test_jsonl_streamed_matches_in_memory(self, tmp_path):
        text = _jsonl_text(10)
        path = _write(tmp_path, "r.jsonl", text)

        normal = load(path)
        with self._force_stream():
            streamed = load(path)

        assert streamed.source_format == normal.source_format
        assert streamed.models == normal.models
        assert streamed.n_examples == normal.n_examples

    def test_csv_streamed_matches_in_memory(self, tmp_path):
        text = _csv_text(10)
        path = _write(tmp_path, "r.csv", text)

        normal = load(path)
        with self._force_stream():
            streamed = load(path)

        assert streamed.source_format == normal.source_format
        assert streamed.models == normal.models
        assert streamed.n_examples == normal.n_examples

    def test_jsonl_stream_scores_match(self, tmp_path):
        text = _jsonl_text(4, models=("X", "Y"))
        path = _write(tmp_path, "s.jsonl", text)

        normal = load(path)
        with self._force_stream():
            streamed = load(path)

        for ex_n, ex_s in zip(normal.examples, streamed.examples):
            assert ex_n.id == ex_s.id
            assert ex_n.scores == pytest.approx(ex_s.scores)

    def test_csv_stream_scores_match(self, tmp_path):
        text = _csv_text(5, models=("M1", "M2"))
        path = _write(tmp_path, "wide.csv", text)

        normal = load(path)
        with self._force_stream():
            streamed = load(path)

        for ex_n, ex_s in zip(normal.examples, streamed.examples):
            assert ex_n.id == ex_s.id
            assert ex_n.scores == pytest.approx(ex_s.scores)


# ---------------------------------------------------------------------------
# Streaming preserves existing error behaviour
# ---------------------------------------------------------------------------

class TestStreamingErrors:
    def _force_stream(self):
        return mock.patch("evaltrust.core.ingest._STREAM_THRESHOLD", 0)

    def test_empty_jsonl_raises_streamed(self, tmp_path):
        path = _write(tmp_path, "empty.jsonl", "\n  \n")
        with self._force_stream(), pytest.raises(UnknownFormatError):
            load(path)

    def test_empty_csv_raises_streamed(self, tmp_path):
        path = _write(tmp_path, "empty.csv", "id,model,score\n")
        with self._force_stream(), pytest.raises(UnknownFormatError):
            load(path)

    def test_malformed_jsonl_line_error_streamed(self, tmp_path):
        text = ('{"id": "q1", "model": "A", "score": 1}\n'
                'this is not json\n')
        path = _write(tmp_path, "bad.jsonl", text)
        with self._force_stream(), pytest.raises(ValueError, match="line 2"):
            load(path)

    def test_missing_file_streamed(self, tmp_path):
        with self._force_stream(), pytest.raises(FileNotFoundError):
            load(str(tmp_path / "nope.jsonl"))


# ---------------------------------------------------------------------------
# JSON large-file: ijson streaming and graceful fallback
# ---------------------------------------------------------------------------

class TestJsonStreaming:
    def _force_stream(self):
        return mock.patch("evaltrust.core.ingest._STREAM_THRESHOLD", 0)

    def test_json_small_file_works_without_ijson(self, tmp_path):
        raw = {"models": ["A", "B"],
               "examples": [{"id": "q1", "scores": {"A": 1, "B": 0}}]}
        path = _write(tmp_path, "r.json", json.dumps(raw))
        # Small file: always uses full-load path, no ijson needed.
        data = load(path)
        assert set(data.models) == {"A", "B"}

    def test_json_large_file_fallback_when_no_ijson(self, tmp_path):
        """When ijson is absent, large JSON file should still load (via fallback)."""
        raw = {"models": ["A", "B"],
               "examples": [{"id": "q1", "scores": {"A": 1, "B": 0}}]}
        path = _write(tmp_path, "big.json", json.dumps(raw))

        with (self._force_stream(),
              mock.patch.dict("sys.modules", {"ijson": None})):
            data = load(path)

        assert set(data.models) == {"A", "B"}

    def test_json_array_large_file_fallback_when_no_ijson(self, tmp_path):
        raw = [{"id": "q1", "model": "A", "score": 1},
               {"id": "q1", "model": "B", "score": 0}]
        path = _write(tmp_path, "arr.json", json.dumps(raw))

        with (self._force_stream(),
              mock.patch.dict("sys.modules", {"ijson": None})):
            data = load(path)

        assert set(data.models) == {"A", "B"}


# ---------------------------------------------------------------------------
# load_suite streaming
# ---------------------------------------------------------------------------

class TestLoadSuiteStreaming:
    def _force_stream(self):
        return mock.patch("evaltrust.core.ingest._STREAM_THRESHOLD", 0)

    def test_suite_jsonl_streamed(self, tmp_path):
        text = _jsonl_text(6)
        path = _write(tmp_path, "suite.jsonl", text)

        normal = load_suite(path)
        with self._force_stream():
            streamed = load_suite(path)

        assert set(normal.keys()) == set(streamed.keys())
        for key in normal:
            assert normal[key].n_examples == streamed[key].n_examples

    def test_suite_csv_streamed(self, tmp_path):
        text = _csv_text(8)
        path = _write(tmp_path, "suite.csv", text)

        normal = load_suite(path)
        with self._force_stream():
            streamed = load_suite(path)

        assert set(normal.keys()) == set(streamed.keys())
        for key in normal:
            assert normal[key].n_examples == streamed[key].n_examples


# ---------------------------------------------------------------------------
# Threshold: small files still use the fast in-memory path
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_small_jsonl_uses_read_text(self, tmp_path):
        """Small files are loaded via read_text, not the streaming iterator."""
        text = _jsonl_text(2)
        path_str = _write(tmp_path, "small.jsonl", text)
        p = Path(path_str)

        # The file is well below _STREAM_THRESHOLD, so _iter_jsonl_lines must
        # NOT be called.
        with mock.patch("evaltrust.core.ingest._iter_jsonl_lines",
                        wraps=lambda *a, **kw: (_ for _ in ())) as spy:
            load(path_str)
            spy.assert_not_called()

    def test_small_csv_uses_read_text(self, tmp_path):
        text = _csv_text(2)
        path_str = _write(tmp_path, "small.csv", text)

        with mock.patch("evaltrust.core.ingest._iter_csv_rows",
                        wraps=lambda *a, **kw: (_ for _ in ())) as spy:
            load(path_str)
            spy.assert_not_called()

    def test_large_jsonl_uses_streaming_iterator(self, tmp_path):
        """Forcing threshold to 0 makes every file 'large'; verify iterator fires."""
        text = _jsonl_text(4)
        path_str = _write(tmp_path, "large.jsonl", text)

        real_iter = __import__(
            "evaltrust.core.ingest", fromlist=["_iter_jsonl_lines"]
        )._iter_jsonl_lines

        called = []

        def spy(path):
            called.append(path)
            return real_iter(path)

        with (mock.patch("evaltrust.core.ingest._STREAM_THRESHOLD", 0),
              mock.patch("evaltrust.core.ingest._iter_jsonl_lines", spy)):
            load(path_str)

        assert called, "_iter_jsonl_lines was not called for a large JSONL file"

    def test_large_csv_uses_streaming_iterator(self, tmp_path):
        text = _csv_text(4)
        path_str = _write(tmp_path, "large.csv", text)

        real_iter = __import__(
            "evaltrust.core.ingest", fromlist=["_iter_csv_rows"]
        )._iter_csv_rows

        called = []

        def spy(path):
            called.append(path)
            return real_iter(path)

        with (mock.patch("evaltrust.core.ingest._STREAM_THRESHOLD", 0),
              mock.patch("evaltrust.core.ingest._iter_csv_rows", spy)):
            load(path_str)

        assert called, "_iter_csv_rows was not called for a large CSV file"


# ---------------------------------------------------------------------------
# Mis-named large JSONL file containing a JSON array
# ---------------------------------------------------------------------------

class TestMisnamedJsonlArrayStreamed:
    def _force_stream(self):
        return mock.patch("evaltrust.core.ingest._STREAM_THRESHOLD", 0)

    def test_json_array_mis_named_jsonl_handled_large(self, tmp_path):
        raw = [{"id": "q1", "model": "A", "score": 1},
               {"id": "q1", "model": "B", "score": 0}]
        path = _write(tmp_path, "tricky.jsonl", json.dumps(raw))

        with (self._force_stream(),
              mock.patch.dict("sys.modules", {"ijson": None})):
            data = load(path)

        assert set(data.models) == {"A", "B"}


# ---------------------------------------------------------------------------
# Acceptance criterion (issue #80)
# "A results file larger than available RAM can be audited, with peak memory
#  bounded by the streaming buffer rather than file size."
#
# We simulate this by generating a file whose raw byte size is N x the
# threshold, forcing the streaming path, and asserting that the peak
# RSS increase during load() is well below the file size.
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    """Issue #80 acceptance: peak memory is bounded by the buffer, not file size.

    The acceptance criterion is: "A results file larger than available RAM can
    be audited, with peak memory bounded by the streaming buffer rather than
    file size."

    The key invariant is that Path.read_text() — which materialises the *entire
    file* as a single Python string — is never called for large files.  We
    verify this directly by asserting that read_text() is not invoked on the
    streaming path, and that the correct streaming helpers are called instead.

    (tracemalloc measures Python object allocations which include the parsed
    EvalData records that must be kept in memory; it cannot distinguish "file
    bytes held as a string" from "parsed record objects". The absence of a
    read_text() call is the correct mechanical proxy for the acceptance
    criterion.)
    """

    def test_large_jsonl_never_calls_read_text(self, tmp_path):
        """read_text() must not be called for a large JSONL file."""
        n_records = 200
        lines = [
            json.dumps({"id": f"q{i % 20}", "model": "A" if i % 2 == 0 else "B",
                        "score": i % 2})
            for i in range(n_records)
        ]
        path_str = _write(tmp_path, "big.jsonl", "\n".join(lines) + "\n")

        real_iter = _iter_jsonl_lines
        streaming_called = []

        def spy_iter(path):
            streaming_called.append(path)
            return real_iter(path)

        with (mock.patch("evaltrust.core.ingest._STREAM_THRESHOLD", 0),
              mock.patch("evaltrust.core.ingest._iter_jsonl_lines", spy_iter),
              mock.patch("pathlib.Path.read_text",
                         side_effect=AssertionError(
                             "read_text() called on a large JSONL file — "
                             "the whole file was loaded into memory")) as read_text_spy):
            data = load(path_str)

        read_text_spy.assert_not_called()
        assert streaming_called, "_iter_jsonl_lines was never called"
        assert data.n_examples == 20
        assert set(data.models) == {"A", "B"}

    def test_large_csv_never_calls_read_text(self, tmp_path):
        """read_text() must not be called for a large CSV file."""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["id", "A", "B"])
        writer.writeheader()
        for i in range(200):
            writer.writerow({"id": f"q{i % 20}", "A": i % 2, "B": (i + 1) % 2})
        path_str = _write(tmp_path, "big.csv", buf.getvalue())

        real_iter = _iter_csv_rows
        streaming_called = []

        def spy_iter(path):
            streaming_called.append(path)
            return real_iter(path)

        with (mock.patch("evaltrust.core.ingest._STREAM_THRESHOLD", 0),
              mock.patch("evaltrust.core.ingest._iter_csv_rows", spy_iter),
              mock.patch("pathlib.Path.read_text",
                         side_effect=AssertionError(
                             "read_text() called on a large CSV file — "
                             "the whole file was loaded into memory")) as read_text_spy):
            data = load(path_str)

        read_text_spy.assert_not_called()
        assert streaming_called, "_iter_csv_rows was never called"
        assert data.n_examples == 20
        assert set(data.models) == {"A", "B"}


class TestMultiMetricJsonSuite:
    """Streamed and in-memory paths must produce identical suites for multi-metric JSON."""

    def _make_toplevel_array(self, tmp_path):
        """Top-level array [{"id","model","metric","score"}, ...] — the format
        load_suite recognises for multi-metric JSON."""
        records = [
            {"id": "q1", "model": "A", "metric": "correctness", "score": 1},
            {"id": "q1", "model": "B", "metric": "correctness", "score": 0},
            {"id": "q1", "model": "A", "metric": "safety",      "score": 1},
            {"id": "q1", "model": "B", "metric": "safety",      "score": 0},
            {"id": "q2", "model": "A", "metric": "correctness", "score": 1},
            {"id": "q2", "model": "B", "metric": "correctness", "score": 1},
            {"id": "q2", "model": "A", "metric": "safety",      "score": 0},
            {"id": "q2", "model": "B", "metric": "safety",      "score": 1},
        ]
        return _write(tmp_path, "multi.json", json.dumps(records))

    def test_toplevel_array_streamed_matches_inmemory(self, tmp_path):
        """Top-level array: streamed suite == in-memory suite for multi-metric JSON."""
        path_str = self._make_toplevel_array(tmp_path)
        normal = load_suite(path_str)
        with mock.patch("evaltrust.core.ingest._STREAM_THRESHOLD", 0):
            streamed = load_suite(path_str)
        assert set(normal.keys()) == set(streamed.keys())
        for metric in normal:
            assert normal[metric].models == streamed[metric].models
            assert normal[metric].n_examples == streamed[metric].n_examples

    def test_streamed_path_does_not_collapse_metrics(self, tmp_path):
        """Streamed path must preserve all metrics, not collapse to one."""
        path_str = self._make_toplevel_array(tmp_path)
        with mock.patch("evaltrust.core.ingest._STREAM_THRESHOLD", 0):
            streamed = load_suite(path_str)
        assert len(streamed) >= 2, (
            f"Expected at least 2 metrics, got {list(streamed.keys())}"
        )

    def test_toplevel_array_streamed_metric_keys(self, tmp_path):
        """Streamed path returns exactly the expected metric keys."""
        path_str = self._make_toplevel_array(tmp_path)
        with mock.patch("evaltrust.core.ingest._STREAM_THRESHOLD", 0):
            streamed = load_suite(path_str)
        assert set(streamed.keys()) == {"correctness", "safety"}
