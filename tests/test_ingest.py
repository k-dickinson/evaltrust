"""Tests for loading evaluation files from disk (JSON and CSV)."""

import json

import pytest

from evaltrust.adapters.registry import UnknownFormatError
from evaltrust.core.ingest import load


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_loads_native_json(tmp_path):
    raw = {"models": ["A", "B"],
           "examples": [{"id": "q1", "scores": {"A": 1, "B": 0}}]}
    data = load(write(tmp_path, "r.json", json.dumps(raw)))
    assert data.source_format == "native"
    assert set(data.models) == {"A", "B"}


def test_loads_promptfoo_json(tmp_path):
    raw = {"results": {"results": [
        {"provider": {"id": "gpt"}, "testIdx": 0, "score": 1},
        {"provider": {"id": "claude"}, "testIdx": 0, "score": 0},
    ]}}
    data = load(write(tmp_path, "pf.json", json.dumps(raw)))
    assert data.source_format == "promptfoo"


def test_loads_generic_records_json(tmp_path):
    raw = [{"id": "q1", "model": "A", "score": 1},
           {"id": "q1", "model": "B", "score": 0}]
    data = load(write(tmp_path, "g.json", json.dumps(raw)))
    assert data.source_format == "generic"


def test_loads_wide_csv(tmp_path):
    csv_text = "question,gpt,claude\nq1,1,0\nq2,0,1\n"
    data = load(write(tmp_path, "scores.csv", csv_text))
    assert data.source_format == "csv"
    assert set(data.models) == {"gpt", "claude"}
    assert data.n_examples == 2


def test_loads_long_csv(tmp_path):
    csv_text = "id,model,score\nq1,A,1\nq1,B,0\nq2,A,0\nq2,B,1\n"
    data = load(write(tmp_path, "long.csv", csv_text))
    assert set(data.models) == {"A", "B"}
    assert data.n_examples == 2


def test_unknown_json_raises_helpful_error(tmp_path):
    with pytest.raises(UnknownFormatError):
        load(write(tmp_path, "x.json", json.dumps({"nope": 1})))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(str(tmp_path / "does_not_exist.json"))


def test_malformed_json_gives_a_friendly_error(tmp_path):
    p = write(tmp_path, "broken.json", "{ not valid json")
    with pytest.raises(ValueError) as exc:
        load(p)
    msg = str(exc.value)
    assert "broken.json" in msg
    assert "JSON" in msg


# --- JSONL (line-delimited records, issue #17) --------------------------------


def test_loads_jsonl_records(tmp_path):
    text = ('{"id": "q1", "model": "A", "score": 1}\n'
            '{"id": "q1", "model": "B", "score": 0}\n'
            '{"id": "q2", "model": "A", "score": 0}\n'
            '{"id": "q2", "model": "B", "score": 1}\n')
    data = load(write(tmp_path, "r.jsonl", text))
    assert data.source_format == "jsonl"
    assert set(data.models) == {"A", "B"}
    assert data.n_examples == 2


def test_jsonl_tolerates_blank_lines_and_trailing_newline(tmp_path):
    text = ('\n{"id": "q1", "model": "A", "score": 1}\n\n'
            '{"id": "q1", "model": "B", "score": 0}\n\n')
    data = load(write(tmp_path, "r.jsonl", text))
    assert data.n_examples == 1
    assert set(data.models) == {"A", "B"}


def test_jsonl_malformed_line_names_the_line_number(tmp_path):
    text = ('{"id": "q1", "model": "A", "score": 1}\n'
            '{"id": "q1", "model": "B", "score": 0}\n'
            'this is not json\n')
    p = write(tmp_path, "bad.jsonl", text)
    with pytest.raises(ValueError) as exc:
        load(p)
    msg = str(exc.value)
    assert "bad.jsonl" in msg
    assert "line 3" in msg
    assert "JSON" in msg


def test_jsonl_unreadable_score_is_skipped_and_counted(tmp_path):
    text = ('{"id": "q1", "model": "A", "score": 1}\n'
            '{"id": "q1", "model": "B", "score": 0}\n'
            '{"id": "q2", "model": "A", "score": "banana"}\n'
            '{"id": "q2", "model": "B", "score": 1}\n')
    data = load(write(tmp_path, "r.jsonl", text))
    assert data.metadata["skipped_rows"] == 1
    assert set(data.models) == {"A", "B"}


def test_jsonl_whole_file_json_array_is_handled_as_json(tmp_path):
    # A JSON array mis-named .jsonl is really a single JSON document; load it
    # through the normal JSON path rather than treating '[' as a bad record.
    text = json.dumps([{"id": "q1", "model": "A", "score": 1},
                       {"id": "q1", "model": "B", "score": 0}])
    data = load(write(tmp_path, "actually.jsonl", text))
    assert data.source_format == "generic"
    assert set(data.models) == {"A", "B"}


def test_jsonl_stray_array_line_is_rejected_with_line_number(tmp_path):
    # Distinct from the whole-file-array case: an array on its own line is not a
    # record, and must fail loudly rather than be silently dropped.
    text = ('{"id": "q1", "model": "A", "score": 1}\n'
            '[1, 2, 3]\n')
    with pytest.raises(ValueError) as exc:
        load(write(tmp_path, "mixed.jsonl", text))
    assert "line 2" in str(exc.value)


def test_jsonl_unicode_line_separator_inside_a_value_does_not_split_record(tmp_path):
    # U+2028 (a Unicode line separator, legal unescaped in a JSON string) must
    # not be treated as a record boundary; str.splitlines() would split on it.
    model = "A\u2028B"
    text = '{"id": "q1", "model": "' + model + '", "score": 1}\n'
    data = load(write(tmp_path, "u.jsonl", text))
    assert data.models == [model]


def test_empty_jsonl_raises(tmp_path):
    with pytest.raises(Exception):
        load(write(tmp_path, "empty.jsonl", "\n  \n"))


def test_jsonl_handles_crlf_and_cr_line_endings(tmp_path):
    body = ['{"id": "q1", "model": "A", "score": 1}',
            '{"id": "q1", "model": "B", "score": 0}']
    for name, sep in [("crlf.jsonl", "\r\n"), ("cr.jsonl", "\r")]:
        data = load(write(tmp_path, name, sep.join(body) + sep))
        assert set(data.models) == {"A", "B"}, name
        assert data.n_examples == 1, name


def test_json_file_with_unknown_shape_still_raises_unknownformat(tmp_path):
    # A .json (not .jsonl) file routes only through JSON detection; an
    # unrecognised object must not silently fall through to the .jsonl reader.
    with pytest.raises(UnknownFormatError):
        load(write(tmp_path, "x.json", json.dumps({"a": 1})))


# --- two-file comparison (single-system tools) --------------------------------

from evaltrust.core.ingest import load_comparison


def single_model_file(tmp_path, name, model, rows):
    raw = {"models": [model],
           "examples": [{"id": k, "scores": {model: v}} for k, v in rows.items()]}
    return write(tmp_path, name, json.dumps(raw))


def test_one_path_loads_normally(tmp_path):
    raw = {"models": ["A", "B"], "examples": [{"id": "q1", "scores": {"A": 1, "B": 0}}]}
    data = load_comparison([write(tmp_path, "r.json", json.dumps(raw))])
    assert set(data.models) == {"A", "B"}


def test_two_paths_pair_and_label_by_model_name(tmp_path):
    a = single_model_file(tmp_path, "a.json", "gpt", {"q1": 1, "q2": 0})
    b = single_model_file(tmp_path, "b.json", "claude", {"q1": 0, "q2": 1})
    data = load_comparison([a, b])
    assert data.models == ["gpt", "claude"]
    assert data.n_examples == 2


def test_two_paths_fall_back_to_filenames_when_models_collide(tmp_path):
    a = single_model_file(tmp_path, "run_gpt.json", "model", {"q1": 1})
    b = single_model_file(tmp_path, "run_claude.json", "model", {"q1": 0})
    data = load_comparison([a, b])
    assert data.models == ["run_gpt", "run_claude"]


def test_explicit_labels_override(tmp_path):
    a = single_model_file(tmp_path, "a.json", "model", {"q1": 1})
    b = single_model_file(tmp_path, "b.json", "model", {"q1": 0})
    data = load_comparison([a, b], label_a="old", label_b="new")
    assert data.models == ["old", "new"]
