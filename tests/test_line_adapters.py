"""Tests for line-format detection and the lm-eval proving adapter."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from evaltrust.adapters import line_registry
from evaltrust.adapters.common import Record
from evaltrust.adapters.line_registry import detect_line_adapter
from evaltrust.adapters.lm_eval import LMEvalAdapter
from evaltrust.core.ingest import load, load_suite


_TESTS_DIR = Path(__file__).parent
_REPO_ROOT = _TESTS_DIR.parent
# Row fields come from lm_eval/evaluator.py and evaluation_tracker.py at
# 97a5e2c. Metric names come from tests/testdata/arc_challenge-v2.0-res.json.
_LM_EVAL_FIXTURE = (
    _TESTS_DIR
    / "fixtures"
    / "samples_arc_challenge_2026-07-09T22-44-17.123456.jsonl"
)
# Sibling results file for the samples fixture. Top-level model_name /
# model_name_sanitized / model_source come from GeneralConfigTracker via
# save_results_aggregated at EleutherAI/lm-evaluation-harness@97a5e2c.
_LM_EVAL_RESULTS_FIXTURE = (
    _TESTS_DIR
    / "fixtures"
    / "results_2026-07-09T22-44-17.123456.json"
)
_LM_EVAL_MODEL = "EleutherAI/pythia-160m"


def _sample_row(doc_id: int = 0, acc: float = 1.0) -> dict:
    return {
        "doc_id": doc_id,
        "resps": [["A"]],
        "metrics": ["acc"],
        "acc": acc,
    }


def _write_samples(directory: Path, name: str, rows: list[dict] | None = None) -> Path:
    path = directory / name
    payload = rows if rows is not None else [_sample_row()]
    path.write_text("".join(json.dumps(row) + "\n" for row in payload))
    return path


def _write_results(
    directory: Path, name: str, model_name: object = _LM_EVAL_MODEL
) -> Path:
    path = directory / name
    path.write_text(json.dumps({
        "task_hashes": {},
        "model_source": "hf",
        "model_name": model_name,
        "model_name_sanitized": "EleutherAI__pythia-160m",
    }))
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class ToyLineAdapter:
    source_format = "toy-lines"

    def detect_lines(self, rows: list[dict]) -> bool:
        return bool(rows and rows[0].get("toy_line"))

    def parse_lines(
        self, rows: list[dict], *, path: Path | None = None
    ) -> tuple[list[Record], dict]:
        records = [
            Record("case-1", "toy-model", 1.0, metric="accuracy"),
            Record("case-1", "toy-model", 0.5, metric="helpfulness"),
        ]
        return records, {
            "skipped_rows": 2,
            "path_name": path.name if path is not None else None,
        }


@pytest.mark.parametrize("name", ["toy.jsonl", "toy"])
@pytest.mark.parametrize("entrypoint", [load, load_suite])
def test_line_hook_dispatches_for_both_entrypoints_and_routes(
    tmp_path, monkeypatch, name, entrypoint
):
    monkeypatch.setattr(line_registry, "LINE_REGISTRY", [ToyLineAdapter()])
    path = tmp_path / name
    path.write_text('{"toy_line": true}\n')

    result = entrypoint(str(path))

    datasets = [result] if entrypoint is load else list(result.values())
    assert all(data.source_format == "toy-lines" for data in datasets)
    assert all(data.metadata["skipped_rows"] == 2 for data in datasets)
    assert all(data.metadata["path_name"] == name for data in datasets)
    if entrypoint is load:
        assert result.examples[0].runs == {"toy-model": [1.0, 0.5]}
        assert result.examples[0].scores == {"toy-model": 0.75}
    else:
        assert list(result) == ["accuracy", "helpfulness"]


@pytest.mark.parametrize("name", ["generic.jsonl", "generic"])
def test_empty_line_registry_preserves_generic_jsonl_load_and_suite(
    tmp_path, monkeypatch, name
):
    monkeypatch.setattr(line_registry, "LINE_REGISTRY", [])
    rows = [
        {"id": "q1", "model": "A", "metric": "accuracy", "score": 1},
        {"id": "q1", "model": "A", "metric": "helpfulness", "score": 0.5},
    ]
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    data = load(str(path))
    suite = load_suite(str(path))

    assert asdict(data) == {
        "models": ["A"],
        "examples": [{
            "id": "q1",
            "scores": {"A": 0.75},
            "runs": {"A": [1.0, 0.5]},
            "judges": None,
            "preferences": None,
            "attributes": None,
        }],
        "source_format": "jsonl",
        "metadata": {"skipped_rows": 0},
    }
    assert list(suite) == ["accuracy", "helpfulness"]
    assert all(item.source_format == "jsonl" for item in suite.values())
    assert all(item.metadata == {"skipped_rows": 0} for item in suite.values())


@pytest.mark.parametrize("name", ["generic.jsonl", "generic"])
@pytest.mark.parametrize("entrypoint", [load, load_suite])
def test_registered_line_adapter_leaves_unclaimed_jsonl_on_generic_path(
    tmp_path, monkeypatch, name, entrypoint
):
    monkeypatch.setattr(line_registry, "LINE_REGISTRY", [ToyLineAdapter()])
    path = tmp_path / name
    path.write_text('{"id": "q1", "model": "A", "score": 1}\n')

    result = entrypoint(str(path))

    datasets = [result] if entrypoint is load else list(result.values())
    assert all(data.source_format == "jsonl" for data in datasets)
    assert all(data.metadata == {"skipped_rows": 0} for data in datasets)


def test_line_registry_uses_first_match():
    first = ToyLineAdapter()
    second = ToyLineAdapter()
    original = line_registry.LINE_REGISTRY
    try:
        line_registry.LINE_REGISTRY = [first, second]
        assert detect_line_adapter([{"toy_line": True}]) is first
    finally:
        line_registry.LINE_REGISTRY = original


def test_lm_eval_fixture_detects_and_parses_source_derived_shape():
    rows = _rows(_LM_EVAL_FIXTURE)
    adapter = LMEvalAdapter()

    assert adapter.detect_lines(rows)
    records, metadata = adapter.parse_lines(rows, path=_LM_EVAL_FIXTURE)

    assert len(records) == 4
    assert {record.metric for record in records} == {"acc", "acc_norm"}
    assert {record.example_id for record in records} == {"0", "1"}
    assert {record.model for record in records} == {_LM_EVAL_MODEL}
    assert metadata == {
        "skipped_rows": 0,
        "model_name_inferred": False,
        "model_name_source": _LM_EVAL_RESULTS_FIXTURE.name,
    }


def test_lm_eval_multi_metric_fixture_fans_into_suite():
    suite = load_suite(str(_LM_EVAL_FIXTURE))

    assert list(suite) == ["acc", "acc_norm"]
    assert all(data.source_format == "lm-eval" for data in suite.values())
    assert all(data.models == [_LM_EVAL_MODEL] for data in suite.values())
    assert all(data.metadata["model_name_inferred"] is False for data in suite.values())
    assert all(
        data.metadata["model_name_source"] == _LM_EVAL_RESULTS_FIXTURE.name
        for data in suite.values()
    )
    assert [example.scores[_LM_EVAL_MODEL] for example in suite["acc"].examples] == [
        1.0,
        1.0,
    ]


def test_lm_eval_load_keeps_existing_multi_metric_load_semantics():
    data = load(str(_LM_EVAL_FIXTURE))

    assert data.source_format == "lm-eval"
    assert data.examples[0].runs == {_LM_EVAL_MODEL: [1.0, 0.75]}
    assert data.examples[0].scores == {_LM_EVAL_MODEL: 0.875}


def test_lm_eval_sibling_matching_timestamp_uses_results_model_name(tmp_path):
    samples = _write_samples(
        tmp_path, "samples_arc_challenge_2026-07-09T22-44-17.123456.jsonl"
    )
    _write_results(tmp_path, "results_2026-07-09T22-44-17.123456.json")

    records, metadata = LMEvalAdapter().parse_lines(_rows(samples), path=samples)

    assert {record.model for record in records} == {_LM_EVAL_MODEL}
    assert metadata == {
        "skipped_rows": 0,
        "model_name_inferred": False,
        "model_name_source": "results_2026-07-09T22-44-17.123456.json",
    }


def test_lm_eval_no_sibling_falls_back_to_filename_inference(tmp_path):
    samples = _write_samples(
        tmp_path, "samples_arc_challenge_2026-07-09T22-44-17.123456.jsonl"
    )

    records, metadata = LMEvalAdapter().parse_lines(_rows(samples), path=samples)

    assert {record.model for record in records} == {"arc_challenge"}
    assert metadata == {"skipped_rows": 0, "model_name_inferred": True}
    assert "model_name_source" not in metadata


def test_lm_eval_single_nonmatching_timestamp_sibling_is_used(tmp_path):
    samples = _write_samples(
        tmp_path, "samples_arc_challenge_2026-07-09T22-44-17.123456.jsonl"
    )
    _write_results(tmp_path, "results_2020-01-01T00-00-00.000000.json")

    records, metadata = LMEvalAdapter().parse_lines(_rows(samples), path=samples)

    assert {record.model for record in records} == {_LM_EVAL_MODEL}
    assert metadata == {
        "skipped_rows": 0,
        "model_name_inferred": False,
        "model_name_source": "results_2020-01-01T00-00-00.000000.json",
    }


def test_lm_eval_multiple_siblings_no_timestamp_match_falls_back(tmp_path):
    samples = _write_samples(
        tmp_path, "samples_arc_challenge_2026-07-09T22-44-17.123456.jsonl"
    )
    _write_results(tmp_path, "results_2020-01-01T00-00-00.000000.json", "model-a")
    _write_results(tmp_path, "results_2021-01-01T00-00-00.000000.json", "model-b")

    records, metadata = LMEvalAdapter().parse_lines(_rows(samples), path=samples)

    assert {record.model for record in records} == {"arc_challenge"}
    assert metadata == {"skipped_rows": 0, "model_name_inferred": True}
    assert "model_name_source" not in metadata


def test_lm_eval_timestamp_match_wins_regardless_of_sort_order(tmp_path):
    samples = _write_samples(
        tmp_path, "samples_arc_challenge_2026-07-09T22-44-17.123456.jsonl"
    )
    # Lexicographically earlier and later non-matches sandwich the match.
    _write_results(tmp_path, "results_2019-01-01T00-00-00.000000.json", "model-early")
    _write_results(
        tmp_path, "results_2026-07-09T22-44-17.123456.json", "model-matched"
    )
    _write_results(tmp_path, "results_2029-01-01T00-00-00.000000.json", "model-late")

    records, metadata = LMEvalAdapter().parse_lines(_rows(samples), path=samples)

    assert {record.model for record in records} == {"model-matched"}
    assert metadata == {
        "skipped_rows": 0,
        "model_name_inferred": False,
        "model_name_source": "results_2026-07-09T22-44-17.123456.json",
    }


def test_lm_eval_malformed_results_json_falls_back(tmp_path):
    samples = _write_samples(
        tmp_path, "samples_arc_challenge_2026-07-09T22-44-17.123456.jsonl"
    )
    (tmp_path / "results_2026-07-09T22-44-17.123456.json").write_text("{not json")

    records, metadata = LMEvalAdapter().parse_lines(_rows(samples), path=samples)

    assert {record.model for record in records} == {"arc_challenge"}
    assert metadata == {"skipped_rows": 0, "model_name_inferred": True}


@pytest.mark.parametrize("model_name", [None, "", 42, ["x"]])
def test_lm_eval_invalid_model_name_in_results_falls_back(tmp_path, model_name):
    samples = _write_samples(
        tmp_path, "samples_arc_challenge_2026-07-09T22-44-17.123456.jsonl"
    )
    _write_results(
        tmp_path, "results_2026-07-09T22-44-17.123456.json", model_name=model_name
    )

    records, metadata = LMEvalAdapter().parse_lines(_rows(samples), path=samples)

    assert {record.model for record in records} == {"arc_challenge"}
    assert metadata == {"skipped_rows": 0, "model_name_inferred": True}


def test_lm_eval_path_none_falls_back_to_inference():
    rows = [_sample_row()]

    records, metadata = LMEvalAdapter().parse_lines(rows, path=None)

    assert {record.model for record in records} == {"model"}
    assert metadata == {"skipped_rows": 0, "model_name_inferred": True}
    assert "model_name_source" not in metadata


def test_lm_eval_results_lookup_is_deterministic_with_multiple_siblings(tmp_path):
    samples = _write_samples(
        tmp_path, "samples_arc_challenge_2026-07-09T22-44-17.123456.jsonl"
    )
    _write_results(tmp_path, "results_2019-01-01T00-00-00.000000.json", "model-a")
    _write_results(
        tmp_path, "results_2026-07-09T22-44-17.123456.json", "model-matched"
    )
    _write_results(tmp_path, "results_2029-01-01T00-00-00.000000.json", "model-b")

    first = LMEvalAdapter().parse_lines(_rows(samples), path=samples)
    second = LMEvalAdapter().parse_lines(_rows(samples), path=samples)

    assert first == second


def test_lm_eval_counts_each_unreadable_metric_value(tmp_path):
    rows = [{
        "doc_id": 7,
        "resps": [["answer"]],
        "metrics": ["acc", "custom_metric"],
        "acc": "unreadable",
        "custom_metric": "pass",
    }]
    path = tmp_path / "samples_custom_task_2026-07-09T22-44-17.123456.jsonl"

    records, metadata = LMEvalAdapter().parse_lines(rows, path=path)

    assert [(record.metric, record.score) for record in records] == [
        ("custom_metric", 1.0)
    ]
    assert metadata == {"skipped_rows": 1, "model_name_inferred": True}


def test_lm_eval_reserves_every_field_emitted_by_serializer(tmp_path):
    row = {
        "doc_id": 3,
        "doc": 1,
        "target": 1,
        "arguments": 1,
        "resps": [1],
        "filtered_resps": [1],
        "filter": 1,
        "metrics": ["task_metric"],
        "doc_hash": 1,
        "prompt_hash": 1,
        "target_hash": 1,
        "task_metric": 0.25,
        "incidental_number": 17,
    }
    path = tmp_path / "samples_task.jsonl"

    records, _ = LMEvalAdapter().parse_lines([row], path=path)

    assert [(record.metric, record.score) for record in records] == [
        ("task_metric", 0.25)
    ]


def test_lm_eval_falls_back_to_non_reserved_fields_without_metrics_list(tmp_path):
    row = {
        "doc_id": 4,
        "resps": [["answer"]],
        "task_metric": 0.5,
        "legacy_numeric_field": 12,
    }
    path = tmp_path / "samples_task.jsonl"

    records, _ = LMEvalAdapter().parse_lines([row], path=path)

    assert [(record.metric, record.score) for record in records] == [
        ("task_metric", 0.5),
        ("legacy_numeric_field", 12.0),
    ]


@pytest.mark.parametrize(
    "rows",
    [
        [{"id": "q1", "model": "A", "score": 1}],
        [{"doc_id": 1, "resps": [["A"]], "model": "A", "acc": 1}],
        [{"doc_id": 1, "acc": 1}],
    ],
)
def test_lm_eval_detection_rejects_generic_and_non_lm_eval_rows(rows):
    assert not LMEvalAdapter().detect_lines(rows)


def test_lm_eval_detection_rejects_existing_json_fixtures():
    adapter = LMEvalAdapter()
    files = list((_TESTS_DIR / "fixtures").glob("*.json")) + list(
        (_REPO_ROOT / "examples").glob("*.json")
    )

    for path in files:
        raw = json.loads(path.read_text())
        rows = raw if isinstance(raw, list) else [raw]
        assert not adapter.detect_lines(rows), path


@pytest.mark.parametrize("entrypoint", [load, load_suite])
def test_line_numbered_jsonl_error_is_unchanged(entrypoint, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "q1", "model": "A", "score": 1}\nnot json\n')

    with pytest.raises(ValueError, match="line 2"):
        entrypoint(str(path))


@pytest.mark.parametrize("entrypoint", [load, load_suite])
def test_array_document_misnamed_jsonl_still_uses_json_route(entrypoint, tmp_path):
    path = tmp_path / "array.jsonl"
    path.write_text(json.dumps([{"id": "q1", "model": "A", "score": 1}]))

    result = entrypoint(str(path))

    datasets = [result] if entrypoint is load else list(result.values())
    assert all(data.source_format == "generic" for data in datasets)
