"""Tests for the Hugging Face Lighteval details adapter.

Lighteval writes per-sample scores to details Parquet files. Recent official
reference artifacts use columns doc / metric / model_response; the current
public documentation describes __doc__ / __metric__ / __model_response__.
Both exact aliases are accepted. The aggregate results.json holds task
summaries only. Detection is structural; filenames are never consulted.
"""

import copy
import json
from pathlib import Path

import pytest

from evaltrust.adapters.deepeval import DeepEvalAdapter
from evaltrust.adapters.generic import GenericRecordsAdapter, NativeNestedAdapter
from evaltrust.adapters.helm import HelmAdapter
from evaltrust.adapters.inspect_ai import InspectAdapter
from evaltrust.adapters.lighteval import LightevalAdapter
from evaltrust.adapters.promptfoo import PromptfooAdapter
from evaltrust.adapters.registry import detect_adapter

_TESTS_DIR = Path(__file__).parent
_REPO_ROOT = _TESTS_DIR.parent


def _load(path):
    return json.loads(Path(path).read_text())


DETAILS = _load(_TESTS_DIR / "fixtures" / "lighteval_details_agieval_aqua_rat.json")
AGGREGATE = _load(_TESTS_DIR / "fixtures" / "lighteval_results_aggregate.json")

# Documented underscore-prefixed shape (public HF docs); not the authentic
# recent Parquet fixture, which uses unprefixed column names.
DOCUMENTED = [
    {
        "__doc__": {
            "query": "What is 2+2?",
            "choices": ["3", "4"],
            "gold_index": 1,
            "id": "d1",
            "task_name": "gsm8k|0",
        },
        "__metric__": {"em": 1.0, "maj@8": 0.0},
        "__model_response__": {"text": ["4"]},
    },
    {
        "__doc__": {
            "query": "What is 3+3?",
            "choices": ["5", "6"],
            "gold_index": 1,
            "id": "d2",
            "task_name": "gsm8k|0",
        },
        "__metric__": {"em": 0.0, "maj@8": 0.0},
        "__model_response__": {"text": ["5"]},
    },
]


# ---------------------------------------------------------------------------
# Detection — authentic fixture
# ---------------------------------------------------------------------------


def test_lighteval_detects_the_real_details_fixture():
    assert LightevalAdapter().detect(DETAILS)


def test_lighteval_detects_aggregate_results_structurally():
    assert LightevalAdapter().detect(AGGREGATE)


def test_lighteval_detection_is_independent_of_filename():
    # Same payload loaded from differently named paths must detect identically.
    raw = copy.deepcopy(DETAILS)
    assert LightevalAdapter().detect(raw)


def test_lighteval_does_not_detect_generic_similar_json():
    generic = {"results": {"task|0": {"em": 0.5, "em_stderr": 0.1}}}
    assert not LightevalAdapter().detect(generic)


def test_lighteval_does_not_detect_native_nested():
    native = {"models": ["A"], "examples": [{"id": "q1", "scores": {"A": 1}}]}
    assert not LightevalAdapter().detect(native)


def test_lighteval_does_not_detect_promptfoo():
    promptfoo = {
        "results": {"results": [{"provider": "gpt", "testIdx": 0, "score": 1}],
                    "table": {"head": {"prompts": []}}},
        "version": 3,
    }
    assert not LightevalAdapter().detect(promptfoo)


def test_lighteval_does_not_detect_helm_per_instance_stats():
    helm = _load(_TESTS_DIR / "fixtures" / "helm_per_instance_stats.json")
    assert not LightevalAdapter().detect(helm)


def test_lighteval_does_not_detect_inspect_log():
    inspect = _load(_TESTS_DIR / "fixtures" / "inspect_log.json")
    assert not LightevalAdapter().detect(inspect)


def test_lighteval_does_not_false_positive_on_other_fixtures():
    adapter = LightevalAdapter()
    files = list((_TESTS_DIR / "fixtures").glob("*.json")) + \
        list((_REPO_ROOT / "examples").glob("*.json"))
    for f in files:
        raw = _load(f)
        detected = adapter.detect(raw)
        if f.name.startswith("lighteval_"):
            assert detected, f.name
        else:
            assert not detected, f.name


# ---------------------------------------------------------------------------
# Conversion — authentic fixture
# ---------------------------------------------------------------------------


def test_lighteval_parses_the_real_details_fixture():
    data = LightevalAdapter().parse(DETAILS)
    assert data.source_format == "lighteval"
    # Details rows alone have no model field; label defaults like HELM.
    assert data.models == ["model"]
    assert data.n_examples == 3
    assert data.examples[0].id == "agieval:aqua-rat|0:44"
    assert data.examples[0].scores["model"] == 0.0


def test_lighteval_suite_from_real_fixture():
    suite = LightevalAdapter().parse_suite(DETAILS)
    assert list(suite) == ["acc"]
    assert suite["acc"].n_examples == 3


def test_lighteval_preserves_example_ids_deterministically():
    data = LightevalAdapter().parse(DETAILS)
    assert [ex.id for ex in data.examples] == [
        "agieval:aqua-rat|0:44",
        "agieval:aqua-rat|0:9",
        "agieval:aqua-rat|0:73",
    ]


# ---------------------------------------------------------------------------
# Multiple metrics
# ---------------------------------------------------------------------------


def test_lighteval_preserves_multiple_metric_names():
    raw = [
        {
            "doc": {
                "query": "Q?",
                "choices": ["A", "B"],
                "gold_index": 0,
                "id": "ex1",
                "task_name": "gsm8k|0",
            },
            "metric": {"em": 1.0, "maj@8": 0.0},
            "model_response": {"text": ["answer"]},
        },
    ]
    suite = LightevalAdapter().parse_suite(raw)
    assert set(suite.keys()) == {"em", "maj@8"}
    assert suite["em"].examples[0].scores["model"] == 1.0
    assert suite["maj@8"].examples[0].scores["model"] == 0.0


def test_lighteval_documented_shape_preserves_multiple_metrics():
    suite = LightevalAdapter().parse_suite(DOCUMENTED)
    assert set(suite.keys()) == {"em", "maj@8"}
    assert [ex.id for ex in suite["em"].examples] == ["gsm8k|0:d1", "gsm8k|0:d2"]
    assert [ex.scores["model"] for ex in suite["em"].examples] == [1.0, 0.0]
    assert [ex.scores["model"] for ex in suite["maj@8"].examples] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Documented underscore-prefixed aliases
# ---------------------------------------------------------------------------


def test_lighteval_detects_and_parses_documented_underscore_shape():
    a = LightevalAdapter()
    assert a.detect(DOCUMENTED)
    data = a.parse(DOCUMENTED)
    assert data.source_format == "lighteval"
    assert data.n_examples == 2
    assert data.examples[0].id == "gsm8k|0:d1"
    assert data.examples[0].scores["model"] == 1.0


def test_lighteval_documented_shape_without_model_response():
    raw = copy.deepcopy(DOCUMENTED[0])
    del raw["__model_response__"]
    data = LightevalAdapter().parse([raw])
    assert data.n_examples == 1
    assert data.examples[0].id == "gsm8k|0:d1"


def test_lighteval_equal_aliases_are_accepted():
    doc = {
        "query": "Q?",
        "choices": ["A"],
        "gold_index": 0,
        "id": "same",
        "task_name": "t|0",
    }
    raw = [{
        "doc": doc,
        "__doc__": copy.deepcopy(doc),
        "metric": {"acc": 1.0},
        "__metric__": {"acc": 1.0},
    }]
    data = LightevalAdapter().parse(raw)
    assert data.n_examples == 1
    assert data.examples[0].id == "t|0:same"


def test_lighteval_conflicting_doc_aliases_are_not_silently_accepted():
    raw = [
        {
            "doc": {
                "query": "Q?",
                "choices": ["A"],
                "gold_index": 0,
                "id": "from-doc",
                "task_name": "t|0",
            },
            "__doc__": {
                "query": "Q?",
                "choices": ["A"],
                "gold_index": 0,
                "id": "from-underscore",
                "task_name": "t|0",
            },
            "metric": {"acc": 1.0},
        },
        {
            "doc": {
                "query": "Q2?",
                "choices": ["B"],
                "gold_index": 0,
                "id": "ok",
                "task_name": "t|0",
            },
            "metric": {"acc": 0.0},
        },
    ]
    assert LightevalAdapter().detect(raw)
    data = LightevalAdapter().parse(raw)
    assert [ex.id for ex in data.examples] == ["t|0:ok"]
    assert data.metadata["skipped_rows"] == 1


def test_lighteval_does_not_detect_generic_json_with_one_similar_key():
    # A lone similarly named key without the detail fingerprint is not enough.
    assert not LightevalAdapter().detect([{"__doc__": {"text": "not a Doc"}}])
    assert not LightevalAdapter().detect([{"doc": {"query": "only query"}}])
    assert not LightevalAdapter().detect([{"__metric__": {"acc": 1.0}}])
    assert not LightevalAdapter().detect({"__results__": {"task|0": {"em": 1.0}}})


def test_detect_routes_documented_shape_before_generic():
    assert detect_adapter(DOCUMENTED).source_format == "lighteval"
    assert GenericRecordsAdapter().detect(DOCUMENTED)


# ---------------------------------------------------------------------------
# Optional fields and supported variants
# ---------------------------------------------------------------------------


def test_lighteval_accepts_details_without_model_response():
    raw = copy.deepcopy(DETAILS[0])
    del raw["model_response"]
    data = LightevalAdapter().parse([raw])
    assert data.n_examples == 1


def test_lighteval_accepts_task_grouped_dict_export():
    grouped = {"agieval:aqua-rat|0": DETAILS}
    data = LightevalAdapter().parse(grouped)
    assert data.n_examples == 3


def test_lighteval_accepts_config_general_wrapped_details_list():
    wrapped = {
        "config_general": {"model_name": "my-model", "lighteval_sha": "abc"},
        "details": DETAILS,
    }
    data = LightevalAdapter().parse(wrapped)
    assert data.models == ["my-model"]
    assert data.n_examples == 3


def test_lighteval_optional_doc_fields_absent_still_parse():
    raw = {
        "doc": {
            "query": "Q?",
            "choices": ["A"],
            "gold_index": 0,
            "id": "only-required",
        },
        "metric": {"acc": 1.0},
    }
    data = LightevalAdapter().parse([raw])
    assert data.examples[0].id == "only-required"
    assert data.examples[0].scores["model"] == 1.0


# ---------------------------------------------------------------------------
# Aggregate-only and malformed input
# ---------------------------------------------------------------------------


def test_lighteval_aggregate_results_raise_without_per_example_rows():
    with pytest.raises(ValueError, match="aggregate"):
        LightevalAdapter().parse(AGGREGATE)


def test_lighteval_malformed_detail_missing_id_is_counted():
    raw = copy.deepcopy(DETAILS)
    del raw[1]["doc"]["id"]
    data = LightevalAdapter().parse(raw)
    assert data.n_examples == 2
    assert data.metadata["skipped_rows"] == 1


def test_lighteval_malformed_metric_structure_skipped():
    raw = [
        {
            "doc": {"query": "Q", "choices": ["A"], "gold_index": 0, "id": "1"},
            "metric": "not-a-dict",
        },
        {
            "doc": {"query": "Q2", "choices": ["B"], "gold_index": 0, "id": "2"},
            "metric": {"acc": 1.0},
        },
    ]
    data = LightevalAdapter().parse(raw)
    assert data.n_examples == 1
    assert data.metadata["skipped_rows"] == 1


def test_lighteval_null_and_non_numeric_metrics_not_treated_as_scores():
    raw = [
        {
            "doc": {"query": "Q", "choices": ["A"], "gold_index": 0, "id": "1"},
            "metric": {"acc": None, "notes": "text", "acc_stderr": 0.1},
        },
        {
            "doc": {"query": "Q2", "choices": ["B"], "gold_index": 0, "id": "2"},
            "metric": {"acc": 0.5},
        },
    ]
    data = LightevalAdapter().parse(raw)
    assert data.n_examples == 1
    assert data.metadata["skipped_rows"] == 1


def test_lighteval_unsupported_shape_not_detected():
    assert not LightevalAdapter().detect({"doc": {"query": "x"}, "metric": {}})


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_detect_routes_lighteval_details_before_generic():
    assert detect_adapter(DETAILS).source_format == "lighteval"


def test_detect_aggregate_lighteval_raises_specific_error_not_unknown_format():
    adapter = detect_adapter(AGGREGATE)
    assert adapter.source_format == "lighteval"
    with pytest.raises(ValueError, match="aggregate"):
        adapter.parse(AGGREGATE)


def test_no_earlier_adapter_claims_lighteval_details():
    assert not PromptfooAdapter().detect(DETAILS)
    assert not DeepEvalAdapter().detect(DETAILS)
    assert not InspectAdapter().detect(DETAILS)
    assert not HelmAdapter().detect(DETAILS)
    assert not NativeNestedAdapter().detect(DETAILS)
    # Generic would claim any list of dicts; Lighteval must win in the registry.
    assert GenericRecordsAdapter().detect(DETAILS)
    assert detect_adapter(DETAILS).source_format == "lighteval"
