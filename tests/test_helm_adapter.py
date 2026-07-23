"""Tests for the HELM per-instance results adapter.

HELM writes ``per_instance_stats.json`` — a list of per-instance blocks where
each entry carries an ``instance_id`` and a ``stats`` list.  Each stat object
has a ``name`` sub-object (with a ``name`` field for the metric) and a ``mean``
value.  One HELM run = one model, so two files are compared with
``evaltrust audit runA.json runB.json``.

Detection is structural (``instance_id`` + nested ``name`` object in stats),
never by file name.  This test module checks:
  - detection / no-false-positives against every other fixture
  - parsing: exact_match is the primary metric on the single-audit path
  - parsing: quality metrics only on the suite path (bookkeeping filtered)
  - skip-and-count for unparsable stat entries
  - missing instance_id counted as a skipped row (not silently indexed)
  - end-to-end auto-detection via the registry
  - no false positives against any other adapter or file fixture
"""

import json
from pathlib import Path

import pytest

from evaltrust.adapters.helm import HelmAdapter, _BOOKKEEPING_STATS
from evaltrust.adapters.registry import detect_adapter

_TESTS_DIR = Path(__file__).parent
_REPO_ROOT = _TESTS_DIR.parent


def _load(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Representative fixture (as close to a real HELM per_instance_stats.json as
# we can craft without running HELM — based on verified file layout from the
# HELM source and the issue's description of the shape).
# ---------------------------------------------------------------------------

HELM = _load(_TESTS_DIR / "fixtures" / "helm_per_instance_stats.json")

# A minimal in-code fixture — two instances, two quality metrics each.
HELM_MINIMAL = [
    {
        "instance_id": "id0",
        "stats": [
            {"name": {"name": "exact_match", "split": "test"}, "mean": 1.0},
            {"name": {"name": "quasi_exact_match", "split": "test"}, "mean": 1.0},
        ],
    },
    {
        "instance_id": "id1",
        "stats": [
            {"name": {"name": "exact_match", "split": "test"}, "mean": 0.0},
            {"name": {"name": "quasi_exact_match", "split": "test"}, "mean": 1.0},
        ],
    },
]

# Fixture that mixes quality metrics with the bookkeeping stats a real HELM run
# emits — num_trials, num_prompt_tokens, finish_reason_stop, etc.
HELM_WITH_BOOKKEEPING = [
    {
        "instance_id": "id0",
        "stats": [
            {"name": {"name": "exact_match"}, "mean": 1.0},
            {"name": {"name": "num_trials"}, "mean": 1.0},
            {"name": {"name": "num_prompt_tokens"}, "mean": 312.0},
            {"name": {"name": "finish_reason_stop"}, "mean": 1.0},
            {"name": {"name": "finish_reason_length"}, "mean": 0.0},
        ],
    },
    {
        "instance_id": "id1",
        "stats": [
            {"name": {"name": "exact_match"}, "mean": 0.0},
            {"name": {"name": "num_trials"}, "mean": 1.0},
            {"name": {"name": "num_prompt_tokens"}, "mean": 298.0},
            {"name": {"name": "finish_reason_stop"}, "mean": 1.0},
            {"name": {"name": "finish_reason_length"}, "mean": 0.0},
        ],
    },
]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_helm_detects_minimal_fixture():
    assert HelmAdapter().detect(HELM_MINIMAL)


def test_helm_detects_the_real_fixture():
    assert HelmAdapter().detect(HELM)


def test_helm_does_not_detect_a_plain_list_of_records():
    raw = [{"id": "q1", "model": "A", "score": 1}]
    assert not HelmAdapter().detect(raw)


def test_helm_does_not_detect_a_list_without_instance_id():
    raw = [{"stats": [{"name": {"name": "exact_match"}, "mean": 1.0}]}]
    assert not HelmAdapter().detect(raw)


def test_helm_does_not_detect_a_list_without_nested_name_object():
    # stats is present but the name is a plain string, not a dict — not HELM
    raw = [{"instance_id": "id0", "stats": [{"name": "exact_match", "mean": 1.0}]}]
    assert not HelmAdapter().detect(raw)


def test_helm_does_not_detect_a_plain_dict():
    assert not HelmAdapter().detect({"instance_id": "id0", "stats": []})


def test_helm_does_not_detect_an_empty_list():
    assert not HelmAdapter().detect([])


# ---------------------------------------------------------------------------
# Parsing: single-audit path (primary metric = exact_match)
# ---------------------------------------------------------------------------

def test_helm_parse_returns_exact_match_as_primary_metric():
    data = HelmAdapter().parse(HELM_MINIMAL)
    assert data.source_format == "helm"
    (model,) = data.models
    assert data.n_examples == 2
    # exact_match is the preferred correctness metric
    assert data.examples[0].scores[model] == 1.0
    assert data.examples[1].scores[model] == 0.0


def test_helm_parse_real_fixture_exact_match_scores():
    data = HelmAdapter().parse(HELM)
    assert data.source_format == "helm"
    (model,) = data.models
    assert data.n_examples == 3
    scores = [ex.scores[model] for ex in data.examples]
    assert scores == [1.0, 0.0, 0.0]


def test_helm_parse_still_picks_exact_match_when_bookkeeping_present():
    # parse() must still find exact_match even when bookkeeping stats exist.
    data = HelmAdapter().parse(HELM_WITH_BOOKKEEPING)
    assert data.n_examples == 2
    scores = [ex.scores["model"] for ex in data.examples]
    assert scores == [1.0, 0.0]


def test_helm_parse_falls_back_to_quasi_exact_match_when_no_exact_match():
    raw = [
        {
            "instance_id": "id0",
            "stats": [
                {"name": {"name": "quasi_exact_match"}, "mean": 0.5},
            ],
        },
        {
            "instance_id": "id1",
            "stats": [
                {"name": {"name": "quasi_exact_match"}, "mean": 1.0},
            ],
        },
    ]
    data = HelmAdapter().parse(raw)
    scores = [ex.scores["model"] for ex in data.examples]
    assert scores == [0.5, 1.0]


def test_helm_parse_falls_back_to_first_metric_when_no_known_correctness_metric():
    raw = [
        {
            "instance_id": "id0",
            "stats": [{"name": {"name": "some_custom_metric"}, "mean": 0.8}],
        }
    ]
    data = HelmAdapter().parse(raw)
    assert data.n_examples == 1
    assert data.examples[0].scores["model"] == pytest.approx(0.8)


def test_helm_parse_uses_generic_model_name():
    data = HelmAdapter().parse(HELM_MINIMAL)
    assert data.models == ["model"]


# ---------------------------------------------------------------------------
# Parsing: suite path — bookkeeping stats must be filtered out
# ---------------------------------------------------------------------------

def test_helm_parse_suite_returns_only_quality_metrics():
    suite = HelmAdapter().parse_suite(HELM_MINIMAL)
    assert set(suite.keys()) == {"exact_match", "quasi_exact_match"}


def test_helm_parse_suite_filters_num_trials():
    suite = HelmAdapter().parse_suite(HELM_WITH_BOOKKEEPING)
    assert "num_trials" not in suite


def test_helm_parse_suite_filters_num_prompt_tokens():
    suite = HelmAdapter().parse_suite(HELM_WITH_BOOKKEEPING)
    assert "num_prompt_tokens" not in suite


def test_helm_parse_suite_filters_all_finish_reason_variants():
    suite = HelmAdapter().parse_suite(HELM_WITH_BOOKKEEPING)
    assert "finish_reason_stop" not in suite
    assert "finish_reason_length" not in suite


def test_helm_parse_suite_keeps_exact_match_after_filtering():
    suite = HelmAdapter().parse_suite(HELM_WITH_BOOKKEEPING)
    assert "exact_match" in suite
    scores = [ex.scores["model"] for ex in suite["exact_match"].examples]
    assert scores == [1.0, 0.0]


def test_helm_parse_suite_all_known_bookkeeping_stats_are_filtered():
    """Every name in _BOOKKEEPING_STATS must be absent from parse_suite output."""
    raw = [
        {
            "instance_id": "id0",
            "stats": (
                [{"name": {"name": "exact_match"}, "mean": 1.0}]
                + [{"name": {"name": s}, "mean": 1.0} for s in _BOOKKEEPING_STATS]
            ),
        }
    ]
    suite = HelmAdapter().parse_suite(raw)
    for bk_stat in _BOOKKEEPING_STATS:
        assert bk_stat not in suite, f"{bk_stat!r} should be filtered from suite"
    assert "exact_match" in suite


def test_helm_parse_suite_real_fixture_no_bookkeeping():
    suite = HelmAdapter().parse_suite(HELM)
    assert "num_trials" not in suite
    assert "exact_match" in suite
    assert "quasi_exact_match" in suite


def test_helm_parse_suite_each_metric_has_correct_source_format():
    suite = HelmAdapter().parse_suite(HELM_MINIMAL)
    for data in suite.values():
        assert data.source_format == "helm"


# ---------------------------------------------------------------------------
# Skip-and-count: unparsable entries must not crash; they must be counted
# ---------------------------------------------------------------------------

def test_helm_skips_and_counts_stat_with_non_numeric_mean():
    raw = [
        {
            "instance_id": "id0",
            "stats": [
                {"name": {"name": "exact_match"}, "mean": 1.0},         # good
                {"name": {"name": "exact_match"}, "mean": "banana"},     # bad mean
            ],
        },
        {
            "instance_id": "id1",
            "stats": [{"name": {"name": "exact_match"}, "mean": 0.0}],  # good
        },
    ]
    data = HelmAdapter().parse(raw)
    assert data.n_examples == 2
    assert data.metadata["skipped_rows"] == 1


def test_helm_skips_and_counts_stat_with_none_mean():
    raw = [
        {
            "instance_id": "id0",
            "stats": [
                {"name": {"name": "exact_match"}, "mean": None},
            ],
        },
        {
            "instance_id": "id1",
            "stats": [{"name": {"name": "exact_match"}, "mean": 1.0}],
        },
    ]
    # id0 has all-bad stats → counts as 1 skipped instance (not per-stat)
    data = HelmAdapter().parse(raw)
    assert data.n_examples == 1
    assert data.metadata["skipped_rows"] == 1


def test_helm_skips_and_counts_stat_with_no_name_object():
    raw = [
        {
            "instance_id": "id0",
            "stats": [
                {"name": "exact_match", "mean": 1.0},   # name is a string, not a dict
            ],
        },
        {
            "instance_id": "id1",
            "stats": [{"name": {"name": "exact_match"}, "mean": 0.5}],
        },
    ]
    data = HelmAdapter().parse(raw)
    assert data.n_examples == 1
    assert data.metadata["skipped_rows"] == 1


def test_helm_counts_only_bad_stats_when_instance_has_some_good_ones():
    raw = [
        {
            "instance_id": "id0",
            "stats": [
                {"name": {"name": "exact_match"}, "mean": 1.0},   # good
                {"name": {"name": "f1_score"}, "mean": None},      # bad → counted
                {"name": {"name": "rouge_l"}, "mean": "banana"},   # bad → counted
            ],
        }
    ]
    data = HelmAdapter().parse(raw)
    assert data.n_examples == 1
    assert data.examples[0].scores["model"] == 1.0
    assert data.metadata["skipped_rows"] == 2  # f1_score + rouge_l, not the instance


def test_helm_skips_entry_with_missing_stats_list():
    raw = [
        {"instance_id": "id0"},                                                     # no stats
        {"instance_id": "id1", "stats": [{"name": {"name": "exact_match"}, "mean": 1.0}]},
    ]
    data = HelmAdapter().parse(raw)
    assert data.n_examples == 1
    assert data.metadata["skipped_rows"] == 1


def test_helm_skips_entry_with_missing_instance_id():
    """Missing instance_id must be skip+counted, not silently given an index ID."""
    raw = [
        {
            # no instance_id — synthetic index "0" could collide with a real "0"
            "stats": [{"name": {"name": "exact_match"}, "mean": 1.0}],
        },
        {
            "instance_id": "id1",
            "stats": [{"name": {"name": "exact_match"}, "mean": 0.5}],
        },
    ]
    data = HelmAdapter().parse(raw)
    assert data.n_examples == 1          # only the valid entry kept
    assert data.metadata["skipped_rows"] == 1
    assert data.examples[0].id == "id1"  # the good entry is present, not the bad one


def test_helm_raises_when_nothing_is_parsable():
    raw = [
        {"instance_id": "id0", "stats": [{"name": {"name": "exact_match"}, "mean": None}]},
    ]
    with pytest.raises(ValueError, match="No parsable per-instance stats"):
        HelmAdapter().parse(raw)


def test_helm_metadata_skipped_rows_is_zero_on_clean_input():
    data = HelmAdapter().parse(HELM_MINIMAL)
    assert data.metadata["skipped_rows"] == 0


# ---------------------------------------------------------------------------
# Auto-detection routing via registry
# ---------------------------------------------------------------------------

def test_helm_auto_detected_by_registry():
    adapter = detect_adapter(HELM_MINIMAL)
    assert adapter.source_format == "helm"


def test_helm_real_fixture_auto_detected():
    adapter = detect_adapter(HELM)
    assert adapter.source_format == "helm"


# ---------------------------------------------------------------------------
# No false positives: HELM must not claim other adapters' fixtures
# ---------------------------------------------------------------------------

from evaltrust.adapters.deepeval import DeepEvalAdapter
from evaltrust.adapters.generic import GenericRecordsAdapter, NativeNestedAdapter
from evaltrust.adapters.inspect_ai import InspectAdapter
from evaltrust.adapters.langfuse import LangfuseAdapter
from evaltrust.adapters.langsmith import LangSmithAdapter
from evaltrust.adapters.openevals import OpenEvalsAdapter
from evaltrust.adapters.promptfoo import PromptfooAdapter
from evaltrust.adapters.ragas import RagasAdapter

_PROMPTFOO = {
    "results": {
        "results": [
            {"provider": {"id": "openai:gpt-4"}, "testIdx": 0, "score": 1, "success": True},
        ],
        "table": {"head": {"prompts": []}},
    },
    "version": 3,
}
_DEEPEVAL = {
    "test_results": [
        {"name": "t0", "success": True,
         "metrics_data": [{"name": "Correctness", "score": 0.9, "success": True}]},
    ],
}
_NATIVE = {
    "models": ["A"],
    "examples": [{"id": "q1", "scores": {"A": 1}}],
}
_LONG = [{"id": "q1", "model": "gpt", "score": 1}]
_OPENEVALS = [{"key": "correctness", "score": 1.0, "input": "q1"}]
_INSPECT = {
    "version": 2,
    "eval": {"eval_id": "e1", "model": "openai/gpt-4o-mini"},
    "samples": [{"id": 1, "scores": {"match": {"value": "C"}}}],
}
_LANGSMITH = [
    {"id": "r1", "reference_example_id": "ex1",
     "feedback_stats": {"correctness": {"n": 1, "avg": 1.0}}},
]
_RAGAS = [
    {"user_input": "q", "retrieved_contexts": ["ctx"],
     "response": "Paris", "reference": "Paris",
     "faithfulness": 1.0, "answer_relevancy": 0.95, "context_precision": 0.9},
]


def test_helm_does_not_claim_promptfoo():
    assert not HelmAdapter().detect(_PROMPTFOO)


def test_helm_does_not_claim_deepeval():
    assert not HelmAdapter().detect(_DEEPEVAL)


def test_helm_does_not_claim_native_nested():
    assert not HelmAdapter().detect(_NATIVE)


def test_helm_does_not_claim_generic_long():
    assert not HelmAdapter().detect(_LONG)


def test_helm_does_not_claim_openevals():
    assert not HelmAdapter().detect(_OPENEVALS)


def test_helm_does_not_claim_inspect():
    assert not HelmAdapter().detect(_INSPECT)


def test_helm_does_not_claim_langsmith():
    assert not HelmAdapter().detect(_LANGSMITH)


def test_helm_does_not_claim_ragas():
    assert not HelmAdapter().detect(_RAGAS)


def test_helm_does_not_false_positive_on_any_existing_json_fixture():
    a = HelmAdapter()
    fixture_dir = _TESTS_DIR / "fixtures"
    example_dir = _REPO_ROOT / "examples"
    files = list(fixture_dir.glob("*.json")) + list(example_dir.glob("*.json"))
    for f in files:
        raw = _load(f)
        detected = a.detect(raw)
        if f.name == "helm_per_instance_stats.json":
            assert detected, f.name
        else:
            assert not detected, f.name


# ---------------------------------------------------------------------------
# No earlier adapter should steal a HELM file
# ---------------------------------------------------------------------------

def test_no_earlier_adapter_claims_a_helm_file():
    for adapter_cls in (
        PromptfooAdapter,
        DeepEvalAdapter,
        OpenEvalsAdapter,
        InspectAdapter,
        LangfuseAdapter,
        LangSmithAdapter,
        RagasAdapter,
        NativeNestedAdapter,
    ):
        assert not adapter_cls().detect(HELM_MINIMAL), adapter_cls.__name__


# ---------------------------------------------------------------------------
# End-to-end: load_suite picks up quality metrics, filters bookkeeping
# ---------------------------------------------------------------------------

def test_helm_end_to_end_suite_ingest():
    from evaltrust.core.ingest import load_suite
    path = str(_TESTS_DIR / "fixtures" / "helm_per_instance_stats.json")
    suite = load_suite(path)
    assert "exact_match" in suite
    assert "num_trials" not in suite       # bookkeeping filtered end-to-end
    assert all(data.source_format == "helm" for data in suite.values())
    assert suite["exact_match"].n_examples == 3
