"""Tests for format adapters and auto-detection.

Each adapter answers two questions: does this raw object look like my format
(detect), and if so, map it to canonical EvalData (parse). Detection is by
structural fingerprint, never by file name.
"""

import pytest

from evaltrust.adapters.deepeval import DeepEvalAdapter
from evaltrust.adapters.generic import GenericRecordsAdapter, NativeNestedAdapter
from evaltrust.adapters.promptfoo import PromptfooAdapter
from evaltrust.adapters.registry import detect_adapter, UnknownFormatError


# ---------------------------------------------------------------------------
# Native nested JSON (structured programmatic output)
# ---------------------------------------------------------------------------

NATIVE = {
    "models": ["A", "B"],
    "examples": [
        {"id": "q1", "scores": {"A": 1, "B": 0}},
        {"id": "q2", "scores": {"A": 0, "B": 1},
         "runs": {"A": [0, 1], "B": [1, 1]}},
    ],
}


def test_native_nested_detects_and_parses():
    a = NativeNestedAdapter()
    assert a.detect(NATIVE)
    data = a.parse(NATIVE)
    assert data.models == ["A", "B"]
    assert data.n_examples == 2
    assert data.examples[1].runs == {"A": [0.0, 1.0], "B": [1.0, 1.0]}


def test_native_nested_rejects_a_plain_list():
    assert not NativeNestedAdapter().detect([{"id": 1, "model": "A", "score": 1}])


# ---------------------------------------------------------------------------
# Generic records: long format (one row per model) and wide format
# ---------------------------------------------------------------------------

LONG = [
    {"id": "q1", "model": "gpt", "score": 1},
    {"id": "q1", "model": "claude", "score": 0},
    {"id": "q2", "model": "gpt", "score": 0},
    {"id": "q2", "model": "claude", "score": 1},
]

WIDE = [
    {"question": "q1", "gpt": 1, "claude": 0},
    {"question": "q2", "gpt": 0, "claude": 1},
]


def test_generic_long_records_parse():
    a = GenericRecordsAdapter()
    assert a.detect(LONG)
    data = a.parse(LONG)
    assert set(data.models) == {"gpt", "claude"}
    assert data.n_examples == 2


def test_generic_wide_records_parse():
    data = GenericRecordsAdapter().parse(WIDE)
    assert set(data.models) == {"gpt", "claude"}
    assert data.examples[0].scores["gpt"] == 1.0


def test_generic_records_wrapped_in_results_key():
    a = GenericRecordsAdapter()
    wrapped = {"results": LONG}
    assert a.detect(wrapped)
    assert a.parse(wrapped).n_examples == 2


# ---------------------------------------------------------------------------
# Promptfoo (the natural multi-provider comparison format)
# ---------------------------------------------------------------------------

PROMPTFOO = {
    "results": {
        "results": [
            {"provider": {"id": "openai:gpt-4"}, "testIdx": 0, "score": 1, "success": True},
            {"provider": {"id": "anthropic:claude"}, "testIdx": 0, "score": 0, "success": False},
            {"provider": {"id": "openai:gpt-4"}, "testIdx": 1, "score": 0, "success": False},
            {"provider": {"id": "anthropic:claude"}, "testIdx": 1, "score": 1, "success": True},
        ],
        "table": {"head": {"prompts": []}},
    },
    "version": 3,
}


def test_promptfoo_detects_and_parses_providers_as_models():
    a = PromptfooAdapter()
    assert a.detect(PROMPTFOO)
    data = a.parse(PROMPTFOO)
    assert set(data.models) == {"openai:gpt-4", "anthropic:claude"}
    assert data.n_examples == 2


def test_promptfoo_falls_back_to_success_when_no_score():
    raw = {"results": {"results": [
        {"provider": "m1", "testIdx": 0, "success": True},
        {"provider": "m2", "testIdx": 0, "success": False},
    ]}}
    data = PromptfooAdapter().parse(raw)
    assert data.examples[0].scores == {"m1": 1.0, "m2": 0.0}


# ---------------------------------------------------------------------------
# DeepEval (single model per run — paired via two files)
# ---------------------------------------------------------------------------

DEEPEVAL_SNAKE = {
    "test_results": [
        {"name": "t0", "success": True,
         "metrics_data": [{"name": "Correctness", "score": 0.9, "success": True}]},
        {"name": "t1", "success": False,
         "metrics_data": [{"name": "Correctness", "score": 0.3, "success": False}]},
    ],
}

DEEPEVAL_CAMEL = {
    "testCases": [
        {"name": "t0", "success": True, "metricsData": [{"name": "M", "score": 1.0}]},
        {"name": "t1", "success": True, "metricsData": [{"name": "M", "score": 0.8}]},
    ],
    "hyperparameters": {"model": "gpt-4"},
}


def test_deepeval_snake_case_detects_and_parses_pass_fail():
    a = DeepEvalAdapter()
    assert a.detect(DEEPEVAL_SNAKE)
    data = a.parse(DEEPEVAL_SNAKE)
    assert data.n_examples == 2
    # One model; scores come from per-case success.
    (model,) = data.models
    assert data.examples[0].scores[model] == 1.0
    assert data.examples[1].scores[model] == 0.0


def test_deepeval_uses_hyperparameter_model_name_when_present():
    data = DeepEvalAdapter().parse(DEEPEVAL_CAMEL)
    assert data.models == ["gpt-4"]


def test_deepeval_does_not_grab_promptfoo():
    assert not DeepEvalAdapter().detect(PROMPTFOO)


# ---------------------------------------------------------------------------
# Auto-detection routing
# ---------------------------------------------------------------------------

def test_detect_routes_promptfoo_before_generic():
    assert detect_adapter(PROMPTFOO).source_format == "promptfoo"


def test_detect_routes_native_nested():
    assert detect_adapter(NATIVE).source_format == "native"


def test_detect_routes_generic_records():
    assert detect_adapter(LONG).source_format == "generic"


def test_detect_routes_deepeval():
    assert detect_adapter(DEEPEVAL_SNAKE).source_format == "deepeval"


def test_detect_raises_helpful_error_on_unknown_shape():
    with pytest.raises(UnknownFormatError):
        detect_adapter({"totally": "unrecognised"})


# ---------------------------------------------------------------------------
# OpenEvals adapter
# ---------------------------------------------------------------------------

from evaltrust.adapters.openevals import OpenEvalsAdapter

OPENEVALS_SAMPLE = [
    {"key": "correctness", "score": 1.0, "comment": "Correct.", "input": "q1"},
    {"key": "correctness", "score": 0.0, "comment": "Wrong.", "input": "q2"},
    {"key": "correctness", "score": 1.0, "comment": "Correct.", "input": "q3"},
]


def test_openevals_detects():
    assert OpenEvalsAdapter().detect(OPENEVALS_SAMPLE)


def test_openevals_does_not_detect_promptfoo():
    assert not OpenEvalsAdapter().detect({"results": {"results": [{"provider": "gpt"}]}})


def test_openevals_does_not_detect_plain_list():
    assert not OpenEvalsAdapter().detect([{"id": "q1", "model": "A", "score": 1}])


def test_openevals_parses_scores():
    data = OpenEvalsAdapter().parse(OPENEVALS_SAMPLE)
    assert data.n_examples == 3
    assert data.models == ["model"]
    assert data.examples[0].scores["model"] == 1.0
    assert data.examples[1].scores["model"] == 0.0


def test_openevals_boolean_score():
    raw = [
        {"key": "pass", "score": True, "input": "q1"},
        {"key": "pass", "score": False, "input": "q2"},
    ]
    data = OpenEvalsAdapter().parse(raw)
    assert data.examples[0].scores["model"] == 1.0
    assert data.examples[1].scores["model"] == 0.0


def test_openevals_auto_detected_by_registry():
    from evaltrust.adapters.registry import detect_adapter
    adapter = detect_adapter(OPENEVALS_SAMPLE)
    assert adapter.source_format == "openevals"


# ---------------------------------------------------------------------------
# Inspect (UK AISI) .json eval logs
# ---------------------------------------------------------------------------

import json
from pathlib import Path

from evaltrust.adapters.inspect_ai import InspectAdapter

_TESTS_DIR = Path(__file__).parent
_REPO_ROOT = _TESTS_DIR.parent


def _load(path):
    return json.loads(Path(path).read_text())


# A minimal Inspect log, matching the shape of tests/fixtures/inspect_log.json.
INSPECT = {
    "version": 2,
    "status": "success",
    "eval": {"eval_id": "e1", "run_id": "r1", "task": "popularity",
             "model": "openai/gpt-4o-mini"},
    "samples": [
        {"id": 1, "epoch": 1, "scores": {"match": {"value": "C"}}},
        {"id": 2, "epoch": 1, "scores": {"match": {"value": "I"}}},
        {"id": 3, "epoch": 1, "scores": {"match": {"value": "C"}}},
    ],
}


def test_inspect_detects_and_parses_the_real_fixture():
    raw = _load(_TESTS_DIR / "fixtures" / "inspect_log.json")
    a = InspectAdapter()
    assert a.detect(raw)
    data = a.parse(raw)
    assert data.source_format == "inspect"
    assert data.models == ["openai/gpt-4o-mini"]     # model comes from eval.model
    assert data.n_examples == 3
    assert data.examples[0].scores == {"openai/gpt-4o-mini": 1.0}   # "C" -> 1.0
    assert data.examples[1].scores == {"openai/gpt-4o-mini": 0.0}   # "I" -> 0.0


def test_inspect_grade_values_map_like_value_to_float():
    # CORRECT="C"->1, INCORRECT="I"->0, PARTIAL="P"->0.5, NOANSWER="N"->0
    raw = {"eval": {"eval_id": "e", "task": "t", "model": "m"},
           "samples": [
               {"id": "a", "scores": {"s": {"value": "C"}}},
               {"id": "b", "scores": {"s": {"value": "I"}}},
               {"id": "c", "scores": {"s": {"value": "P"}}},
               {"id": "d", "scores": {"s": {"value": "N"}}},
           ]}
    data = InspectAdapter().parse(raw)
    got = [ex.scores["m"] for ex in data.examples]
    assert got == [1.0, 0.0, 0.5, 0.0]


def test_inspect_numeric_and_boolean_values():
    raw = {"eval": {"eval_id": "e", "task": "t", "model": "m"},
           "samples": [
               {"id": "a", "scores": {"s": {"value": 0.75}}},
               {"id": "b", "scores": {"s": {"value": True}}},
               {"id": "c", "scores": {"s": {"value": "yes"}}},
           ]}
    got = [ex.scores["m"] for ex in InspectAdapter().parse(raw).examples]
    assert got == [0.75, 1.0, 1.0]


def test_inspect_multiple_scorers_use_the_first_like_openevals():
    # A dedicated adapter yields a single-metric EvalData (the suite fan-out is
    # only for the generic record path), so, as with the OpenEvals adapter, the
    # first scorer becomes the audited metric.
    raw = {"eval": {"eval_id": "e", "task": "t", "model": "m"},
           "samples": [
               {"id": "a", "scores": {"match": {"value": "C"},
                                      "includes": {"value": "I"}}},
           ]}
    data = InspectAdapter().parse(raw)
    assert data.examples[0].scores == {"m": 1.0}     # "match" (first scorer) -> C -> 1.0


def test_inspect_skips_unscored_values_and_counts_them():
    # A null / non-scalar value, or a malformed (unwrapped) Score entry, is
    # skipped like the CSV path -- not fatal -- and every one is counted.
    raw = {"eval": {"eval_id": "e", "task": "t", "model": "m"},
           "samples": [
               {"id": 1, "scores": {"s": {"value": "C"}}},
               {"id": 2, "scores": {"s": {"value": None, "explanation": "error"}}},
               {"id": 3, "scores": {"s": {"value": ["a", "b"]}}},
               {"id": 4, "scores": {"s": "C"}},           # not a {"value": ...} Score
               {"id": 5, "scores": {"s": {"value": "I"}}},
           ]}
    data = InspectAdapter().parse(raw)
    assert data.n_examples == 2                      # only ids 1 and 5 scored
    assert data.metadata["skipped_rows"] == 3        # null + list + malformed, counted


def test_inspect_detect_requires_a_score_shaped_value():
    # detect() must stay in step with parse(): an empty scores map, or scores
    # keyed model->number (a native record misplaced under "samples"), is not an
    # Inspect log and must not be claimed then rejected.
    a = InspectAdapter()
    assert not a.detect({"eval": {"eval_id": "e", "model": "m"},
                         "samples": [{"id": 1, "scores": {}}]})
    native_under_samples = {"eval": {"model": "m"},
                            "samples": [{"id": 1, "scores": {"m": 0.9}}]}
    assert not a.detect(native_under_samples)


def test_inspect_epochs_become_repeated_runs():
    # Inspect epochs re-run the same sample; repeated (id, model) records become
    # that example's runs (which unlocks the Repeatability check).
    raw = {"eval": {"eval_id": "e", "task": "t", "model": "m"},
           "samples": [
               {"id": 1, "epoch": 1, "scores": {"s": {"value": "C"}}},
               {"id": 1, "epoch": 2, "scores": {"s": {"value": "I"}}},
           ]}
    data = InspectAdapter().parse(raw)
    assert data.n_examples == 1
    assert data.examples[0].runs == {"m": [1.0, 0.0]}
    assert data.examples[0].scores["m"] == 0.5       # mean of the two epochs


def test_inspect_routed_by_registry_before_the_generic_fallback():
    # A generic record adapter would grab the "samples" list; the specific
    # Inspect adapter must win.
    assert detect_adapter(INSPECT).source_format == "inspect"
    assert GenericRecordsAdapter().detect(INSPECT)   # generic *would* have claimed it


def test_generic_records_under_eval_samples_still_route_to_generic():
    # A plain record list nested under eval/samples but WITHOUT Inspect's
    # fingerprint (no eval_id, flat `score` not a `scores` map) must not be
    # hijacked by the Inspect adapter -- it should still parse as generic.
    raw = {"eval": {"model": "gpt-4", "task": "smoke"},
           "samples": [{"id": 1, "model": "gpt-4", "score": 0.9},
                       {"id": 2, "model": "gpt-4", "score": 0.4}]}
    assert not InspectAdapter().detect(raw)
    assert detect_adapter(raw).source_format == "generic"


def test_no_earlier_adapter_claims_an_inspect_log():
    raw = _load(_TESTS_DIR / "fixtures" / "inspect_log.json")
    assert not PromptfooAdapter().detect(raw)
    assert not DeepEvalAdapter().detect(raw)
    assert not OpenEvalsAdapter().detect(raw)
    assert not NativeNestedAdapter().detect(raw)


def test_inspect_does_not_false_positive_on_any_existing_fixture():
    a = InspectAdapter()
    # In-code fixtures from the rest of this module.
    for other in (PROMPTFOO, NATIVE, LONG, WIDE, DEEPEVAL_SNAKE, DEEPEVAL_CAMEL,
                  OPENEVALS_SAMPLE):
        assert not a.detect(other), other
    # Every JSON fixture shipped in tests/fixtures and examples/.
    files = list((_TESTS_DIR / "fixtures").glob("*.json")) + \
        list((_REPO_ROOT / "examples").glob("*.json"))
    for f in files:
        raw = _load(f)
        detected = a.detect(raw)
        if f.name == "inspect_log.json":
            assert detected, f.name          # our own fixture must detect
        else:
            assert not detected, f.name      # nothing else may
