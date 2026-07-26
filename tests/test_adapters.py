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


def test_native_nested_skips_and_counts_unreadable_score():
    # One junk score must not crash the whole file: drop that model's score,
    # keep the example's other scores, and count it so the Data Quality finding
    # reflects the drop (like the CSV and generic record paths).
    raw = {
        "models": ["A", "B"],
        "examples": [
            {"id": "q1", "scores": {"A": 1, "B": 0}},          # clean
            {"id": "q2", "scores": {"A": 0, "B": "banana"}},   # B unreadable
        ],
    }
    data = NativeNestedAdapter().parse(raw)
    assert data.n_examples == 2
    assert data.examples[1].scores == {"A": 0.0}   # B dropped, A kept
    assert data.metadata["skipped_rows"] == 1


def test_native_nested_records_metadata_when_all_scores_are_clean():
    data = NativeNestedAdapter().parse(NATIVE)
    assert data.metadata["skipped_rows"] == 0


def test_native_nested_drops_bad_runs_and_judges_without_counting():
    # A junk value inside runs/judges only gates an optional check, so it drops
    # that block to None and leaves the main scores untouched -- and it is NOT
    # counted as a skipped row (skipped_rows means dropped main scores only).
    raw = {
        "models": ["A", "B"],
        "examples": [
            {"id": "q1", "scores": {"A": 1, "B": 0},
             "runs": {"A": [0, "banana"], "B": [1, 1]},        # A's runs unreadable
             "judges": {"human": {"A": 1, "B": "nope"}}},      # one judge score bad
        ],
    }
    data = NativeNestedAdapter().parse(raw)
    ex = data.examples[0]
    assert ex.scores == {"A": 1.0, "B": 0.0}          # main scores untouched
    assert ex.runs == {"B": [1.0, 1.0]}               # A's bad run list dropped
    assert ex.judges == {"human": {"A": 1.0}}         # only B's bad judge score dropped
    assert data.metadata["skipped_rows"] == 0         # runs/judges are not counted


def test_native_nested_collapses_fully_bad_runs_and_judges_to_none():
    # When every value in a block is unreadable, the whole block collapses to
    # None (nothing left to gate the optional check on), main scores stay intact,
    # and it is still not counted.
    raw = {
        "models": ["A", "B"],
        "examples": [
            {"id": "q1", "scores": {"A": 1, "B": 0},
             "runs": {"A": ["banana"], "B": ["kiwi"]},         # every run list bad
             "judges": {"human": {"A": "banana", "B": "nope"}}},  # every judge score bad
        ],
    }
    data = NativeNestedAdapter().parse(raw)
    ex = data.examples[0]
    assert ex.scores == {"A": 1.0, "B": 0.0}          # main scores untouched
    assert ex.runs is None                            # whole runs block collapsed
    assert ex.judges is None                          # whole judges block collapsed
    assert data.metadata["skipped_rows"] == 0


def test_native_nested_tolerates_malformed_runs_and_judges_structures():
    # Structurally wrong (not just unreadable-value) optional blocks must not
    # abort the parse: a non-iterable run list, or a runs/judges block that isn't
    # a dict, is dropped like any other bad optional data. Main scores survive
    # and structural issues are not counted.
    raw = {
        "models": ["A", "B"],
        "examples": [
            {"id": "q1", "scores": {"A": 1, "B": 0},
             "runs": {"A": 5, "B": [1, 1]},          # A's run list isn't iterable
             "judges": {"human": 7}},                # a judge's value isn't a dict
            {"id": "q2", "scores": {"A": 0, "B": 1},
             "runs": [1, 2, 3],                      # runs block isn't a dict
             "judges": [1, 2]},                      # judges block isn't a dict
        ],
    }
    data = NativeNestedAdapter().parse(raw)           # must not raise
    assert data.n_examples == 2
    assert data.examples[0].scores == {"A": 1.0, "B": 0.0}
    assert data.examples[0].runs == {"B": [1.0, 1.0]}   # A dropped, B kept
    assert data.examples[0].judges is None              # bad judge value -> None
    assert data.examples[1].runs is None                # non-dict runs -> None
    assert data.examples[1].judges is None              # non-dict judges -> None
    assert data.metadata["skipped_rows"] == 0


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


def test_generic_long_records_parse_common_aliases():
    raw = [
        {"sample_id": "q1", "system_name": "gpt", "accuracy": "85%", "skill": "math"},
        {"sample_id": "q1", "system_name": "claude", "accuracy": "good", "skill": "math"},
    ]

    data = GenericRecordsAdapter().parse(raw)

    assert data.models == ["gpt", "claude"]
    assert data.examples[0].id == "q1"
    assert data.examples[0].scores == {"gpt": 0.85, "claude": 1.0}


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


def test_openevals_skips_and_counts_unreadable_score():
    # One junk cell must not sink the whole file: skip it, keep the good rows,
    # and count it so the Data Quality finding reflects the drop (like the
    # Inspect and CSV paths).
    raw = [
        {"key": "correctness", "score": 1.0, "input": "q1"},
        {"key": "correctness", "score": "banana", "input": "q2"},  # unreadable
        {"key": "correctness", "score": 0.0, "input": "q3"},
    ]
    data = OpenEvalsAdapter().parse(raw)
    assert data.n_examples == 2
    assert data.metadata["skipped_rows"] == 1


def test_openevals_counts_missing_score_as_skipped():
    raw = [
        {"key": "correctness", "score": 1.0, "input": "q1"},
        {"key": "correctness", "score": None, "input": "q2"},  # missing
    ]
    data = OpenEvalsAdapter().parse(raw)
    assert data.n_examples == 1
    assert data.metadata["skipped_rows"] == 1


def test_openevals_records_metadata_when_all_rows_are_clean():
    data = OpenEvalsAdapter().parse(OPENEVALS_SAMPLE)
    assert data.metadata["skipped_rows"] == 0


def test_openevals_does_not_merge_distinct_rows_sharing_an_input():
    # Two separate evaluations that happen to share the same input text are
    # distinct examples, not repeated runs of one -- they must not be merged.
    raw = [
        {"key": "correctness", "score": 1.0, "input": "same prompt"},
        {"key": "correctness", "score": 0.0, "input": "same prompt"},
    ]
    data = OpenEvalsAdapter().parse(raw)
    assert data.n_examples == 2


def test_openevals_prefers_explicit_id_field():
    raw = [{"key": "correctness", "score": 1.0, "id": "case-7", "input": "q"}]
    data = OpenEvalsAdapter().parse(raw)
    assert data.examples[0].id == "case-7"


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


def test_inspect_multiple_scorers_use_the_first_on_the_single_audit_path():
    # On the single-audit path, parse() yields a single-metric EvalData -- the
    # first scorer -- as with the OpenEvals adapter. (The suite path fans every
    # scorer out into its own metric; see test_load_suite_inspect_multi_scorer.)
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


def test_inspect_counts_samples_with_no_usable_scores_like_openevals():
    # A real sample whose scores are missing or not a mapping is a dropped row
    # (counted), matching how OpenEvals counts a row with score=None. A non-dict
    # entry is not a sample at all, so -- as in OpenEvals -- it is not counted.
    raw = {"eval": {"eval_id": "e", "task": "t", "model": "m"},
           "samples": [
               {"id": 1, "scores": {"s": {"value": "C"}}},   # scored
               {"id": 2},                                    # no scores -> counted
               {"id": 3, "scores": "oops"},                  # scores not a mapping -> counted
               "garbage",                                    # not a sample -> not counted
               {"id": 4, "scores": {"s": {"value": "I"}}},   # scored
           ]}
    data = InspectAdapter().parse(raw)
    assert data.n_examples == 2                      # ids 1 and 4
    assert data.metadata["skipped_rows"] == 2        # ids 2 and 3; "garbage" not counted


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


# ---------------------------------------------------------------------------
# LangSmith run export (one experiment/model per file — paired via two files)
# ---------------------------------------------------------------------------

from evaltrust.adapters.langsmith import LangSmithAdapter

LANGSMITH = [
    {"id": "r1", "reference_example_id": "ex1",
     "feedback_stats": {"correctness": {"n": 1, "avg": 1.0}}},
    {"id": "r2", "reference_example_id": "ex2",
     "feedback_stats": {"correctness": {"n": 1, "avg": 0.0},
                        "conciseness": {"n": 1, "avg": 0.5}}},
]


def test_langsmith_detects_and_parses_averaging_multiple_metrics():
    a = LangSmithAdapter()
    assert a.detect(LANGSMITH)
    data = a.parse(LANGSMITH)
    assert data.n_examples == 2
    (model,) = data.models
    assert data.examples[0].scores[model] == 1.0
    assert data.examples[1].scores[model] == 0.25   # mean(0.0, 0.5)


def test_langsmith_skips_runs_without_a_reference_example_id():
    raw = LANGSMITH + [{"id": "r3", "reference_example_id": None, "feedback_stats": {}}]
    data = LangSmithAdapter().parse(raw)
    assert data.n_examples == 2


def test_langsmith_raises_when_no_run_has_a_reference_example_id():
    raw = [{"id": "r1", "reference_example_id": None, "feedback_stats": {}}]
    with pytest.raises(ValueError):
        LangSmithAdapter().parse(raw)


def test_langsmith_skips_and_counts_a_run_with_no_usable_avg():
    # A run with a reference_example_id but empty/unusable feedback_stats must
    # not sink the whole export -- skip it and count it, like the CSV/generic/
    # Inspect/OpenEvals adapters already do for a single bad row.
    raw = LANGSMITH + [
        {"id": "r3", "reference_example_id": "ex3", "feedback_stats": {}},
        {"id": "r4", "reference_example_id": "ex4",
         "feedback_stats": {"correctness": {"n": 0, "avg": None}}},
    ]
    data = LangSmithAdapter().parse(raw)
    assert data.n_examples == 2                  # ex1, ex2 only
    assert data.metadata["skipped_rows"] == 2     # ex3, ex4 counted, not dropped silently


def test_langsmith_raises_when_every_run_has_no_usable_avg():
    raw = [
        {"id": "r1", "reference_example_id": "ex1", "feedback_stats": {}},
        {"id": "r2", "reference_example_id": "ex2",
         "feedback_stats": {"correctness": {"n": 0, "avg": None}}},
    ]
    with pytest.raises(ValueError):
        LangSmithAdapter().parse(raw)


def test_langsmith_does_not_false_positive_on_other_fixtures():
    a = LangSmithAdapter()
    for other in (PROMPTFOO, NATIVE, LONG, WIDE, DEEPEVAL_SNAKE, DEEPEVAL_CAMEL,
                  OPENEVALS_SAMPLE, INSPECT):
        assert not a.detect(other), other


def test_no_earlier_adapter_claims_a_langsmith_export():
    assert not PromptfooAdapter().detect(LANGSMITH)
    assert not DeepEvalAdapter().detect(LANGSMITH)
    assert not OpenEvalsAdapter().detect(LANGSMITH)
    assert not InspectAdapter().detect(LANGSMITH)
    assert not NativeNestedAdapter().detect(LANGSMITH)


def test_detect_routes_langsmith():
    assert detect_adapter(LANGSMITH).source_format == "langsmith"


# ---------------------------------------------------------------------------
# Ragas result export (one RAG pipeline per run — paired via two files)
# ---------------------------------------------------------------------------

from evaltrust.adapters.ragas import RagasAdapter

RAGAS = [
    {"user_input": "What is the capital of France?",
     "retrieved_contexts": ["Paris is the capital of France."],
     "response": "Paris", "reference": "Paris",
     "faithfulness": 1.0, "answer_relevancy": 0.95, "context_precision": 0.9},
    {"user_input": "What is the capital of Germany?",
     "retrieved_contexts": ["Berlin is a city in Germany."],
     "response": "Munich", "reference": "Berlin",
     "faithfulness": 0.2, "answer_relevancy": 0.6, "context_precision": 0.7},
]


def test_ragas_detects_and_parses_averaging_multiple_metrics():
    a = RagasAdapter()
    assert a.detect(RAGAS)
    data = a.parse(RAGAS)
    assert data.n_examples == 2
    (model,) = data.models
    assert data.examples[0].scores[model] == pytest.approx((1.0 + 0.95 + 0.9) / 3)
    assert data.examples[1].scores[model] == pytest.approx((0.2 + 0.6 + 0.7) / 3)


def test_ragas_skips_and_counts_a_row_with_no_usable_metric_score():
    raw = RAGAS + [{"user_input": "no metrics here", "response": "?"}]
    data = RagasAdapter().parse(raw)
    assert data.n_examples == 2
    assert data.metadata["skipped_rows"] == 1


def test_ragas_raises_when_no_row_has_a_usable_metric_score():
    raw = [{"user_input": "no metrics here", "response": "?"}]
    with pytest.raises(ValueError):
        RagasAdapter().parse(raw)


def test_ragas_does_not_false_positive_on_other_fixtures():
    a = RagasAdapter()
    for other in (PROMPTFOO, NATIVE, LONG, WIDE, DEEPEVAL_SNAKE, DEEPEVAL_CAMEL,
                  OPENEVALS_SAMPLE, INSPECT, LANGSMITH):
        assert not a.detect(other), other


def test_no_earlier_adapter_claims_a_ragas_export():
    assert not PromptfooAdapter().detect(RAGAS)
    assert not DeepEvalAdapter().detect(RAGAS)
    assert not OpenEvalsAdapter().detect(RAGAS)
    assert not InspectAdapter().detect(RAGAS)
    assert not LangSmithAdapter().detect(RAGAS)
    assert not NativeNestedAdapter().detect(RAGAS)


def test_detect_routes_ragas():
    assert detect_adapter(RAGAS).source_format == "ragas"


# ---------------------------------------------------------------------------
# Langfuse score export (one model/run per file — paired via two files)
# ---------------------------------------------------------------------------

from evaltrust.adapters.langfuse import LangfuseAdapter

LANGFUSE_V3 = _load(_TESTS_DIR / "fixtures" / "langfuse_scores_v3.json")


def test_langfuse_detects_and_parses_the_v3_api_fixture():
    a = LangfuseAdapter()
    assert a.detect(LANGFUSE_V3)

    suite = a.parse_suite(LANGFUSE_V3)
    assert list(suite) == ["correctness", "helpfulness", "quality_label"]
    assert [ex.id for ex in suite["correctness"].examples] == ["trace-1", "trace-2"]
    assert [ex.scores["model"] for ex in suite["correctness"].examples] == [0.9, 0.2]
    assert [ex.scores["model"] for ex in suite["helpfulness"].examples] == [1.0, 0.0]
    assert [ex.scores["model"] for ex in suite["quality_label"].examples] == [1.0, 0.0]
    assert suite["correctness"].metadata["skipped_rows"] == 1  # TEXT score


def test_langfuse_single_audit_path_uses_the_first_metric():
    data = LangfuseAdapter().parse(LANGFUSE_V3)
    assert data.source_format == "langfuse"
    assert [ex.scores["model"] for ex in data.examples] == [0.9, 0.2]


def test_langfuse_accepts_legacy_flat_v2_score_rows():
    raw = [
        {"id": "s1", "traceId": "t1", "observationId": None,
         "name": "correctness", "value": 1.0, "dataType": "NUMERIC"},
        {"id": "s2", "traceId": "t2", "observationId": None,
         "name": "correctness", "value": None, "stringValue": "incorrect",
         "dataType": "CATEGORICAL"},
    ]
    data = LangfuseAdapter().parse(raw)
    assert [ex.id for ex in data.examples] == ["t1", "t2"]
    assert [ex.scores["model"] for ex in data.examples] == [1.0, 0.0]


def test_langfuse_v3_without_subject_has_a_helpful_error():
    raw = {"data": [{
        "id": "s1", "projectId": "p1", "name": "correctness",
        "value": 0.9, "dataType": "NUMERIC", "source": "EVAL",
    }], "meta": {"limit": 50}}
    a = LangfuseAdapter()
    assert a.detect(raw)
    with pytest.raises(ValueError, match="fields=subject"):
        a.parse(raw)


def test_langfuse_skips_and_counts_unusable_score_rows():
    raw = [
        {"id": "s1", "traceId": "t1", "name": "correctness",
         "value": 1.0, "dataType": "NUMERIC"},
        {"id": "s2", "traceId": "t2", "name": "correctness",
         "value": "not-a-score", "dataType": "CATEGORICAL"},
        {"id": "s3", "traceId": None, "name": "correctness",
         "value": 0.5, "dataType": "NUMERIC"},
        {"id": "s4", "traceId": "t4", "name": "notes",
         "value": "review text", "dataType": "TEXT"},
    ]
    data = LangfuseAdapter().parse(raw)
    assert data.n_examples == 1
    assert data.metadata["skipped_rows"] == 3


def test_langfuse_end_to_end_auto_detection_and_suite_ingest():
    from evaltrust.core.ingest import load_suite

    path = _TESTS_DIR / "fixtures" / "langfuse_scores_v3.json"
    suite = load_suite(str(path))
    assert list(suite) == ["correctness", "helpfulness", "quality_label"]
    assert all(data.source_format == "langfuse" for data in suite.values())


def test_langfuse_does_not_false_positive_on_other_fixtures():
    a = LangfuseAdapter()
    for other in (PROMPTFOO, NATIVE, LONG, WIDE, DEEPEVAL_SNAKE,
                  DEEPEVAL_CAMEL, OPENEVALS_SAMPLE, INSPECT, LANGSMITH, RAGAS):
        assert not a.detect(other), other

    files = list((_TESTS_DIR / "fixtures").glob("*.json")) + \
        list((_REPO_ROOT / "examples").glob("*.json"))
    for f in files:
        detected = a.detect(_load(f))
        if f.name == "langfuse_scores_v3.json":
            assert detected, f.name
        else:
            assert not detected, f.name


def test_detect_routes_langfuse_before_generic_fallback():
    assert detect_adapter(LANGFUSE_V3).source_format == "langfuse"
    assert GenericRecordsAdapter().detect(LANGFUSE_V3)


# ---------------------------------------------------------------------------
# CATEGORICAL: v2 and v3 disagree on where the label lives.
#
# v2 (legacy flat traceId/sessionId/observationId rows): `value` is a number
# that is only meaningful when `configId` links it to a score config - the
# schema says it "defaults to 0" otherwise - and `stringValue` always carries
# the human-readable label.
#
# v3 (`projectId`/`subject`-shaped rows): `value` IS the category string.
# There is no `stringValue` field in v3 at all, and `configId` (only present
# when `fields=details` was requested) is provenance, not a signal to treat
# `value` as a number.
# ---------------------------------------------------------------------------

def test_langfuse_v2_categorical_with_config_id_uses_the_numeric_value():
    # A configId means a score config maps the category to a number - trust it,
    # even though a (possibly stale/unrelated) stringValue is also present.
    raw = [
        {"id": "s1", "traceId": "t1", "name": "quality", "dataType": "CATEGORICAL",
         "configId": "cfg-1", "value": 0.75, "stringValue": "somewhat correct"},
    ]
    data = LangfuseAdapter().parse(raw)
    assert data.examples[0].scores["model"] == 0.75


def test_langfuse_v2_categorical_without_config_id_uses_string_value():
    # No configId: the numeric `value` defaults to 0 per the Langfuse schema and
    # isn't trustworthy, so the human-readable stringValue is used instead -
    # even when `value` is set to something else.
    raw = [
        {"id": "s1", "traceId": "t1", "name": "quality", "dataType": "CATEGORICAL",
         "value": 99, "stringValue": "correct"},
    ]
    data = LangfuseAdapter().parse(raw)
    assert data.examples[0].scores["model"] == 1.0


def test_langfuse_v2_categorical_without_config_id_skips_ambiguous_string_value():
    raw = [
        {"id": "s1", "traceId": "t1", "name": "quality", "dataType": "CATEGORICAL",
         "value": None, "stringValue": "somewhat correct"},
        {"id": "s2", "traceId": "t2", "name": "quality", "dataType": "CATEGORICAL",
         "value": None, "stringValue": "correct"},
    ]
    data = LangfuseAdapter().parse(raw)
    assert [ex.id for ex in data.examples] == ["t2"]
    assert data.metadata["skipped_rows"] == 1


def test_langfuse_v3_categorical_without_config_id_coerces_the_string_value_field():
    # The common case: `fields=subject` only, no `details`, so configId is
    # absent entirely. v3's `value` is still the category string directly.
    raw = [{
        "id": "s1", "projectId": "p1", "name": "quality", "value": "correct",
        "dataType": "CATEGORICAL", "source": "ANNOTATION",
        "subject": {"kind": "trace", "id": "t1"},
    }]
    data = LangfuseAdapter().parse(raw)
    assert data.examples[0].scores["model"] == 1.0


def test_langfuse_v3_categorical_with_config_id_still_reads_value_as_a_string():
    # Even when `fields=details` surfaces a configId, v3 never turns `value`
    # into a number - unlike v2, there is no numeric-mapping behavior to trust.
    raw = [{
        "id": "s1", "projectId": "p1", "name": "quality", "value": "incorrect",
        "dataType": "CATEGORICAL", "source": "ANNOTATION", "configId": "cfg-1",
        "subject": {"kind": "trace", "id": "t1"},
    }]
    data = LangfuseAdapter().parse(raw)
    assert data.examples[0].scores["model"] == 0.0


def test_langfuse_v3_categorical_skips_ambiguous_value():
    raw = [
        {"id": "s1", "projectId": "p1", "name": "quality",
         "value": "somewhat correct", "dataType": "CATEGORICAL",
         "source": "ANNOTATION", "subject": {"kind": "trace", "id": "t1"}},
        {"id": "s2", "projectId": "p1", "name": "quality", "value": "correct",
         "dataType": "CATEGORICAL", "source": "ANNOTATION",
         "subject": {"kind": "trace", "id": "t2"}},
    ]
    data = LangfuseAdapter().parse(raw)
    assert [ex.id for ex in data.examples] == ["t2"]
    assert data.metadata["skipped_rows"] == 1


def test_langfuse_boolean_never_falls_back_to_string_value():
    # v2 BOOLEAN always uses its numeric 0/1 value; a row with only a
    # stringValue ("true"/"false") and no numeric value is skipped, not
    # guessed at.
    raw = [
        {"id": "s1", "traceId": "t1", "name": "helpful", "dataType": "BOOLEAN",
         "value": 1, "stringValue": "true"},
        {"id": "s2", "traceId": "t2", "name": "helpful", "dataType": "BOOLEAN",
         "value": None, "stringValue": "false"},
    ]
    data = LangfuseAdapter().parse(raw)
    assert [ex.id for ex in data.examples] == ["t1"]
    assert data.metadata["skipped_rows"] == 1


def test_langfuse_v3_boolean_reads_a_json_boolean_value():
    # v3 BooleanScoreV3.value is a JSON boolean, not numeric 0/1.
    raw = [
        {"id": "s1", "projectId": "p1", "name": "helpful", "value": True,
         "dataType": "BOOLEAN", "source": "ANNOTATION",
         "subject": {"kind": "trace", "id": "t1"}},
        {"id": "s2", "projectId": "p1", "name": "helpful", "value": False,
         "dataType": "BOOLEAN", "source": "ANNOTATION",
         "subject": {"kind": "trace", "id": "t2"}},
    ]
    data = LangfuseAdapter().parse(raw)
    assert [ex.scores["model"] for ex in data.examples] == [1.0, 0.0]


# ---------------------------------------------------------------------------
# Incomplete pagination must be rejected, not silently partially audited
# ---------------------------------------------------------------------------

def test_langfuse_v3_rejects_a_response_with_a_next_page_cursor():
    raw = {
        "data": [{"id": "s1", "name": "correctness", "value": 1.0,
                  "dataType": "NUMERIC", "subject": {"kind": "trace", "id": "t1"}}],
        "meta": {"cursor": "eyJpZCI6InMxIn0="},
    }
    a = LangfuseAdapter()
    assert a.detect(raw)  # still recognised as Langfuse - the error must be specific
    with pytest.raises(ValueError, match="page"):
        a.parse(raw)


def test_langfuse_v2_rejects_a_response_with_more_pages_remaining():
    raw = {
        "data": [{"id": "s1", "traceId": "t1", "name": "correctness",
                  "value": 1.0, "dataType": "NUMERIC"}],
        "meta": {"page": 1, "totalPages": 3},
    }
    with pytest.raises(ValueError, match="page"):
        LangfuseAdapter().parse(raw)


def test_langfuse_v2_rejects_the_last_page_of_a_multi_page_response():
    # Being the *last* page does not mean this response's `data` array holds
    # everything - page 2 of 2 only ever contains page 2's rows, not page 1's.
    raw = {
        "data": [{"id": "s1", "traceId": "t1", "name": "correctness",
                  "value": 1.0, "dataType": "NUMERIC"}],
        "meta": {"page": 2, "totalPages": 2},
    }
    with pytest.raises(ValueError, match="page"):
        LangfuseAdapter().parse(raw)


def test_langfuse_accepts_a_v2_response_wrapped_as_data_meta_when_it_is_the_only_page():
    raw = {
        "data": [{"id": "s1", "traceId": "t1", "name": "correctness",
                  "value": 1.0, "dataType": "NUMERIC"}],
        "meta": {"page": 1, "totalPages": 1},
    }
    data = LangfuseAdapter().parse(raw)
    assert [ex.id for ex in data.examples] == ["t1"]


def test_langfuse_accepts_a_single_page_with_no_cursor_and_no_page_fields():
    raw = {
        "data": [{"id": "s1", "traceId": "t1", "name": "correctness",
                  "value": 1.0, "dataType": "NUMERIC"}],
        "meta": {"limit": 100},
    }
    data = LangfuseAdapter().parse(raw)
    assert [ex.id for ex in data.examples] == ["t1"]


# ---------------------------------------------------------------------------
# Duplicate (trace, metric) observation scores must raise, not average
# ---------------------------------------------------------------------------

def test_langfuse_raises_on_duplicate_trace_metric_scores_instead_of_averaging():
    raw = [
        {"id": "s1", "name": "correctness", "value": 1.0, "dataType": "NUMERIC",
         "subject": {"kind": "trace", "id": "t1"}},
        {"id": "s2", "name": "correctness", "value": 0.0, "dataType": "NUMERIC",
         "subject": {"kind": "observation", "id": "obs-1", "traceId": "t1"}},
    ]
    with pytest.raises(ValueError, match="aggregation policy"):
        LangfuseAdapter().parse(raw)


def test_langfuse_duplicate_check_does_not_trigger_across_different_metrics_or_traces():
    raw = [
        {"id": "s1", "traceId": "t1", "name": "correctness",
         "value": 1.0, "dataType": "NUMERIC"},
        {"id": "s2", "traceId": "t1", "name": "helpfulness",
         "value": 0.0, "dataType": "NUMERIC"},
        {"id": "s3", "traceId": "t2", "name": "correctness",
         "value": 0.5, "dataType": "NUMERIC"},
    ]
    suite = LangfuseAdapter().parse_suite(raw)
    assert list(suite) == ["correctness", "helpfulness"]
    assert [ex.id for ex in suite["correctness"].examples] == ["t1", "t2"]


# ---------------------------------------------------------------------------
# Detection must not depend on a numeric `value` key being present
# ---------------------------------------------------------------------------

def test_langfuse_detects_a_v2_row_with_string_value_and_no_value_key():
    raw = [
        {"id": "s1", "traceId": "t1", "name": "notes",
         "stringValue": "needs a citation", "dataType": "TEXT"},
    ]
    assert LangfuseAdapter().detect(raw)


def test_langfuse_text_only_export_gives_an_unsupported_type_error_not_unknown_format():
    raw = [
        {"id": "s1", "traceId": "t1", "name": "notes",
         "stringValue": "needs a citation", "dataType": "TEXT"},
        {"id": "s2", "traceId": "t2", "name": "notes",
         "stringValue": "looks good", "dataType": "TEXT"},
    ]
    # Detection must succeed so registry routing reaches this adapter's specific
    # error, instead of falling through to UnknownFormatError.
    assert detect_adapter(raw).source_format == "langfuse"
    with pytest.raises(ValueError, match="TEXT and CORRECTION"):
        LangfuseAdapter().parse(raw)


# ---------------------------------------------------------------------------
# Distinguishable errors: missing subject vs unsupported subject vs unsupported
# type vs duplicate scores vs incomplete pagination
# ---------------------------------------------------------------------------

def test_langfuse_session_and_experiment_subjects_get_a_specific_error():
    raw = [
        {"id": "s1", "name": "correctness", "value": 1.0, "dataType": "NUMERIC",
         "subject": {"kind": "session", "id": "sess-1"}},
        {"id": "s2", "name": "correctness", "value": 1.0, "dataType": "NUMERIC",
         "subject": {"kind": "experiment", "id": "exp-1"}},
    ]
    a = LangfuseAdapter()
    assert a.detect(raw)
    with pytest.raises(ValueError, match="session or experiment"):
        a.parse(raw)


def test_langfuse_missing_subject_error_is_distinct_from_unsupported_type_error():
    missing_subject = [{
        "id": "s1", "projectId": "p1", "name": "correctness",
        "value": 0.9, "dataType": "NUMERIC", "source": "EVAL",
    }]
    unsupported_type = [
        {"id": "s1", "traceId": "t1", "name": "notes",
         "stringValue": "x", "dataType": "TEXT"},
    ]
    with pytest.raises(ValueError, match="fields=subject") as missing_exc:
        LangfuseAdapter().parse(missing_subject)
    with pytest.raises(ValueError, match="TEXT and CORRECTION") as type_exc:
        LangfuseAdapter().parse(unsupported_type)
    assert "fields=subject" not in str(type_exc.value)
    assert "TEXT and CORRECTION" not in str(missing_exc.value)
