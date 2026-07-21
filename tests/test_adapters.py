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
# MLflow evaluate (per-row eval_results_table -- one model per run)
# ---------------------------------------------------------------------------

from evaltrust.adapters.mlflow_evaluate import MlflowEvaluateAdapter

# The direct eval_results_table.json artifact shape: a pandas DataFrame dumped
# with to_json(orient="split", index=False) by mlflow.log_table(). See
# tests/fixtures/mlflow_eval_results_table.json for a fuller, real-shaped table.
MLFLOW_EVALUATE = {
    "columns": ["questions", "outputs", "exact_match/score", "toxicity/v1/score"],
    "data": [
        ["What is MLflow?", "MLflow manages the ML lifecycle.", 0.0, 0.0002],
        ["What is Databricks?", "Databricks is a cloud analytics platform.", 0.0, 0.0001],
    ],
}


def test_mlflow_evaluate_detects_and_parses_the_real_fixture():
    raw = _load(_TESTS_DIR / "fixtures" / "mlflow_eval_results_table.json")
    a = MlflowEvaluateAdapter()
    assert a.detect(raw)
    data = a.parse(raw)
    assert data.source_format == "mlflow_evaluate"
    assert data.models == ["model"]
    assert data.n_examples == 2


def test_mlflow_evaluate_single_audit_uses_first_metric_column():
    # Column order: exact_match/score precedes toxicity/v1/score.
    data = MlflowEvaluateAdapter().parse(MLFLOW_EVALUATE)
    assert [ex.scores["model"] for ex in data.examples] == [0.0, 0.0]


def test_mlflow_evaluate_suite_fans_out_every_metric_column():
    suite = MlflowEvaluateAdapter().parse_suite(MLFLOW_EVALUATE)
    assert set(suite.keys()) == {"exact_match", "toxicity/v1"}
    assert [ex.scores["model"] for ex in suite["toxicity/v1"].examples] == [0.0002, 0.0001]


def test_mlflow_evaluate_accepts_the_eval_results_table_wrapper():
    # result.tables round-tripped as {"eval_results_table": <split dict>}.
    wrapped = {"eval_results_table": MLFLOW_EVALUATE}
    a = MlflowEvaluateAdapter()
    assert a.detect(wrapped)
    assert a.parse(wrapped).n_examples == 2


def test_mlflow_evaluate_accepts_a_records_orient_wrapped_table():
    # A user who dumps result.tables["eval_results_table"].to_dict(orient="records")
    # instead of round-tripping the raw split-orient artifact.
    wrapped = {"eval_results_table": [
        {"questions": "q1", "exact_match/score": 1.0},
        {"questions": "q2", "exact_match/score": 0.0},
    ]}
    a = MlflowEvaluateAdapter()
    assert a.detect(wrapped)
    data = a.parse(wrapped)
    assert [ex.scores["model"] for ex in data.examples] == [1.0, 0.0]


def test_mlflow_evaluate_bare_token_count_and_latency_columns_count_as_metrics():
    raw = {"columns": ["questions", "token_count", "latency"],
           "data": [["q1", 12, 0.53], ["q2", 9, 0.41]]}
    a = MlflowEvaluateAdapter()
    assert a.detect(raw)
    suite = a.parse_suite(raw)
    assert set(suite.keys()) == {"token_count", "latency"}


def test_mlflow_evaluate_skips_and_counts_an_unreadable_metric_score():
    raw = {"columns": ["questions", "exact_match/score", "toxicity/v1/score"],
           "data": [["q1", 1.0, 0.1], ["q2", 0.0, None]]}
    suite = MlflowEvaluateAdapter().parse_suite(raw)
    assert suite["exact_match"].n_examples == 2
    assert suite["toxicity/v1"].n_examples == 1
    assert suite["toxicity/v1"].metadata["skipped_rows"] == 1


def test_mlflow_evaluate_raises_when_no_row_has_a_usable_metric_score():
    raw = {"columns": ["questions", "exact_match/score"],
           "data": [["q1", None], ["q2", None]]}
    with pytest.raises(ValueError):
        MlflowEvaluateAdapter().parse(raw)


def test_mlflow_evaluate_rejects_a_table_with_no_mlflow_metric_columns():
    # A generic wide-format split-orient table with no MLflow metric naming
    # convention must not be claimed.
    raw = {"columns": ["questions", "gpt4_score"], "data": [["q1", 1]]}
    assert not MlflowEvaluateAdapter().detect(raw)


def test_mlflow_evaluate_does_not_false_positive_on_other_fixtures():
    a = MlflowEvaluateAdapter()
    for other in (PROMPTFOO, NATIVE, LONG, WIDE, DEEPEVAL_SNAKE, DEEPEVAL_CAMEL,
                  OPENEVALS_SAMPLE, INSPECT, LANGSMITH, RAGAS):
        assert not a.detect(other), other


def test_no_earlier_adapter_claims_an_mlflow_evaluate_export():
    assert not PromptfooAdapter().detect(MLFLOW_EVALUATE)
    assert not DeepEvalAdapter().detect(MLFLOW_EVALUATE)
    assert not OpenEvalsAdapter().detect(MLFLOW_EVALUATE)
    assert not InspectAdapter().detect(MLFLOW_EVALUATE)
    assert not LangSmithAdapter().detect(MLFLOW_EVALUATE)
    assert not RagasAdapter().detect(MLFLOW_EVALUATE)
    assert not NativeNestedAdapter().detect(MLFLOW_EVALUATE)


def test_detect_routes_mlflow_evaluate():
    assert detect_adapter(MLFLOW_EVALUATE).source_format == "mlflow_evaluate"
