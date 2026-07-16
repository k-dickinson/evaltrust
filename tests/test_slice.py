"""Tests for the Per-slice Comparison audit (issue #84)."""

from evaltrust.audit.slice import audit_slices
from evaltrust.audit.runner import run_audit
from evaltrust.core.schema import EvalData, Example, Status


def _data(examples):
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


def _by_check(findings, check):
    (f,) = [f for f in findings if f.details.get("check") == check]
    return f


def test_missing_attribute_returns_skip():
    data = _data([Example(str(i), {"A": 0.0, "B": 1.0}) for i in range(10)])
    (f,) = audit_slices(data, "A", "B", slice_by="category")
    assert f.status is Status.SKIP
    assert f.details["reason"] == "attribute_absent"


def test_flags_slice_that_regresses_against_overall():
    # Overall B > A (B wins because 'easy' dominates), but the 'hard' slice
    # regresses (A > B on it).
    examples = []
    for i in range(60):
        examples.append(Example(f"e{i}",
                                {"A": 0.0, "B": 1.0},
                                attributes={"difficulty": "easy"}))
    for i in range(20):
        examples.append(Example(f"h{i}",
                                {"A": 1.0, "B": 0.0},
                                attributes={"difficulty": "hard"}))
    data = _data(examples)
    (f,) = audit_slices(data, "A", "B", slice_by="difficulty", seed=0)
    assert f.status is Status.WARN
    assert "hard" in f.details["regressions"]
    assert "easy" not in f.details["regressions"]
    # Bonferroni across k=2 slices halves alpha.
    assert f.details["corrected_alpha"] == 0.05 / 2


def test_no_regression_when_all_slices_agree_with_overall():
    examples = []
    for i in range(30):
        examples.append(Example(f"e{i}",
                                {"A": 0.0, "B": 1.0},
                                attributes={"language": "en"}))
    for i in range(30):
        examples.append(Example(f"f{i}",
                                {"A": 0.0, "B": 1.0},
                                attributes={"language": "fr"}))
    data = _data(examples)
    (f,) = audit_slices(data, "A", "B", slice_by="language", seed=0)
    assert f.status is Status.PASS
    assert f.details["regressions"] == []


def test_small_slices_are_reported_but_not_tested():
    examples = [
        Example("s0", {"A": 0.0, "B": 1.0}, attributes={"cat": "tiny"}),
        Example("s1", {"A": 0.0, "B": 1.0}, attributes={"cat": "tiny"}),
    ]
    examples += [
        Example(f"b{i}", {"A": 0.0, "B": 1.0}, attributes={"cat": "big"})
        for i in range(20)
    ]
    data = _data(examples)
    (f,) = audit_slices(data, "A", "B", slice_by="cat", seed=0)
    tiny = next(s for s in f.details["slices"] if s["value"] == "tiny")
    big = next(s for s in f.details["slices"] if s["value"] == "big")
    assert tiny["assessed"] is False
    assert tiny["reason"] == "too_few_examples"
    assert big["assessed"] is True
    # Bonferroni family = tested slices only, not the raw group count.
    assert f.details["n_slices"] == 1
    assert f.details["n_slices_total"] == 2
    assert f.details["corrected_alpha"] == 0.05 / 1


def test_bonferroni_family_uses_only_tested_slices():
    # 40 easy examples (B wins), 8 hard (A wins), plus five 2-example slices
    # that must not enlarge the Bonferroni family: with k=7 (raw) the threshold
    # 0.05/7 ~= 0.0071 misses the hard McNemar p ~ 0.0078; with k=2 (tested)
    # the threshold 0.025 correctly rejects and flags 'hard' as a regression.
    examples = []
    for i in range(40):
        examples.append(Example(f"e{i}", {"A": 0.0, "B": 1.0},
                                attributes={"cat": "easy"}))
    for i in range(8):
        examples.append(Example(f"h{i}", {"A": 1.0, "B": 0.0},
                                attributes={"cat": "hard"}))
    for j in range(5):
        for i in range(2):
            examples.append(Example(f"t{j}_{i}", {"A": 0.0, "B": 1.0},
                                    attributes={"cat": f"tiny{j}"}))
    data = _data(examples)
    (f,) = audit_slices(data, "A", "B", slice_by="cat", seed=0)
    assert f.details["n_slices"] == 2      # only 'easy' and 'hard' were tested
    assert f.details["n_slices_total"] == 7
    assert f.details["corrected_alpha"] == 0.05 / 2
    assert "hard" in f.details["regressions"]
    assert f.status is Status.WARN


def test_all_slices_too_small_returns_skip():
    examples = [
        Example("s0", {"A": 0.0, "B": 1.0}, attributes={"cat": "a"}),
        Example("s1", {"A": 0.0, "B": 1.0}, attributes={"cat": "b"}),
    ]
    data = _data(examples)
    (f,) = audit_slices(data, "A", "B", slice_by="cat", seed=0)
    assert f.status is Status.SKIP
    assert f.details["reason"] == "all_slices_too_small"


def test_run_audit_emits_slice_skip_when_only_preferences_are_present():
    from evaltrust.core.schema import Preference
    examples = [
        Example(f"e{i}", scores={},
                preferences={"j": "B" if i < 6 else Preference.TIE},
                attributes={"cat": "x"})
        for i in range(10)
    ]
    data = EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})
    report = run_audit(data, model_a="A", model_b="B", slice_by="cat", seed=0)
    slice_f = _by_check(report.findings, "slice_comparison")
    assert slice_f.status is Status.SKIP
    assert slice_f.details["reason"] == "preference_only"


def test_run_audit_appends_slice_finding_when_slice_by_is_set():
    examples = []
    for i in range(40):
        examples.append(Example(f"e{i}", {"A": 0.0, "B": 1.0},
                                attributes={"topic": "math"}))
    for i in range(20):
        examples.append(Example(f"g{i}", {"A": 1.0, "B": 0.0},
                                attributes={"topic": "grammar"}))
    data = _data(examples)
    report = run_audit(data, model_a="A", model_b="B", slice_by="topic", seed=0)
    slice_f = _by_check(report.findings, "slice_comparison")
    assert slice_f.details["slice_by"] == "topic"
    # 'grammar' regresses against the (B-favoured) overall direction.
    assert "grammar" in slice_f.details["regressions"]


def test_run_audit_without_slice_by_does_not_add_slice_finding():
    data = _data([Example(f"e{i}", {"A": 0.0, "B": 1.0}) for i in range(10)])
    report = run_audit(data, model_a="A", model_b="B", seed=0)
    assert not any(f.details.get("check") == "slice_comparison"
                   for f in report.findings)


def test_permutation_seed_is_offset_per_slice():
    # Two continuous slices with the same difference distribution: with a
    # shared seed the permutation p-values would be identical (correlated
    # Monte-Carlo error). Under the per-slice seed offset each slice draws an
    # independent stream, so the p-values differ.
    import numpy as np
    rng = np.random.default_rng(0)
    diffs_shared = rng.normal(0.1, 0.5, size=40)  # identical differences
    examples = []
    for i, d in enumerate(diffs_shared):
        examples.append(Example(f"a{i}", {"A": 0.0, "B": float(d)},
                                attributes={"g": "one"}))
    for i, d in enumerate(diffs_shared):
        examples.append(Example(f"b{i}", {"A": 0.0, "B": float(d)},
                                attributes={"g": "two"}))
    data = _data(examples)
    (f,) = audit_slices(data, "A", "B", slice_by="g", seed=0)
    slices = {s["value"]: s for s in f.details["slices"] if s["assessed"]}
    # With the per-slice offset the two identically-distributed slices no
    # longer produce byte-identical p-values from the permutation test.
    assert slices["one"]["p_value"] != slices["two"]["p_value"]


def test_native_adapter_reads_attributes_field():
    from evaltrust.adapters.generic import NativeNestedAdapter
    raw = {
        "examples": [
            {"id": "q1", "scores": {"A": 1.0, "B": 0.0},
             "attributes": {"category": "math", "difficulty": "easy"}},
            {"id": "q2", "scores": {"A": 0.0, "B": 1.0},
             "attributes": {"category": "code"}},
            {"id": "q3", "scores": {"A": 1.0, "B": 1.0}},
        ]
    }
    data = NativeNestedAdapter().parse(raw)
    assert data.examples[0].attributes == {"category": "math", "difficulty": "easy"}
    assert data.examples[1].attributes == {"category": "code"}
    assert data.examples[2].attributes is None
