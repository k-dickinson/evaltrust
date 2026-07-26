"""Pairwise-preference schema, audit, dispatcher, and ingestion tests."""

from __future__ import annotations

from collections import OrderedDict
import json

import evaltrust
import numpy as np
import pytest
from scipy.stats import binomtest

from evaltrust.adapters.common import (
    DEFAULT_PREFERENCE_JUDGE,
    PreferenceRecord,
    Record,
    coerce_score,
    records_to_evaldata,
)
from evaltrust.audit.preference import _preference_magnitude, audit_preferences
from evaltrust.audit.runner import run_audit
from evaltrust.audit.suite import audit_suite
from evaltrust.config import AuditConfig
from evaltrust.core.ingest import load
from evaltrust.core.pairing import merge_two
from evaltrust.core.schema import EvalData, Example, Preference, Status
from evaltrust.stats.resampling import bootstrap_ci


FAST_CONFIG = AuditConfig(n_resamples=500, seed=7)


def _data(votes, models=("A", "B"), scores=None):
    if scores is None:
        scores = [{} for _ in votes]
    return EvalData(
        models=list(models),
        examples=[
            Example(id=str(i), scores=score, preferences=preference)
            for i, (score, preference) in enumerate(zip(scores, votes, strict=True))
        ],
        source_format="test",
    )


def _finding(findings, check):
    return next(f for f in findings if f.details.get("check") == check)


def test_data_helper_rejects_mismatched_scores_and_votes():
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is longer"):
        _data([{"judge": "A"}, {"judge": "B"}], scores=[{}])


def test_data_helper_preserves_explicit_empty_scores():
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is longer"):
        _data([{"judge": "A"}], scores=[])


def test_schema_reports_preference_evidence():
    empty = _data([None])
    populated = _data([{"judge": "A"}])
    assert not empty.has_preferences
    assert populated.has_preferences
    assert evaltrust.Preference is Preference


def test_preference_only_comparison_reaches_the_new_check_and_skips_score_pillars():
    # This is the exact shape that used to fail at runner.py's empty-differences
    # guard before benchmark_health could run.
    data = _data([{"judge": "A"} for _ in range(8)])
    report = run_audit(data, config=FAST_CONFIG)

    assert _finding(report.findings, "preference_significance").details["wins_a"] == 8
    assert _finding(report.findings, "preference_effect").details["win_rate_a"] == 1.0
    skipped = {
        f.pillar for f in report.findings
        if f.status is Status.SKIP and f.details.get("reason") == "preference_only"
    }
    assert skipped == {
        "Statistical Validity",
        "Benchmark Health",
        "Repeatability",
        "Judge Reliability",
    }


def test_missing_scores_and_preferences_keeps_a_helpful_error():
    data = _data([None])
    with pytest.raises(ValueError, match="scores or preferences"):
        run_audit(data, "A", "B", config=FAST_CONFIG)


def test_mixed_scores_and_preferences_runs_both_audits():
    votes = [{"judge": "A"}, {"judge": "B"}, {"judge": "A"}, {"judge": "A"}]
    scores = [
        {"A": 1.0, "B": 0.0},
        {"A": 0.0, "B": 1.0},
        {"A": 1.0, "B": 0.0},
        {"A": 1.0, "B": 0.0},
    ]
    report = run_audit(_data(votes, scores=scores), config=FAST_CONFIG)
    checks = {f.details.get("check") for f in report.findings}
    assert "decision" in checks
    assert "preference_significance" in checks
    assert "preference_effect" in checks
    assert not any(f.details.get("reason") == "preference_only" for f in report.findings)


def test_unpaired_scores_run_safe_score_pillars_without_claiming_scores_are_absent():
    data = _data(
        [{"judge": "A"}, {"judge": "B"}],
        scores=[{"A": 1.0}, {"B": 0.0}],
    )
    report = run_audit(data, "A", "B", config=FAST_CONFIG)
    stat = _finding(report.findings, "score_statistical_validity")
    assert stat.status is Status.SKIP
    assert stat.details["reason"] == "no_paired_scores"
    assert any(
        f.pillar == "Benchmark Health" and f.status is not Status.SKIP
        for f in report.findings
    )


def test_no_preferences_returns_a_skip():
    findings = audit_preferences(_data([None]), "A", "B", n_resamples=50)
    assert len(findings) == 1
    assert findings[0].status is Status.SKIP
    assert findings[0].details["check"] == "preference"
    assert findings[0].details["assessed"] is False


def test_all_ties_skip_with_the_tie_count_visible():
    data = _data([
        {"human": Preference.TIE},
        {"human": Preference.TIE},
    ])
    finding = audit_preferences(data, "A", "B", n_resamples=50)[0]
    assert finding.status is Status.SKIP
    assert finding.details["wins_a"] == 0
    assert finding.details["wins_b"] == 0
    assert finding.details["ties"] == 2
    assert finding.details["n_decisive"] == 0
    assert finding.details["p_value"] == 1.0


def test_split_vote_is_one_example_level_tie():
    data = _data([
        {"j1": "A", "j2": "B"},
        {"j1": "A", "j2": "A"},
    ])
    details = _finding(
        audit_preferences(data, "A", "B", n_resamples=50),
        "preference_significance",
    ).details
    assert details["wins_a"] == 1
    assert details["wins_b"] == 0
    assert details["ties"] == 1


def test_per_judge_tallies_keep_direction_disagreement_visible():
    # Majority by example says A wins 2-1. Pooling the nine votes says B wins 5-4.
    data = _data([
        {"j1": "A", "j2": "A", "j3": "B"},
        {"j1": "A", "j2": "A", "j3": "B"},
        {"j1": "B", "j2": "B", "j3": "B"},
    ])
    details = _finding(
        audit_preferences(data, "A", "B", n_resamples=50),
        "preference_significance",
    ).details
    assert (details["wins_a"], details["wins_b"], details["ties"]) == (2, 1, 0)
    assert details["per_judge"] == {
        "j1": {"wins_a": 2, "wins_b": 1, "ties": 0},
        "j2": {"wins_a": 2, "wins_b": 1, "ties": 0},
        "j3": {"wins_a": 0, "wins_b": 3, "ties": 0},
    }


def test_significance_matches_scipy_exact_binomial_reference():
    data = _data([{"judge": "A"}] * 8 + [{"judge": "B"}] * 2)
    details = _finding(
        audit_preferences(data, "A", "B", n_resamples=50),
        "preference_significance",
    ).details
    expected = float(binomtest(2, 10, 0.5, alternative="two-sided").pvalue)
    assert details["p_value"] == expected


def test_significant_but_negligible_preference_effect_warns():
    data = _data([{"judge": "A"}] * 53 + [{"judge": "B"}] * 47)
    findings = audit_preferences(
        data,
        "A",
        "B",
        n_resamples=50,
        significant=True,
    )

    assert _finding(findings, "preference_significance").status is Status.PASS
    effect = _finding(findings, "preference_effect")
    assert effect.status is Status.WARN
    assert effect.details["cohens_g"] == pytest.approx(0.03)
    assert effect.details["magnitude"] == "negligible"
    assert "too small to matter" in effect.how_to_fix


@pytest.mark.parametrize(
    ("cohens_g", "expected"),
    [
        (0.049, "negligible"),
        (0.05, "small"),
        (0.149, "small"),
        (0.15, "medium"),
        (0.249, "medium"),
        (0.25, "large"),
    ],
)
def test_preference_magnitude_boundaries(cohens_g, expected):
    assert _preference_magnitude(cohens_g) == expected


def test_preference_bootstrap_ci_is_seeded_and_deterministic():
    data = _data([{"judge": "A"}] * 8 + [{"judge": "B"}] * 2)
    first = _finding(
        audit_preferences(data, "A", "B", n_resamples=500, seed=11),
        "preference_effect",
    ).details
    second = _finding(
        audit_preferences(data, "A", "B", n_resamples=500, seed=11),
        "preference_effect",
    ).details
    expected = bootstrap_ci(
        np.array([1.0] * 8 + [0.0] * 2), n_resamples=500, seed=11)
    assert first == second
    assert (first["ci_low"], first["ci_high"]) == expected


def test_model_named_tie_does_not_collide_with_the_tie_enum():
    assert Preference.TIE != "tie"
    data = _data([
        {"model-vote": "tie"},
        {"tie-vote": Preference.TIE},
    ], models=("tie", "other"))
    details = _finding(
        audit_preferences(data, "tie", "other", n_resamples=50),
        "preference_significance",
    ).details
    assert details["wins_a"] == 1
    assert details["ties"] == 1


def test_preference_only_auto_selection_requires_an_unambiguous_pair():
    report = run_audit(_data([{"judge": "A"}], models=("A", "B")), config=FAST_CONFIG)
    assert (report.model_a, report.model_b) == ("A", "B")

    ambiguous = _data([{"judge": "A"}], models=("A", "B", "C"))
    with pytest.raises(ValueError, match="name the two models"):
        run_audit(ambiguous, config=FAST_CONFIG)


def test_all_pairs_skips_preference_only_data_with_explicit_primary_pair():
    data = _data([{"judge": "A"} for _ in range(8)], models=("A", "B", "C"))
    report = run_audit(
        data,
        model_a="A",
        model_b="B",
        config=AuditConfig(all_pairs=True, n_resamples=50, seed=7),
    )
    finding = _finding(report.findings, "all_pairs")

    assert finding.status is Status.SKIP
    assert finding.details["assessed"] is False
    assert finding.details["reason"] == "preference_only"


def test_all_pairs_does_not_bypass_ambiguous_preference_model_selection():
    data = _data([{"judge": "A"}], models=("A", "B", "C"))
    with pytest.raises(ValueError, match="name the two models"):
        run_audit(
            data,
            config=AuditConfig(all_pairs=True, n_resamples=50, seed=7),
        )


def test_preference_only_input_is_not_dispatched_as_a_single_model_audit():
    data = _data([{"judge": "A"}], models=("A",))
    with pytest.raises(ValueError, match="two models"):
        run_audit(data, config=FAST_CONFIG)


def test_preference_only_input_is_not_dispatched_as_a_threshold_audit():
    data = _data([{"judge": "A"}], models=("A", "B"))
    with pytest.raises(ValueError, match="threshold audit needs scores"):
        run_audit(data, threshold=0.5, config=FAST_CONFIG)


def test_preference_only_suite_uses_real_pvalues_for_holm_and_model_selection():
    suite = OrderedDict([
        ("m1", _data([{"judge": "A"}] * 9 + [{"judge": "B"}])),
        ("m2", _data([{"judge": "A"}] * 8 + [{"judge": "B"}])),
    ])
    report = audit_suite(
        suite,
        config=AuditConfig(correction="holm", n_resamples=50, seed=3),
    )
    assert report.metric_alphas == {"m1": 0.025, "m2": 0.05}
    assert report.adjusted_p == {
        "m1": 0.04296875,
        "m2": 0.04296875,
    }
    assert all(
        _finding(metric_report.findings, "preference_significance").details["significant"]
        for metric_report in report.reports.values()
    )


def test_corrected_suite_rejects_two_hypothesis_families_in_one_metric():
    scores = [{"A": 1.0, "B": 0.0}] * 10
    mixed = _data([{"judge": "A"}] * 10, scores=scores)
    suite = OrderedDict([("m1", mixed), ("m2", mixed)])
    with pytest.raises(ValueError, match="score and preference significance"):
        audit_suite(suite, correction="holm", config=FAST_CONFIG)


@pytest.mark.parametrize(
    ("name", "text", "expected_format"),
    [
        (
            "preferences.csv",
            "id,A,B,winner,judge\nq1,,,A,human\nq2,,,B,ai\nq3,,,tIe,human\n",
            "csv",
        ),
        (
            "preferences.json",
            json.dumps([
                {"id": "q1", "A": "", "B": "", "preference": "A", "judge": "human"},
                {"id": "q2", "A": "", "B": "", "preference": "TIE", "judge": "ai"},
                {"id": "q3", "A": "", "B": "", "preference": "B", "judge": "human"},
            ]),
            "generic",
        ),
        (
            "preferences.jsonl",
            '{"id":"q1","A":"","B":"","winner":"A","judge":"human"}\n'
            '{"id":"q2","A":"","B":"","winner":"tie","judge":"ai"}\n'
            '{"id":"q3","A":"","B":"","winner":"B","judge":"human"}\n',
            "jsonl",
        ),
    ],
)
def test_generic_preference_columns_round_trip(name, text, expected_format, tmp_path):
    path = tmp_path / name
    path.write_text(text)
    data = load(str(path))
    assert data.source_format == expected_format
    assert data.models == ["A", "B"]
    assert data.has_preferences
    assert data.examples[0].preferences["human"] == "A"
    assert any(
        Preference.TIE in (example.preferences or {}).values()
        for example in data.examples
    )


def test_missing_judge_uses_the_default_key_and_mixed_scores_survive(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text("id,A,B,winner\nq1,1,0,A\nq2,0,1,B\n")
    data = load(str(path))
    assert data.examples[0].scores == {"A": 1.0, "B": 0.0}
    assert data.examples[0].preferences == {DEFAULT_PREFERENCE_JUDGE: "A"}
    assert coerce_score("win") == 1.0
    assert coerce_score("loss") == 0.0


@pytest.mark.parametrize("model_column", ["winner", "preference"])
def test_preference_alias_does_not_hijack_an_existing_wide_score_model(
    model_column, tmp_path
):
    path = tmp_path / "score-alias.json"
    path.write_text(json.dumps([
        {"id": "q1", model_column: "win", "other": "loss"},
        {"id": "q2", model_column: "loss", "other": "win"},
    ]))
    data = load(str(path))
    assert data.models == [model_column, "other"]
    assert data.examples[0].scores == {model_column: 1.0, "other": 0.0}
    assert not data.has_preferences


def test_blank_optional_preference_cell_keeps_mixed_scores(tmp_path):
    path = tmp_path / "partial.csv"
    path.write_text(
        "id,A,B,winner\n"
        "q1,1,0,A\n"
        "q2,0,1,\n"
    )
    data = load(str(path))
    assert data.n_examples == 2
    assert data.examples[0].preferences == {DEFAULT_PREFERENCE_JUDGE: "A"}
    assert data.examples[1].preferences is None
    assert data.examples[1].scores == {"A": 0.0, "B": 1.0}


def test_unscored_preference_winner_is_kept_as_a_model(tmp_path):
    path = tmp_path / "unscored-winner.csv"
    path.write_text("id,A,B,winner\nq1,1,,B\n")
    data = load(str(path))
    assert data.models == ["A", "B"]
    assert data.examples[0].scores == {"A": 1.0}
    assert data.examples[0].preferences == {DEFAULT_PREFERENCE_JUDGE: "B"}


def test_unknown_winner_is_rejected_when_preference_records_declare_models():
    records = [PreferenceRecord("q1", "AA", models=("A", "B"))]
    with pytest.raises(
        ValueError,
        match=r"unknown preference winner 'AA'.*known models.*'A'.*'B'",
    ):
        records_to_evaldata(records, "test")


def test_unknown_winner_is_rejected_when_scores_establish_models():
    records = [
        PreferenceRecord("q1", "AA"),
        Record("q1", "A", 1.0),
        Record("q1", "B", 0.0),
    ]
    with pytest.raises(
        ValueError,
        match=r"unknown preference winner 'AA'.*known models.*'A'.*'B'",
    ):
        records_to_evaldata(records, "test")


def test_preference_winners_declare_models_when_no_other_declaration_exists():
    data = records_to_evaldata(
        [PreferenceRecord("q1", "A"), PreferenceRecord("q2", "B")],
        "test",
    )
    assert data.models == ["A", "B"]


def test_empty_metadata_column_is_not_inferred_as_a_model(tmp_path):
    path = tmp_path / "preference-only.csv"
    path.write_text(
        "id,A,B,notes,winner\n"
        "q1,,,,A\n"
        "q2,,,,B\n"
    )
    data = load(str(path))
    assert data.models == ["A", "B"]
    assert "notes" not in data.models


def test_model_named_tie_round_trips_when_the_winner_is_unambiguous(tmp_path):
    path = tmp_path / "tie-model.csv"
    path.write_text(
        "id,tie,other,winner,judge\n"
        "q1,1,0,other,human\n"
    )
    data = load(str(path))
    assert data.models == ["tie", "other"]
    assert data.examples[0].preferences == {"human": "other"}


@pytest.mark.parametrize("token", ["tie", "TIE", "TiE"])
def test_tie_token_with_a_declared_tie_model_is_rejected_as_ambiguous(
    token, tmp_path
):
    path = tmp_path / "long-tie.csv"
    path.write_text(
        "id,model,score,winner,judge\n"
        f"q1,TIE,1,{token},human\n"
        f"q1,other,0,{token},human\n"
    )
    with pytest.raises(ValueError, match="ambiguous"):
        load(str(path))


def test_two_file_pairing_rejects_preferences_instead_of_dropping_them():
    left = EvalData(
        models=["left"],
        examples=[Example("q1", {"left": 1.0}, preferences={"judge": "left"})],
        source_format="left",
    )
    right = EvalData(
        models=["right"],
        examples=[Example("q1", {"right": 0.0})],
        source_format="right",
    )
    with pytest.raises(ValueError, match="one file"):
        merge_two(left, right, "A", "B")


def test_native_nested_json_preserves_preferences(tmp_path):
    path = tmp_path / "native-preferences.json"
    path.write_text(json.dumps({
        "models": ["A", "B"],
        "examples": [
            {
                "id": "q1",
                "scores": {"A": 1.0, "B": 0.0},
                "preferences": {"human": "A"},
            },
            {
                "id": "q2",
                "scores": {"A": 0.5, "B": 0.5},
                "preferences": {"human": "tie"},
            },
        ],
    }))
    data = load(str(path))
    assert data.source_format == "native"
    assert data.examples[0].preferences == {"human": "A"}
    assert data.examples[1].preferences == {"human": Preference.TIE}


def test_native_nested_json_uses_global_scores_to_reject_unknown_winner(tmp_path):
    path = tmp_path / "native-unknown-winner.json"
    path.write_text(json.dumps({
        "examples": [
            {"id": "q1", "scores": {"A": 1.0, "B": 0.0}},
            {"id": "q2", "scores": {}, "preferences": {"human": "AA"}},
        ],
    }))
    with pytest.raises(
        ValueError,
        match=r"unknown preference winner 'AA'.*known models.*'A'.*'B'",
    ):
        load(str(path))
