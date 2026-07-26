"""Tests for the Statistical Validity audit — EvalTrust's flagship check.

The audit produces three findings:
  - decision:   is there a real, meaningful improvement? (significant /
                equivalent / inconclusive) — never a blunt "not significant = fail"
  - effect_size: how big is it, in interpretable terms (Cohen's d, or a proportion
                effect size for binary data)
  - precision:  was the sample large enough, framed prospectively (minimum
                detectable effect), not as post-hoc power
"""

import numpy as np
import pytest

from evaltrust.audit.statistical import audit_statistical_validity
from evaltrust.core.schema import EvalData, Example, Status


def make_data(a_scores, b_scores):
    examples = [
        Example(id=str(i), scores={"A": float(a), "B": float(b)})
        for i, (a, b) in enumerate(zip(a_scores, b_scores))
    ]
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


def by_check(findings, check):
    (f,) = [f for f in findings if f.details.get("check") == check]
    return f


def test_produces_three_checks():
    findings = audit_statistical_validity(make_data([0] * 50, [1] * 50), "A", "B")
    assert {f.details["check"] for f in findings} == {
        "decision", "effect_size", "precision"}


def test_clean_win_is_a_significant_improvement():
    a, b = [0] * 200, [1] * 180 + [0] * 20
    findings = audit_statistical_validity(make_data(a, b), "A", "B", seed=0)
    decision = by_check(findings, "decision")
    assert decision.status is Status.PASS
    assert decision.details["outcome"] == "significant"
    assert by_check(findings, "effect_size").status is Status.PASS
    assert by_check(findings, "precision").status is Status.PASS


def test_binary_data_uses_mcnemar():
    a, b = [0] * 100, [1] * 90 + [0] * 10
    decision = by_check(
        audit_statistical_validity(make_data(a, b), "A", "B"), "decision")
    assert "McNemar" in decision.how_detected


def test_binary_effect_size_reports_proportion_difference():
    a, b = [0] * 100, [1] * 80 + [0] * 20
    effect = by_check(
        audit_statistical_validity(make_data(a, b), "A", "B"), "effect_size")
    # 80% vs 0% -> risk difference of 0.8, reported in percentage points.
    assert effect.details["risk_difference"] == 0.8
    assert "cohens_h" in effect.details


def test_binary_effect_size_uses_only_the_paired_sample():
    # The effect size must be computed on the SAME paired sample as the p-value
    # (McNemar) and the CI. Ten examples are scored by both models (A passes
    # 5/10, B passes 6/10 -> a paired risk difference of 0.10), plus ten extra
    # examples only B scored. McNemar and the bootstrap CI drop the unpaired
    # extras, so the effect size must drop them too -- otherwise B's pass rate is
    # inflated and the magnitude (which drives PASS/WARN) is computed on a
    # different sample than the significance test.
    examples = [
        Example(id=f"p{i}", scores={"A": 1.0 if i < 5 else 0.0,
                                    "B": 1.0 if i < 6 else 0.0})
        for i in range(10)
    ]
    examples += [Example(id=f"only_b{i}", scores={"B": 1.0}) for i in range(10)]
    data = EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})

    effect = by_check(audit_statistical_validity(data, "A", "B"), "effect_size")

    # Reference values, computed independently on the paired sample only
    # (B 60% vs A 50%): risk difference 0.10, and Cohen's h derived here from the
    # arcsine formula on those proportions rather than by calling the
    # implementation's own cohens_h -- so the assertion validates against a
    # reference, not against the code under test.
    expected_rd = 0.6 - 0.5
    expected_h = 2 * np.arcsin(np.sqrt(0.6)) - 2 * np.arcsin(np.sqrt(0.5))
    assert effect.details["risk_difference"] == pytest.approx(expected_rd)
    assert effect.details["cohens_h"] == pytest.approx(expected_h)
    # The paired proportions must also show through in the prose.
    assert "60.0%" in effect.how_detected and "50.0%" in effect.how_detected
    # Paired truth is a small effect (WARN); the unpaired bug reported it as a
    # medium effect (PASS) -- a flipped verdict.
    assert effect.details["magnitude"] == "small"
    assert effect.status is Status.WARN


def test_continuous_effect_size_reports_a_confidence_interval():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.2, size=120)
    b = a + rng.normal(0.4, 0.2, size=120)   # a clear, real positive effect
    effect = by_check(
        audit_statistical_validity(make_data(a, b), "A", "B", seed=0), "effect_size")
    assert "ci_low" in effect.details and "ci_high" in effect.details
    # The interval brackets the point estimate.
    assert (effect.details["ci_low"] <= effect.details["cohens_d"]
            <= effect.details["ci_high"])
    assert "CI" in effect.how_detected


def test_ci_percentage_preserves_fractional_confidence_levels():
    """Prose must show 92.5% for confidence=0.925, not a rounded 92%."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.2, size=120)
    b = a + rng.normal(0.4, 0.2, size=120)
    data = make_data(a, b)

    fractional = audit_statistical_validity(data, "A", "B", seed=0, confidence=0.925)
    effect_f = by_check(fractional, "effect_size")
    decision_f = by_check(fractional, "decision")
    assert "92.5%" in effect_f.how_detected
    assert "92.5%" in decision_f.how_detected
    assert "92%" not in effect_f.how_detected.replace("92.5%", "")
    assert "92%" not in decision_f.how_detected.replace("92.5%", "")

    default = audit_statistical_validity(data, "A", "B", seed=0, confidence=0.95)
    effect_d = by_check(default, "effect_size")
    decision_d = by_check(default, "decision")
    assert "95%" in effect_d.how_detected
    assert "95%" in decision_d.how_detected
    assert "95.0%" not in effect_d.how_detected
    assert "95.0%" not in decision_d.how_detected


def test_binary_effect_size_reports_a_risk_difference_interval():
    a, b = [0] * 100, [1] * 80 + [0] * 20
    effect = by_check(
        audit_statistical_validity(make_data(a, b), "A", "B", seed=0), "effect_size")
    assert "ci_low" in effect.details and "ci_high" in effect.details
    # The interval is on the (paired) risk difference and brackets it.
    assert (effect.details["ci_low"] <= effect.details["risk_difference"]
            <= effect.details["ci_high"])
    assert "CI" in effect.how_detected


def test_effect_size_ci_does_not_change_the_pass_warn_rule():
    # PASS/WARN stays driven by magnitude, not by whether the CI excludes 0.
    big = by_check(audit_statistical_validity(
        make_data([0] * 100, [1] * 80 + [0] * 20), "A", "B", seed=0), "effect_size")
    assert big.details["magnitude"] in {"medium", "large"}
    assert big.status is Status.PASS
    # A tiny gap is a small effect -> WARN, even though its CI is reported.
    a, b = [0] * 100, [1] * 3 + [0] * 97   # 3% vs 0% -> negligible
    small = by_check(audit_statistical_validity(make_data(a, b), "A", "B", seed=0),
                     "effect_size")
    assert small.details["magnitude"] in {"negligible", "small"}
    assert small.status is Status.WARN


def test_effect_size_ci_is_deterministic():
    data = make_data([0] * 100, [1] * 70 + [0] * 30)
    e1 = by_check(audit_statistical_validity(data, "A", "B", seed=3), "effect_size")
    e2 = by_check(audit_statistical_validity(data, "A", "B", seed=3), "effect_size")
    assert e1.details["ci_low"] == e2.details["ci_low"]
    assert e1.details["ci_high"] == e2.details["ci_high"]


def test_tiny_symmetric_difference_is_equivalent_not_a_failure():
    # Continuous scores essentially identical (well within a 0.05 margin).
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, size=400)
    b = a + rng.normal(0.0, 0.001, size=400)
    decision = by_check(
        audit_statistical_validity(make_data(a, b), "A", "B",
                                   equivalence_margin=0.05, seed=0), "decision")
    assert decision.details["outcome"] == "equivalent"
    assert decision.status is Status.WARN


def test_underpowered_noise_is_inconclusive_not_equivalent():
    # A handful of noisy examples: can't call it a win, can't call it equivalent.
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 1.0, size=8)
    b = a + rng.normal(0.3, 1.0, size=8)
    decision = by_check(
        audit_statistical_validity(make_data(a, b), "A", "B",
                                   equivalence_margin=0.05, seed=0), "decision")
    assert decision.details["outcome"] == "inconclusive"
    assert decision.status is Status.FAIL


def test_precision_reports_minimum_detectable_effect():
    findings = audit_statistical_validity(make_data([0] * 40, [1] * 30 + [0] * 10),
                                          "A", "B")
    precision = by_check(findings, "precision")
    assert "minimum_detectable_effect" in precision.details
    assert precision.details["minimum_detectable_effect"] > 0


def test_underpowered_precision_recommends_more_examples():
    rng = np.random.default_rng(2)
    a = rng.normal(0.5, 1.0, size=8)
    b = a + rng.normal(0.2, 1.0, size=8)
    precision = by_check(
        audit_statistical_validity(make_data(a, b), "A", "B", seed=0), "precision")
    assert precision.status is Status.WARN
    assert "example" in precision.how_to_fix.lower()


def test_every_finding_obeys_the_golden_rule():
    findings = audit_statistical_validity(make_data([0] * 40, [1] * 40), "A", "B")
    for f in findings:
        assert f.why.strip() and f.how_detected.strip() and f.how_to_fix.strip()
        assert f.pillar == "Statistical Validity"


def test_is_deterministic():
    data = make_data([0] * 100, [1] * 90 + [0] * 10)
    f1 = audit_statistical_validity(data, "A", "B", seed=1)
    f2 = audit_statistical_validity(data, "A", "B", seed=1)
    assert [f.details for f in f1] == [f.details for f in f2]


def test_significant_override_lets_a_procedure_own_the_decision():
    # A multiplicity procedure can OWN the significance call: passing `significant`
    # bypasses the strict `p < alpha`. `None` (the default) is a byte-for-byte
    # no-op, so every existing caller is unchanged.
    win = make_data([0] * 200, [1] * 180 + [0] * 20)      # a clear, significant win
    base = audit_statistical_validity(win, "A", "B", seed=0)
    none = audit_statistical_validity(win, "A", "B", seed=0, significant=None)
    assert [f.to_dict() for f in base] == [f.to_dict() for f in none]
    assert by_check(base, "decision").details["p_value"] < 0.05     # really significant

    # Force NOT significant despite p < alpha -> the audit honours the override.
    off = by_check(
        audit_statistical_validity(win, "A", "B", seed=0, significant=False), "decision")
    assert off.details["outcome"] != "significant"

    # Force significant on data whose p >= alpha -> outcome follows the override.
    rng = np.random.default_rng(0)
    flat = rng.normal(0.5, 0.1, size=60)
    tie = make_data(flat, flat + rng.normal(0.0, 0.001, size=60))
    assert by_check(audit_statistical_validity(tie, "A", "B", seed=0),
                    "decision").details["p_value"] >= 0.05
    on = by_check(
        audit_statistical_validity(tie, "A", "B", seed=0, significant=True), "decision")
    assert on.details["outcome"] == "significant"
    # The override forces significance with p >> alpha, so the decision prose must
    # NOT claim `p < alpha` or `p <= alpha` (both false here); it must report the
    # truth, `p > alpha`. A correctness library never emits a false comparison even
    # on a caller-forced decision.
    assert on.details["p_value"] > 0.05
    assert "(< alpha" not in on.how_detected and "(<= alpha" not in on.how_detected
    assert "(> alpha 0.05)" in on.how_detected


def test_significant_prose_is_accurate_when_p_equals_alpha_exactly():
    # A Holm-carried rejection hands this audit significant=True with p == alpha
    # EXACTLY (Holm rejects via adjusted_p <= alpha, so a metric can be rejected on
    # the boundary). The significant prose must then reflect reality: `p <= alpha`,
    # never the false `p < alpha`. Reference for reachability: permutation_test
    # returns (count + 1) / (n_resamples + 1); with n_resamples=39 the smallest
    # attainable p is 1/40 = 0.025, exactly the Holm step threshold alpha/(k-rank)
    # = 0.05/2 for the top-ranked metric of a two-metric suite.
    data = make_data([0.5] * 12, [0.8] * 12)      # constant +0.30 paired difference
    dec = by_check(
        audit_statistical_validity(data, "A", "B", alpha=0.025, n_resamples=39,
                                   seed=0, significant=True), "decision")
    assert dec.details["outcome"] == "significant"
    assert dec.details["p_value"] == 0.025        # p sits exactly on alpha
    assert not (dec.details["p_value"] < 0.025)   # NOT strictly below the threshold
    # The prose must be true at the boundary: `<=`, never the false `<`.
    assert "(< alpha" not in dec.how_detected
    assert "(<= alpha 0.025)" in dec.how_detected


def test_equivalence_ci_and_decision_prose_share_the_reported_alpha():
    # The TOST equivalence interval uses `1 - 2*alpha` and the `equivalent` prose
    # quotes that same interval. Post-refactor `alpha` is the metric's Holm STEP
    # THRESHOLD, and BOTH the interval and the prose read from it, so they cannot
    # drift. With alpha = 0.025 (a Holm step at k=2), the quoted interval must be
    # 1 - 2*0.025 = 95%.
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, size=400)
    b = a + rng.normal(0.0, 0.001, size=400)              # essentially identical
    dec = by_check(
        audit_statistical_validity(make_data(a, b), "A", "B", alpha=0.025,
                                   equivalence_margin=0.05, seed=0, significant=False),
        "decision")
    assert dec.details["outcome"] == "equivalent"
    assert dec.details["alpha"] == 0.025
    assert "95%" in dec.how_detected      # round((1 - 2*0.025) * 100) == 95


# ---------------------------------------------------------------------------
# cluster-aware statistical audit (issue #83)
# ---------------------------------------------------------------------------
from evaltrust.core.schema import EvalData, Example
from evaltrust.audit.statistical import audit_statistical_validity


def _clustered_eval(n_clusters=10, cluster_size=5, effect=0.3):
    import numpy as np
    rng = np.random.default_rng(42)
    examples = []
    i = 0
    for c in range(n_clusters):
        for _ in range(cluster_size):
            a = float(rng.normal(0.0, 0.5))
            b = float(rng.normal(effect, 0.5))
            examples.append(
                Example(id=str(i), scores={"A": a, "B": b}, group_id=f"g{c}")
            )
            i += 1
    return EvalData(models=["A", "B"], examples=examples, source_format="test")


def _unclustered_eval(n=50, effect=0.3):
    import numpy as np
    rng = np.random.default_rng(42)
    examples = []
    for i in range(n):
        a = float(rng.normal(0.0, 0.5))
        b = float(rng.normal(effect, 0.5))
        examples.append(Example(id=str(i), scores={"A": a, "B": b}))
    return EvalData(models=["A", "B"], examples=examples, source_format="test")


def test_clustered_audit_runs_without_error():
    data = _clustered_eval()
    findings = audit_statistical_validity(data, "A", "B", n_resamples=500, seed=0)
    assert len(findings) == 3


def test_clustered_audit_uses_cluster_test_name():
    data = _clustered_eval(effect=1.0)
    findings = audit_statistical_validity(data, "A", "B", n_resamples=500, seed=0)
    decision = next(f for f in findings if f.details.get("check") == "decision")
    assert "cluster" in decision.how_detected.lower()


def test_unclustered_audit_mentions_independence_assumption():
    data = _unclustered_eval()
    findings = audit_statistical_validity(data, "A", "B", n_resamples=500, seed=0)
    decision = next(f for f in findings if f.details.get("check") == "decision")
    assert "independent" in decision.how_detected.lower()


def test_clustered_audit_decision_finding_has_test_key():
    data = _clustered_eval()
    findings = audit_statistical_validity(data, "A", "B", n_resamples=500, seed=0)
    decision = next(f for f in findings if f.details.get("check") == "decision")
    assert "test" in decision.details


def test_clustered_ci_same_sign_and_wider_than_unclustered():
    """Clustered CI must have same sign as non-clustered CI and be wider.

    On clustered data where B outperforms A, the clustered CI must be
    positive (leader-minus-trailer > 0), same sign as the non-clustered
    CI on the same data, and wider (clustering inflates variance).
    """
    from evaltrust.core.schema import EvalData, Example
    import numpy as np

    rng = np.random.default_rng(42)
    examples_clustered = []
    examples_unclustered = []
    i = 0
    for c in range(10):
        cluster_effect = 0.5 if c % 2 == 0 else -0.5
        for _ in range(5):
            a = float(rng.normal(0.0, 0.5))
            b = a + 0.4 + cluster_effect + float(rng.normal(0.0, 0.05))
            examples_clustered.append(
                Example(id=str(i), scores={"A": a, "B": b}, group_id=f"g{c}")
            )
            examples_unclustered.append(
                Example(id=str(i), scores={"A": a, "B": b})
            )
            i += 1

    data_c = EvalData(models=["A", "B"], examples=examples_clustered, source_format="test")
    data_u = EvalData(models=["A", "B"], examples=examples_unclustered, source_format="test")

    findings_c = audit_statistical_validity(data_c, "A", "B", n_resamples=500, seed=0)
    findings_u = audit_statistical_validity(data_u, "A", "B", n_resamples=500, seed=0)

    dec_c = next(f for f in findings_c if f.details.get("check") == "decision")
    dec_u = next(f for f in findings_u if f.details.get("check") == "decision")

    lo_c, hi_c = dec_c.details["ci_low"], dec_c.details["ci_high"]
    lo_u, hi_u = dec_u.details["ci_low"], dec_u.details["ci_high"]

    # Same sign: both CIs should be positive (B > A)
    assert lo_c > 0, f"Clustered CI lower bound should be positive, got {lo_c}"
    assert lo_u > 0, f"Unclustered CI lower bound should be positive, got {lo_u}"

    # Clustered interval must be wider than unclustered
    width_c = hi_c - lo_c
    width_u = hi_u - lo_u
    assert width_c > width_u, (
        f"Clustered CI width {width_c:.4f} should exceed unclustered {width_u:.4f}"
    )
