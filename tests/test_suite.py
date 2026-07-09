"""Tests for auditing a multi-metric suite.

A suite is audited as one single-metric audit per metric, sharing the same model
pair, with the significance threshold corrected for the number of metrics tested
(Bonferroni) so testing many metrics doesn't manufacture false positives.
"""

import json

import pytest

from evaltrust.audit.suite import audit_suite
from evaltrust.audit.verdict import VerdictLevel
from evaltrust.config import AuditConfig
from evaltrust.core.schema import EvalData, Example


def metric_data(a_scores, b_scores):
    examples = [
        Example(id=str(i), scores={"A": float(a), "B": float(b)})
        for i, (a, b) in enumerate(zip(a_scores, b_scores))
    ]
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


def outcome(report, metric):
    (decision,) = [f for f in report.reports[metric].findings
                   if f.details.get("check") == "decision"]
    return decision.details["outcome"]


def test_audits_every_metric():
    suite = {
        "correctness": metric_data([0] * 200, [1] * 180 + [0] * 20),
        "safety": metric_data([1] * 100, [1] * 100),
    }
    report = audit_suite(suite, seed=0)
    assert set(report.reports.keys()) == {"correctness", "safety"}


def test_shares_one_model_pair_across_metrics():
    suite = {"m1": metric_data([0] * 30, [1] * 30),
             "m2": metric_data([1] * 30, [0] * 30)}
    report = audit_suite(suite)
    pairs = {(r.model_a, r.model_b) for r in report.reports.values()}
    assert len(pairs) == 1  # same two models compared for every metric


def test_bonferroni_corrects_alpha_by_metric_count():
    suite = {f"m{i}": metric_data([0] * 40, [1] * 40) for i in range(5)}
    report = audit_suite(suite, alpha=0.05)
    assert report.corrected_alpha == 0.05 / 5
    assert "bonferroni" in report.correction.lower()


def test_no_correction_for_single_metric():
    report = audit_suite({"score": metric_data([0] * 40, [1] * 40)}, alpha=0.05)
    assert report.corrected_alpha == 0.05


def test_overall_level_is_the_worst_metric():
    suite = {
        "good": metric_data([0] * 200, [1] * 180 + [0] * 20),   # clear win -> HIGH
        "noise": metric_data([0, 1] * 60, [1, 0] * 60),         # noise -> LOW
    }
    report = audit_suite(suite, seed=0)
    assert report.overall_level is VerdictLevel.LOW


def test_to_dict_is_json_serializable():
    suite = {"correctness": metric_data([0] * 60, [1] * 55 + [0] * 5),
             "safety": metric_data([1] * 60, [1] * 58 + [0] * 2)}
    d = audit_suite(suite, seed=0).to_dict()
    text = json.dumps(d)
    parsed = json.loads(text)
    assert set(parsed["metrics"].keys()) == {"correctness", "safety"}
    assert parsed["overall_level"] in {"HIGH", "MODERATE", "LOW"}
    assert "corrected_alpha" in parsed
    assert "metric_alphas" in parsed
    assert "adjusted_p" in parsed


# ---------------------------------------------------------------------------
# Holm-Bonferroni step-down correction
# ---------------------------------------------------------------------------

# Two metrics engineered so Holm and Bonferroni disagree at alpha = 0.05, k = 2
# (per-metric threshold alpha/2 = 0.025):
#   - "strong":     20 discordant pairs all favouring B -> McNemar p ~ 2e-6.
#   - "borderline": 10 vs 2 discordant pairs -> McNemar p = 0.03857, which sits
#     between the Bonferroni threshold (0.025) and Holm's second step (0.05).
def _strong():
    return metric_data([0] * 20, [1] * 20)


def _borderline():
    return metric_data([0] * 10 + [1] * 2, [1] * 10 + [0] * 2)


def test_holm_rejects_a_metric_bonferroni_does_not():
    suite = {"strong": _strong(), "borderline": _borderline()}
    bonf = audit_suite(suite, alpha=0.05, correction="bonferroni", seed=0)
    holm = audit_suite(suite, alpha=0.05, correction="holm", seed=0)

    # Both agree the strong metric is a real win.
    assert outcome(bonf, "strong") == "significant"
    assert outcome(holm, "strong") == "significant"
    # The borderline metric (p = 0.0386) clears Holm's second step (0.05) but not
    # the Bonferroni threshold (0.025) — this is the whole point of the feature.
    assert outcome(bonf, "borderline") != "significant"
    assert outcome(holm, "borderline") == "significant"


def test_holm_reruns_each_metric_at_its_effective_alpha():
    # The decision finding records the alpha it used; it must equal the metric's
    # Holm-effective alpha, so its status AND its equivalence CI (which uses
    # 1 - 2*alpha) are consistent with the correction actually applied.
    suite = {"strong": _strong(), "borderline": _borderline()}
    holm = audit_suite(suite, alpha=0.05, correction="holm", seed=0)
    for metric, report in holm.reports.items():
        (decision,) = [f for f in report.findings
                       if f.details.get("check") == "decision"]
        assert decision.details["alpha"] == holm.metric_alphas[metric]
    # strong is rejected first (step alpha/2), borderline second (step alpha/1).
    assert holm.metric_alphas["strong"] == pytest.approx(0.05 / 2)
    assert holm.metric_alphas["borderline"] == pytest.approx(0.05 / 1)


def test_metric_alphas_are_plain_floats_on_every_path():
    # metric_alphas must be plain builtin `float` on BOTH the Holm path (step
    # thresholds) and the Bonferroni/none paths, so the mapping never mixes numpy
    # and builtin scalars. np.float64 subclasses float and serializes fine, but the
    # type must stay uniform; `type(a) is float` rejects np.float64.
    suite = {"strong": _strong(), "borderline": _borderline()}
    for correction in ("holm", "bonferroni", "none"):
        rep = audit_suite(suite, alpha=0.05, correction=correction, seed=0)
        for m, a in rep.metric_alphas.items():
            assert type(a) is float, (correction, m, type(a))
    one = audit_suite({"only": _borderline()}, alpha=0.05, seed=0)   # single-metric path
    assert type(one.metric_alphas["only"]) is float


def test_holm_boundary_metric_prose_is_accurate_at_p_equals_step_threshold():
    # The amendment's repro: a two-metric Holm suite whose top metric's permutation
    # p lands EXACTLY on its step threshold (p = 1/40 = 0.025 = 0.05 / 2). Holm
    # rejects it (adjusted_p <= alpha), so its decision is significant with
    # p == alpha; the prose must read `<= alpha`, never the false `< alpha`.
    def const(gap):
        ex = [Example(id=f"e{i}", scores={"A": 0.5, "B": 0.5 + gap})
              for i in range(12)]
        return EvalData(models=["A", "B"], examples=ex, source_format="test",
                        metadata={})

    cfg = AuditConfig(alpha=0.05, n_resamples=39)
    rep = audit_suite({"m1": const(0.30), "m2": const(0.0)}, "A", "B",
                      config=cfg, correction="holm")
    (dec,) = [f for f in rep.reports["m1"].findings
              if f.details.get("check") == "decision"]
    assert dec.details["outcome"] == "significant"
    assert dec.details["p_value"] == 0.025
    assert dec.details["p_value"] == rep.metric_alphas["m1"]     # p == alpha exactly
    assert "(< alpha" not in dec.how_detected
    assert "(<= alpha 0.025)" in dec.how_detected


def test_holm_keeps_corrected_alpha_scalar_and_names_itself():
    suite = {f"m{i}": _borderline() for i in range(5)}
    holm = audit_suite(suite, alpha=0.05, correction="holm", seed=0)
    # Back-compat: corrected_alpha stays a scalar = Holm's most conservative step.
    assert holm.corrected_alpha == pytest.approx(0.05 / 5)
    assert "holm" in holm.correction.lower()


def test_holm_adjusted_p_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    suite = {"strong": _strong(), "borderline": _borderline()}
    holm = audit_suite(suite, alpha=0.05, correction="holm", seed=0)
    # Recover the raw per-metric p-values from an uncorrected run and adjust them
    # with the reference implementation.
    raw = audit_suite(suite, alpha=0.05, correction="none", seed=0)
    metrics = list(suite.keys())
    pvals = []
    for m in metrics:
        (dec,) = [f for f in raw.reports[m].findings
                  if f.details.get("check") == "decision"]
        pvals.append(dec.details["p_value"])
    _, ref_adjusted, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    got = [holm.adjusted_p[m] for m in metrics]
    assert got == pytest.approx(list(ref_adjusted))


def test_correction_none_applies_no_correction():
    suite = {"a": _borderline(), "b": _borderline()}
    none = audit_suite(suite, alpha=0.05, correction="none", seed=0)
    assert none.corrected_alpha == 0.05
    assert all(a == 0.05 for a in none.metric_alphas.values())


def test_invalid_correction_raises():
    suite = {"a": _borderline(), "b": _borderline()}
    with pytest.raises(ValueError):
        audit_suite(suite, correction="bogus")


def test_legacy_correct_false_means_no_correction():
    suite = {"a": _borderline(), "b": _borderline()}
    report = audit_suite(suite, alpha=0.05, correct=False, seed=0)
    assert report.corrected_alpha == 0.05


def test_correction_read_from_config():
    suite = {"strong": _strong(), "borderline": _borderline()}
    cfg = AuditConfig(alpha=0.05, correction="holm")
    report = audit_suite(suite, config=cfg, seed=0)
    # The config-driven Holm run rejects the borderline metric.
    assert outcome(report, "borderline") == "significant"
    assert "holm" in report.correction.lower()


def test_explicit_correction_overrides_config():
    suite = {"strong": _strong(), "borderline": _borderline()}
    cfg = AuditConfig(alpha=0.05, correction="holm")
    # An explicit argument beats the config file.
    report = audit_suite(suite, config=cfg, correction="bonferroni", seed=0)
    assert "bonferroni" in report.correction.lower()
    assert outcome(report, "borderline") != "significant"


def _holm_pvalues(suite, alpha=0.05):
    raw = audit_suite(suite, alpha=alpha, correction="none", seed=0)
    pvals = []
    for m in suite:
        (dec,) = [f for f in raw.reports[m].findings
                  if f.details.get("check") == "decision"]
        pvals.append(dec.details["p_value"])
    return pvals


def test_holm_outcomes_match_the_rejection_mask():
    # The per-metric decision outcome ("significant" or not) must agree with
    # holm_bonferroni's reject array — the audit's re-run and the reference must
    # not disagree.
    from evaltrust.stats.multiplicity import holm_bonferroni
    suite = {"strong": _strong(), "borderline": _borderline()}
    holm = audit_suite(suite, alpha=0.05, correction="holm", seed=0)
    rejected, _ = holm_bonferroni(_holm_pvalues(suite), 0.05)
    significant = [outcome(holm, m) == "significant" for m in suite]
    assert significant == rejected


def test_holm_suite_decision_matches_statsmodels_reject_mask():
    # The suite's CARRIED per-metric decision must equal statsmodels' Holm reject
    # mask for the same p-values -- validated against the REFERENCE directly, not
    # against the library's own holm_bonferroni. Three metrics of decreasing
    # strength give a non-trivial mask (one rejected, two retained).
    from statsmodels.stats.multitest import multipletests
    suite = {"strong": _strong(),
             "borderline": _borderline(),
             "noise": metric_data([0, 1] * 40, [1, 0] * 40)}
    holm = audit_suite(suite, alpha=0.05, correction="holm", seed=0)
    ref_reject = list(multipletests(_holm_pvalues(suite), alpha=0.05,
                                    method="holm")[0])
    got = [outcome(holm, m) == "significant" for m in suite]
    assert got == ref_reject
    assert got.count(True) == 1 and got.count(False) == 2   # non-trivial mask


def test_holm_boundary_decision_is_carried_not_re_derived():
    # A p-value sitting EXACTLY on its Holm step threshold (p == alpha/(k-rank)) is
    # the one place a strict `p < alpha` re-derivation and Holm's `adjusted_p <=
    # alpha` disagree. Holm now OWNS the decision — it is carried into each metric's
    # audit, not re-derived from the threshold — so the two agree structurally with
    # NO nextafter/ULP nudge. p = 0.025 at k=2, alpha=0.05 lands on the first step
    # (0.05 / 2 = 0.025).
    from statsmodels.stats.multitest import multipletests

    from evaltrust.audit.suite import _holm_step_thresholds
    from evaltrust.stats.multiplicity import holm_bonferroni

    pvals, alpha = [0.025, 0.5], 0.05
    ref_reject = list(multipletests(pvals, alpha=alpha, method="holm")[0])
    rejected, _ = holm_bonferroni(pvals, alpha)
    assert rejected == ref_reject == [True, False]   # matches statsmodels' reject mask
    # The step threshold is reported verbatim: p == threshold, NOT nudged up an ULP.
    thresholds = _holm_step_thresholds(pvals, rejected, alpha)
    assert thresholds[0] == pytest.approx(0.025)
    assert thresholds[0] == pvals[0]
    # A strict `p < threshold` re-derivation would MISS the boundary metric; it is
    # carrying Holm's `rejected` that makes it come out significant.
    assert not (pvals[0] < thresholds[0])


def test_uncorrected_paths_still_decide_by_strict_p_less_than_alpha():
    # The refactor adds a `significant` override, but ONLY Holm uses it. The
    # bonferroni, none, and single-metric paths must still decide exactly as main
    # did — a strict `p < metric_alpha`. Validate the audit's outcome against that
    # rule recomputed independently from the raw p-values (a reference, never the
    # audit's own decision), so this is a byte-for-byte guard on those paths.
    suite = {"strong": _strong(), "borderline": _borderline(), "mid": _borderline()}
    raw_p = dict(zip(suite, _holm_pvalues(suite)))
    for correction in ("none", "bonferroni"):
        rep = audit_suite(suite, alpha=0.05, correction=correction, seed=0)
        for m in suite:
            expected = raw_p[m] < rep.metric_alphas[m]
            assert (outcome(rep, m) == "significant") is expected
    # single-metric path (k == 1) never corrects and never overrides.
    one = {"only": _borderline()}
    p1 = _holm_pvalues(one)[0]
    rep1 = audit_suite(one, alpha=0.05, seed=0)
    assert (outcome(rep1, "only") == "significant") is (p1 < rep1.metric_alphas["only"])


def test_bonferroni_adjusted_p_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    suite = {"strong": _strong(), "borderline": _borderline()}
    bonf = audit_suite(suite, alpha=0.05, correction="bonferroni", seed=0)
    _, ref_adjusted, _, _ = multipletests(
        _holm_pvalues(suite), alpha=0.05, method="bonferroni")
    got = [bonf.adjusted_p[m] for m in suite]
    assert got == pytest.approx(list(ref_adjusted))
