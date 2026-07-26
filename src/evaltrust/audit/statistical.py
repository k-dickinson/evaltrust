"""Statistical Validity audit.

Three findings: decision (significant / equivalent / inconclusive), effect_size,
and precision. Pass/fail data uses McNemar plus a proportion effect size;
continuous scores use a permutation test plus Cohen's d.
"""

from __future__ import annotations

import numpy as np

from ..core.schema import EvalData, Finding, Status
from ..stats.effect import (
    cohens_d_paired,
    cohens_d_paired_along_rows,
    cohens_h,
    magnitude_label,
)
from ..stats.paired import mcnemar_exact
from ..stats.power import minimum_detectable_effect, required_n
from ..stats.resampling import (
    bootstrap_ci,
    bootstrap_ci_clustered,
    bootstrap_statistic_ci,
    permutation_test,
    permutation_test_clustered,
)

PILLAR = "Statistical Validity"


def audit_statistical_validity(
    data: EvalData,
    model_a: str,
    model_b: str,
    alpha: float = 0.05,
    equivalence_margin: float = 0.05,
    power_target: float = 0.8,
    smallest_meaningful_effect: float = 0.2,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
    *,
    significant: bool | None = None,
) -> list[Finding]:
    """Audit whether the gap between two models is real, meaningful evidence.

    ``significant`` stays ``None`` unless a multiplicity procedure (Holm) already
    decided rejection and passes the result in; then ``alpha`` is only the step
    threshold for the TOST interval, not the operative test.
    """
    raw = data.differences(model_a, model_b)  # score_b - score_a
    n = int(raw.size)

    # Orient toward the leader so reported numbers favour the winner.
    if float(raw.mean()) >= 0:
        leader, trailer, diffs = model_b, model_a, raw
    else:
        leader, trailer, diffs = model_a, model_b, -raw

    binary = _is_binary(data, model_a, model_b)
    clustered = data.has_clusters
    cluster_note = (
        " (examples treated as independent — supply a group_id per example "
        "to enable cluster-aware resampling)"
        if not clustered
        else ""
    )
    binary_cluster_note = (
        " (McNemar's test does not yet use cluster-aware resampling; "
        "examples are treated as independent for this test)"
        if clustered
        else cluster_note
    )

    # --- significance ---
    if binary:
        b_only, a_only = _discordant_counts(data, leader, trailer)
        p = mcnemar_exact(b_only, a_only)
        test_name = "McNemar's exact test"
        test_detail = (f"{b_only + a_only} discordant pairs "
                       f"({b_only} for {leader}, {a_only} for {trailer})"
                       + binary_cluster_note)
    elif clustered:
        clusters = data.cluster_groups(leader, trailer)
        p = permutation_test_clustered(clusters, n_resamples=n_resamples, seed=seed)
        k = len(clusters)
        test_name = "a cluster-aware paired permutation test"
        test_detail = f"{n} paired examples across {k} clusters"
    else:
        p = permutation_test(diffs, n_resamples=n_resamples, seed=seed)
        test_name = "a paired permutation test"
        test_detail = f"{n} paired examples{cluster_note}"
    # Strict `p < alpha` unless a multiplicity procedure supplied the decision.
    if significant is None:
        significant = p < alpha

    # --- confidence interval (leader-minus-trailer) ---
    if clustered and not binary:
        clusters = data.cluster_groups(trailer, leader)
        lo, hi = bootstrap_ci_clustered(
            clusters, confidence=confidence, n_resamples=n_resamples, seed=seed
        )
    else:
        lo, hi = bootstrap_ci(diffs, confidence=confidence,
                              n_resamples=n_resamples, seed=seed)

    # --- equivalence (TOST): (1 - 2*alpha) CI on the signed gap inside the margin ---
    # Route through bootstrap_ci_clustered when clusters are present so the
    # "statistically equivalent" verdict uses the same variance estimator as
    # the significance CI — plain bootstrap_ci would assume independence and
    # produce an interval that is too narrow on clustered data (TOST over-optimism).
    if clustered and not binary:
        tost_clusters = data.cluster_groups(model_a, model_b)
        eq_lo, eq_hi = bootstrap_ci_clustered(
            tost_clusters, confidence=1 - 2 * alpha, n_resamples=n_resamples, seed=seed
        )
    else:
        eq_lo, eq_hi = bootstrap_ci(raw, confidence=1 - 2 * alpha,
                                    n_resamples=n_resamples, seed=seed)
    equivalent = eq_lo > -equivalence_margin and eq_hi < equivalence_margin

    if significant:
        outcome = "significant"
    elif equivalent:
        outcome = "equivalent"
    else:
        outcome = "inconclusive"

    return [
        _decision(outcome, p, alpha, test_name, test_detail, lo, hi, confidence,
                  equivalence_margin, leader, trailer),
        _effect_size(data, diffs, binary, leader, trailer,
                     confidence, n_resamples, seed),
        _precision(outcome, n, alpha, power_target, smallest_meaningful_effect),
    ]


def _is_binary(data: EvalData, model_a: str, model_b: str) -> bool:
    vals = []
    for ex in data.examples:
        for m in (model_a, model_b):
            if m in ex.scores:
                vals.append(ex.scores[m])
    return bool(vals) and set(np.unique(vals)).issubset({0.0, 1.0})


def _discordant_counts(data, leader, trailer) -> tuple[int, int]:
    b_only = a_only = 0
    for ex in data.examples:
        if leader in ex.scores and trailer in ex.scores:
            lead, trail = ex.scores[leader], ex.scores[trailer]
            if lead == 1 and trail == 0:
                b_only += 1
            elif lead == 0 and trail == 1:
                a_only += 1
    return b_only, a_only


def _decision(outcome, p, alpha, test_name, test_detail, lo, hi, confidence,
              margin, leader, trailer) -> Finding:
    conf_pct = f"{confidence * 100:g}"
    ci = f"[{lo:+.4f}, {hi:+.4f}]"
    cap = test_name[0].upper() + test_name[1:]  # keep "McNemar" intact

    if outcome == "significant":
        title = f"{leader} is significantly better than {trailer}"
        status = Status.PASS
        # Holm can reject on the boundary p == alpha, and the public `significant`
        # override can force it on p > alpha, so pick the operator from reality.
        op = "<" if p < alpha else ("<=" if p == alpha else ">")
        how = (f"{cap} over {test_detail} gave p = {p:.4f} "
               f"({op} alpha {alpha}); the {conf_pct}% interval for the gap is {ci}.")
        fix = "It's a real improvement. Safe to act on."
    elif outcome == "equivalent":
        title = f"{leader} and {trailer} are statistically equivalent"
        status = Status.WARN
        how = (f"The gap was not significant (p = {p:.4f}) and the "
               f"{round((1 - 2 * alpha) * 100)}% interval falls within "
               f"+/-{margin}, so any real difference is smaller than that margin. "
               f"{test_detail}.")
        fix = "Treat them as equal on quality. Decide on cost or speed."
    else:  # inconclusive
        title = f"Improvement of {leader} over {trailer} is inconclusive"
        status = Status.FAIL
        how = (f"{cap} gave p = {p:.4f} (not significant), and "
               f"the interval {ci} is too wide to rule out a real difference. "
               f"That's missing data, not proof the two are equal. "
               f"{test_detail}.")
        fix = "Don't call a winner yet. Collect more examples first."

    return Finding(
        pillar=PILLAR, title=title, status=status,
        why=(
            "A raw gap means nothing until you know whether it is a real "
            "improvement, no real difference, or simply too little data to tell. "
            "Shipping on the wrong one of these is the mistake this prevents."
        ),
        how_detected=how, how_to_fix=fix,
        details={"check": "decision", "outcome": outcome, "p_value": p,
                 "alpha": alpha, "ci_low": lo, "ci_high": hi, "test": test_name},
    )


def _fmt_bound(x: float) -> str:
    """Format a Cohen's d CI bound, keeping infinities readable."""
    if np.isposinf(x):
        return "+inf"
    if np.isneginf(x):
        return "-inf"
    return f"{x:+.3f}"


def _effect_size(data, diffs, binary, leader, trailer,
                 confidence=0.95, n_resamples=10_000, seed=0) -> Finding:
    conf_pct = f"{confidence * 100:g}"
    if binary:
        # Pass rates come from the paired sample (the same examples McNemar and
        # the CI use), so the effect size is computed on the same data as the test.
        leader_scores, trailer_scores = data.paired_scores(leader, trailer)
        p_leader = float(leader_scores.mean())
        p_trailer = float(trailer_scores.mean())
        rd = p_leader - p_trailer
        h = cohens_h(p_leader, p_trailer)
        magnitude = magnitude_label(h)
        # CI on the paired risk difference (the mean of the paired differences).
        rd_lo, rd_hi = bootstrap_statistic_ci(
            diffs, lambda m: m.mean(axis=-1),
            confidence=confidence, n_resamples=n_resamples, seed=seed)
        how = (f"{leader} passed {p_leader:.1%} vs {trailer}'s {p_trailer:.1%}, a "
               f"{rd * 100:+.1f} point gap ({conf_pct}% CI "
               f"[{rd_lo * 100:+.1f}, {rd_hi * 100:+.1f}] pts; "
               f"Cohen's h {h:+.3f}, {magnitude}).")
        details = {"check": "effect_size", "risk_difference": rd,
                   "cohens_h": h, "magnitude": magnitude,
                   "ci_low": rd_lo, "ci_high": rd_hi}
    else:
        d = cohens_d_paired(diffs)
        magnitude = magnitude_label(d)
        d_lo, d_hi = bootstrap_statistic_ci(
            diffs, cohens_d_paired_along_rows,
            confidence=confidence, n_resamples=n_resamples, seed=seed)
        d_str = "infinite" if np.isinf(d) else f"{d:+.3f}"
        how = (f"Cohen's d on the paired differences was {d_str} "
               f"({conf_pct}% CI [{_fmt_bound(d_lo)}, {_fmt_bound(d_hi)}]), "
               f"a {magnitude} effect by conventional thresholds.")
        details = {"check": "effect_size", "cohens_d": float(d),
                   "magnitude": magnitude, "ci_low": d_lo, "ci_high": d_hi}

    meaningful = magnitude in {"medium", "large"}
    return Finding(
        pillar=PILLAR,
        title=f"Effect size is {magnitude}",
        status=Status.PASS if meaningful else Status.WARN,
        why=(
            "Significance says a gap is real; effect size says whether it is big "
            "enough to matter. A tiny gap can be real yet make no practical "
            "difference in production."
        ),
        how_detected=how,
        how_to_fix=(
            f"The advantage of {leader} is large enough to matter."
            if meaningful else
            "Gap may be too small to matter. Weigh it against cost and risk."
        ),
        details=details,
    )


def _precision(outcome, n, alpha, power_target, smallest_meaningful_effect) -> Finding:
    mde = minimum_detectable_effect(n, power=power_target, alpha=alpha)
    conclusive = outcome in {"significant", "equivalent"}

    if conclusive:
        title = "Sample size was sufficient"
        status = Status.PASS
        how = (f"With {n} examples the evaluation reached a conclusion; it could "
               f"reliably detect effects down to Cohen's d ~ {mde:.2f} at "
               f"{power_target:.0%} power.")
        fix = "The sample was large enough for this comparison."
    else:
        need_n = required_n(smallest_meaningful_effect, power=power_target, alpha=alpha)
        extra = max(0, need_n - n)
        title = "Sample size may be too small"
        status = Status.WARN
        how = (f"With only {n} examples the smallest effect reliably detectable at "
               f"{power_target:.0%} power is Cohen's d ~ {mde:.2f}; a smaller real "
               "difference would be missed.")
        fix = (f"Collect ~{extra} more examples (~{need_n} total) to catch a "
               "small effect.")

    return Finding(
        pillar=PILLAR, title=title, status=status,
        why=(
            "An underpowered evaluation can miss a real difference entirely, so "
            "'inconclusive' might just mean 'not enough data'. This reports how "
            "small a difference the sample could actually have caught."
        ),
        how_detected=how, how_to_fix=fix,
        details={"check": "precision", "n": n,
                 "minimum_detectable_effect": mde,
                 "conclusive": conclusive},
    )
