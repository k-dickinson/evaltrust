"""Unpaired two-sample comparison audit for run-level (aggregate) scores.

When a harness emits only a single aggregate score per run, per-example
pairing is impossible.  This module implements the *second comparison path*
described in issue #132: an unpaired two-sample analysis that produces three
findings parallel to the paired Statistical Validity audit:

``decision``
    Is the gap between A and B real?  Based on:
    - Bootstrap P(A > B) with a percentile CI.
    - Mann-Whitney U two-sided p-value as a rank-test complement.

``effect_size``
    How large is the difference?  Reported as P(A > B) distance from 0.5
    (so 0.0 = no difference, +0.5 = A always wins, -0.5 = B always wins)
    together with the raw score means and standard deviations.

``precision``
    Was the sample large enough?  Reports the per-model run counts and warns
    when either model has fewer than 10 runs (below which bootstrap CIs are
    very wide and the Mann-Whitney test has limited power).

The *paired path* (``audit_statistical_validity``) is completely unaffected:
it still consumes ``EvalData`` and operates on per-example score differences.
This module consumes ``RunLevelData`` and never touches ``EvalData``.
"""

from __future__ import annotations

import numpy as np

from ..core.schema import Finding, RunLevelData, Status
from ..stats.two_sample import bootstrap_p_a_gt_b, mann_whitney_u

PILLAR = "Statistical Validity"

# Minimum per-model run count below which precision is flagged as a warning.
_MIN_RUNS_RECOMMENDED = 10


def audit_two_sample(
    data: RunLevelData,
    alpha: float = 0.05,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> list[Finding]:
    """Run the unpaired two-sample audit on run-level scores.

    Parameters
    ----------
    data:
        A :class:`~evaltrust.core.schema.RunLevelData` object produced by
        :func:`~evaltrust.core.ingest.load_run_level`.
    alpha:
        Significance level for the Mann-Whitney p-value (default 0.05).
    confidence:
        Coverage for the bootstrap CI on P(A > B) (default 0.95).
    n_resamples:
        Bootstrap iterations (default 10 000).
    seed:
        RNG seed for reproducibility (default 0).

    Returns
    -------
    list[Finding]
        Three findings: ``decision``, ``effect_size``, ``precision``.
    """
    a, b = data.scores_a, data.scores_b
    model_a, model_b = data.model_a, data.model_b

    p_hat, ci_lo, ci_hi = bootstrap_p_a_gt_b(
        a, b,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    u_stat, p_mw = mann_whitney_u(a, b)

    # Determine leader: the model with the higher expected win probability.
    if p_hat >= 0.5:
        leader, trailer = model_a, model_b
        p_leader = p_hat
        ci_leader_lo, ci_leader_hi = ci_lo, ci_hi
    else:
        leader, trailer = model_b, model_a
        p_leader = 1.0 - p_hat
        ci_leader_lo, ci_leader_hi = 1.0 - ci_hi, 1.0 - ci_lo

    significant = p_mw < alpha

    return [
        _decision(
            significant=significant,
            p_mw=p_mw,
            alpha=alpha,
            p_hat=p_hat,
            p_leader=p_leader,
            ci_leader_lo=ci_leader_lo,
            ci_leader_hi=ci_leader_hi,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            confidence=confidence,
            leader=leader,
            trailer=trailer,
            model_a=model_a,
            model_b=model_b,
            n_a=data.n_a,
            n_b=data.n_b,
        ),
        _effect_size(
            a=a,
            b=b,
            p_hat=p_hat,
            leader=leader,
            trailer=trailer,
        ),
        _precision(
            n_a=data.n_a,
            n_b=data.n_b,
            model_a=model_a,
            model_b=model_b,
            significant=significant,
        ),
    ]


# ---------------------------------------------------------------------------
# Internal finding builders
# ---------------------------------------------------------------------------

def _decision(
    *,
    significant: bool,
    p_mw: float,
    alpha: float,
    p_hat: float,
    p_leader: float,
    ci_leader_lo: float,
    ci_leader_hi: float,
    ci_lo: float,
    ci_hi: float,
    confidence: float,
    leader: str,
    trailer: str,
    model_a: str,
    model_b: str,
    n_a: int,
    n_b: int,
) -> Finding:
    conf_pct = f"{confidence * 100:g}"
    ci_str = f"[{ci_leader_lo:.3f}, {ci_leader_hi:.3f}]"

    # The CI straddles 0.5 when we cannot distinguish the two models.
    ci_straddles_half = ci_lo <= 0.5 <= ci_hi

    if significant and not ci_straddles_half:
        title = f"{leader} has a higher run-level score than {trailer}"
        status = Status.PASS
        how = (
            f"Mann-Whitney U gave p = {p_mw:.4f} (< alpha {alpha}). "
            f"Bootstrap P({leader} > {trailer}) = {p_leader:.3f}, "
            f"{conf_pct}% CI {ci_str} "
            f"(based on {n_a} runs for {model_a} and {n_b} for {model_b})."
        )
        fix = "The run-level advantage is statistically supported. Safe to act on."
        outcome = "significant"
    elif ci_straddles_half and p_hat > 0.5 - 0.05 and p_hat < 0.5 + 0.05:
        # Near-0.5 P(A>B) and CI straddles 0.5: consistent with no difference.
        title = f"{model_a} and {model_b} appear equivalent on run-level scores"
        status = Status.WARN
        how = (
            f"P({model_a} > {model_b}) = {p_hat:.3f}, "
            f"{conf_pct}% CI {ci_str} — the interval includes 0.5, "
            f"consistent with no systematic difference. "
            f"Mann-Whitney p = {p_mw:.4f} (alpha {alpha}). "
            f"({n_a} runs for {model_a}, {n_b} for {model_b}.)"
        )
        fix = "Treat the two models as equivalent on aggregate quality. Decide on cost or speed."
        outcome = "equivalent"
    else:
        title = f"Run-level comparison of {model_a} vs {model_b} is inconclusive"
        status = Status.FAIL
        how = (
            f"Mann-Whitney p = {p_mw:.4f} (not < alpha {alpha}), and the "
            f"{conf_pct}% CI on P({leader} > {trailer}) = {ci_str} "
            f"is too wide to draw a firm conclusion. "
            f"({n_a} runs for {model_a}, {n_b} for {model_b}.)"
        )
        fix = "Don't call a winner yet. Collect more runs per model first."
        outcome = "inconclusive"

    return Finding(
        pillar=PILLAR,
        title=title,
        status=status,
        why=(
            "A raw score gap between aggregate run scores means nothing until "
            "you know if it reflects a real distributional difference or just "
            "run-to-run noise. The Mann-Whitney test and P(A > B) answer that "
            "without requiring per-example pairing."
        ),
        how_detected=how,
        how_to_fix=fix,
        details={
            "check": "decision",
            "outcome": outcome,
            "p_value_mann_whitney": p_mw,
            "p_a_gt_b": p_hat,
            "alpha": alpha,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "n_a": n_a,
            "n_b": n_b,
            "comparison_path": "unpaired_two_sample",
        },
    )


def _effect_size(
    *,
    a: np.ndarray,
    b: np.ndarray,
    p_hat: float,
    leader: str,
    trailer: str,
) -> Finding:
    """Effect-size finding: P(A>B) distance from 0.5 + descriptive stats."""
    # Distance of P(A>B) from 0.5.  Ranges 0–0.5 so it mirrors Cohen's d sign.
    # Named "common language effect size" (CLES), also known as A12 statistic.
    cles = abs(p_hat - 0.5)

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    std_a, std_b = float(np.std(a, ddof=1)) if a.size > 1 else 0.0, \
                   float(np.std(b, ddof=1)) if b.size > 1 else 0.0

    # Map CLES to a label (Vargha-Delaney A12 thresholds: 0.06 small, 0.14 medium, 0.21 large)
    if cles < 0.06:
        magnitude = "negligible"
    elif cles < 0.14:
        magnitude = "small"
    elif cles < 0.21:
        magnitude = "medium"
    else:
        magnitude = "large"

    meaningful = magnitude in {"medium", "large"}

    how = (
        f"P(A > B) = {p_hat:.3f} → common-language effect size = {cles:.3f} "
        f"({magnitude} by Vargha-Delaney thresholds). "
        f"{leader}: mean {mean_a:.4f} ± {std_a:.4f} (SD); "
        f"{trailer}: mean {mean_b:.4f} ± {std_b:.4f} (SD)."
    )

    return Finding(
        pillar=PILLAR,
        title=f"Run-level effect size is {magnitude}",
        status=Status.PASS if meaningful else Status.WARN,
        why=(
            "Significance says a gap exists; effect size says how large it is. "
            "P(A > B) is the probability that a randomly drawn run from A beats "
            "a run from B — interpretable without assuming normality."
        ),
        how_detected=how,
        how_to_fix=(
            f"The run-level advantage of {leader} is large enough to matter practically."
            if meaningful
            else
            "The gap may be too small to matter in practice. Weigh against cost and latency."
        ),
        details={
            "check": "effect_size",
            "p_a_gt_b": p_hat,
            "common_language_effect_size": cles,
            "magnitude": magnitude,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "std_a": std_a,
            "std_b": std_b,
            "comparison_path": "unpaired_two_sample",
        },
    )


def _precision(
    *,
    n_a: int,
    n_b: int,
    model_a: str,
    model_b: str,
    significant: bool,
) -> Finding:
    """Precision finding: warn when either model has very few runs."""
    min_n = min(n_a, n_b)
    sufficient = min_n >= _MIN_RUNS_RECOMMENDED

    if significant and sufficient:
        title = "Run count was sufficient"
        status = Status.PASS
        how = (
            f"{model_a} has {n_a} runs, {model_b} has {n_b} runs — "
            f"enough for a reliable Mann-Whitney test and bootstrap CI."
        )
        fix = "The run count is adequate for this comparison."
    elif significant and not sufficient:
        title = "Significant result but run count is low — interpret cautiously"
        status = Status.WARN
        how = (
            f"The test was significant, but {model_a} has {n_a} runs and "
            f"{model_b} has {n_b} runs; at least {_MIN_RUNS_RECOMMENDED} runs "
            f"per model is recommended for stable bootstrap CIs."
        )
        fix = (
            f"Collect more runs (aim for ≥ {_MIN_RUNS_RECOMMENDED} per model) "
            "to narrow the confidence interval before acting on this result."
        )
    else:
        title = "Run count may be too small"
        status = Status.WARN
        shortage = _MIN_RUNS_RECOMMENDED - min_n
        how = (
            f"{model_a} has {n_a} runs, {model_b} has {n_b} runs. "
            f"The minimum recommended is {_MIN_RUNS_RECOMMENDED} runs per model; "
            f"{'both meet' if sufficient else 'at least one falls below'} that threshold."
        )
        fix = (
            f"Collect ~{shortage} more runs for the smaller sample "
            f"(at least {_MIN_RUNS_RECOMMENDED} total per model) to get a "
            "reliable comparison."
        )

    return Finding(
        pillar=PILLAR,
        title=title,
        status=status,
        why=(
            "Bootstrap CIs and Mann-Whitney become unreliable with very small "
            "samples.  This reports the per-model run counts so you know how "
            "much evidence the comparison rests on."
        ),
        how_detected=how,
        how_to_fix=fix,
        details={
            "check": "precision",
            "n_a": n_a,
            "n_b": n_b,
            "min_n": min_n,
            "sufficient": sufficient,
            "comparison_path": "unpaired_two_sample",
        },
    )
