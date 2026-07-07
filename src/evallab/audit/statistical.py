"""Statistical Validity audit.

The question: is the reported gap between two models real evidence, or noise?
Four complementary views, each a separate finding so nothing hides:

  1. Significance     - a paired permutation test: could this gap arise by chance?
  2. Confidence CI    - a paired bootstrap CI on the gap: does it exclude zero?
  3. Effect size      - Cohen's d: is the gap big enough to matter, not just real?
  4. Power            - was the sample large enough to have detected this gap?

Together they stop the two classic mistakes: shipping on noise (a lucky gap) and
shipping on a real-but-trivial gap that won't survive contact with production.
"""

from __future__ import annotations

from ..core.schema import EvalData, Finding, Status
from ..stats.effect import cohens_d_paired, magnitude_label
from ..stats.power import achieved_power, required_n
from ..stats.resampling import bootstrap_ci, permutation_test

PILLAR = "Statistical Validity"


def audit_statistical_validity(
    data: EvalData,
    model_a: str,
    model_b: str,
    alpha: float = 0.05,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> list[Finding]:
    diffs = data.differences(model_a, model_b)
    n = int(diffs.size)

    mean_diff = float(diffs.mean())
    leader, trailer = (model_b, model_a) if mean_diff >= 0 else (model_a, model_b)
    gap = abs(mean_diff)

    p = permutation_test(diffs, n_resamples=n_resamples, seed=seed)
    lo, hi = bootstrap_ci(diffs, confidence=confidence,
                          n_resamples=n_resamples, seed=seed)
    d = cohens_d_paired(diffs)
    magnitude = magnitude_label(d)
    power = achieved_power(d, n=n, alpha=alpha)
    need_n = required_n(d, power=0.8, alpha=alpha)
    conf_pct = round(confidence * 100)

    return [
        _significance(p, alpha, gap, leader, trailer, n),
        _confidence_interval(lo, hi, conf_pct, leader, trailer),
        _effect_size(d, magnitude, leader, trailer),
        _power(power, need_n, n, magnitude, leader, trailer),
    ]


def _significance(p, alpha, gap, leader, trailer, n) -> Finding:
    significant = p < alpha
    return Finding(
        pillar=PILLAR,
        title=("Improvement is statistically significant" if significant
               else "Improvement is not statistically significant"),
        status=Status.PASS if significant else Status.FAIL,
        why=(
            f"A raw gap between {leader} and {trailer} means nothing until you "
            "rule out chance. If the gap could easily arise from noise, shipping "
            "on it is shipping on luck."
        ),
        how_detected=(
            f"A paired permutation test over {n} examples (randomly flipping the "
            f"sign of each per-example difference) gave p = {p:.4f} "
            f"against alpha = {alpha}."
        ),
        how_to_fix=(
            "The gap is unlikely under chance — no action needed."
            if significant else
            f"Do not claim {leader} is better yet. Collect more examples or "
            "confirm the gap is real before deciding."
        ),
        details={"check": "significance", "p_value": p, "alpha": alpha,
                 "significant": significant, "n": n},
    )


def _confidence_interval(lo, hi, conf_pct, leader, trailer) -> Finding:
    excludes_zero = lo > 0 or hi < 0
    return Finding(
        pillar=PILLAR,
        title=(f"{conf_pct}% confidence interval excludes zero" if excludes_zero
               else f"{conf_pct}% confidence interval overlaps zero"),
        status=Status.PASS if excludes_zero else Status.WARN,
        why=(
            "The confidence interval is the range of gaps consistent with your "
            "data. If it includes zero, 'no difference' is still plausible and "
            f"the two models are statistically indistinguishable."
        ),
        how_detected=(
            f"A paired bootstrap (resampling examples with replacement) put the "
            f"{conf_pct}% interval for the {leader}-minus-{trailer} gap at "
            f"[{lo:+.4f}, {hi:+.4f}]."
        ),
        how_to_fix=(
            "The interval is clear of zero — the direction of the gap is solid."
            if excludes_zero else
            "Treat the models as tied for now. More examples will narrow the "
            "interval; if it still spans zero, there is no real difference."
        ),
        details={"check": "confidence_interval", "ci_low": lo, "ci_high": hi,
                 "excludes_zero": excludes_zero},
    )


def _effect_size(d, magnitude, leader, trailer) -> Finding:
    meaningful = magnitude in {"medium", "large"}
    d_str = "infinite" if d == float("inf") or d == float("-inf") else f"{d:+.3f}"
    return Finding(
        pillar=PILLAR,
        title=f"Effect size is {magnitude}",
        status=Status.PASS if meaningful else Status.WARN,
        why=(
            "Significance says a gap is real; effect size says whether it is big "
            "enough to care about. A tiny gap can be real yet make no practical "
            "difference in production."
        ),
        how_detected=(
            f"Cohen's d on the paired differences was {d_str}, which is a "
            f"{magnitude} effect by conventional thresholds."
        ),
        how_to_fix=(
            f"The advantage of {leader} over {trailer} is large enough to matter."
            if meaningful else
            "The gap may be too small to be worth acting on. Weigh it against "
            "cost, latency, and risk before switching models."
        ),
        details={"check": "effect_size", "cohens_d": d, "magnitude": magnitude},
    )


def _power(power, need_n, n, magnitude, leader, trailer) -> Finding:
    adequate = power >= 0.8
    if need_n >= 10_000_000:
        fix_more = ("There is no measurable effect to power for — the models "
                    "look equivalent on this benchmark.")
        extra = None
    else:
        extra = max(0, need_n - n)
        fix_more = (f"Collect about {extra} more comparable examples "
                    f"(~{need_n} total) to reach 80% power.")
    return Finding(
        pillar=PILLAR,
        title=("Sample size is sufficient" if adequate
               else "Sample size may be too small"),
        status=Status.PASS if adequate else Status.WARN,
        why=(
            "An underpowered evaluation can miss a real difference entirely. If "
            "power is low, 'not significant' might just mean 'not enough data', "
            "not 'no difference'."
        ),
        how_detected=(
            f"With {n} examples and the observed {magnitude} effect, the paired "
            f"test had {power:.0%} power to detect it (80% is the usual target)."
        ),
        how_to_fix=("The sample was large enough to detect this effect."
                    if adequate else fix_more),
        details={"check": "power", "achieved_power": power,
                 "required_n": need_n, "n": n},
    )
