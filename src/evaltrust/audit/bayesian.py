"""Interpret a Bayesian decisive-pair win probability as an advisory finding."""

from __future__ import annotations

import math
from numbers import Real

from ..core.schema import EvalData, Finding, Status
from ..stats.bayesian import bayesian_win_probability

PILLAR = "Statistical Validity"


def audit_bayesian_win_probability(
    data: EvalData,
    model_a: str,
    model_b: str,
    confidence: float = 0.95,
) -> list[Finding]:
    """Report the posterior decisive-example win rate for the first model."""
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ValueError("confidence must be between 0 and 1")
    confidence = float(confidence)
    if not math.isfinite(confidence) or confidence <= 0 or confidence >= 1:
        raise ValueError("confidence must be between 0 and 1")

    differences = data.differences(model_a, model_b)
    wins_a = int((differences < 0).sum())
    wins_b = int((differences > 0).sum())
    ties = int((differences == 0).sum())
    n_decisive = wins_a + wins_b
    common = {
        "check": "bayesian_win_probability",
        "confidence": confidence,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "n_decisive": n_decisive,
        "prior": "jeffreys",
        "prior_alpha": 0.5,
        "prior_beta": 0.5,
        "estimand": "decisive_pair_win_rate_a",
    }

    why = (
        "The frequentist p-value and confidence interval do not answer how likely "
        f"it is that {model_a} wins more often than {model_b} on decisive examples."
    )
    if n_decisive == 0:
        detected = (
            "The results contain no paired scores for the selected models."
            if differences.size == 0
            else f"All {ties} paired score differences were ties."
        )
        return [Finding(
            pillar=PILLAR,
            title="Bayesian win probability not assessed",
            status=Status.SKIP,
            why=why,
            how_detected=(
                f"{detected} With no decisive examples the posterior equals the "
                "prior, so no win probability or interval is reported."
            ),
            how_to_fix=(
                "Add paired per-model scores with at least one non-tied example, then "
                "read the probability and interval as an advisory view, not a ship gate."
            ),
            details={**common, "assessed": False},
        )]

    probability, low, high = bayesian_win_probability(
        wins_a, wins_b, confidence=confidence
    )
    confidence_label = f"{confidence:.0%}"
    return [Finding(
        pillar=PILLAR,
        title=(
            f"Bayesian: P({model_a} wins more often than {model_b} on decisive "
            f"examples) = {probability:.2%} ({confidence_label} CrI for "
            f"{model_a} win rate {low:.2%}-{high:.2%})"
        ),
        status=Status.PASS,
        why=why,
        how_detected=(
            f"A closed-form Jeffreys-Beta posterior used {wins_a} {model_a} wins "
            f"and {wins_b} {model_b} wins after excluding {ties} ties. The "
            f"{confidence_label} credible interval describes the latent "
            f"decisive-example win rate for {model_a}, not the posterior probability."
        ),
        how_to_fix=(
            "Read the probability and interval as an advisory view, not a ship gate "
            "on their own."
        ),
        details={
            **common,
            "probability_a_better": probability,
            "ci_low": low,
            "ci_high": high,
            "assessed": True,
        },
    )]
