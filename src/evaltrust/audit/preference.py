"""Audit pairwise judgments where each judge names a winner or a tie."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..core.schema import EvalData, Finding, Preference, Status
from ..stats.paired import mcnemar_exact
from ..stats.resampling import bootstrap_ci

PILLAR = "Pairwise Preference"


def _skip(reason: str, details: dict | None = None) -> Finding:
    payload = {"check": "preference", "assessed": False}
    payload.update(details or {})
    return Finding(
        pillar=PILLAR,
        title="Not assessed",
        status=Status.SKIP,
        why=(
            "Pairwise preferences only support a winner when judges record which "
            "model they preferred on the same examples."
        ),
        how_detected=reason,
        how_to_fix=(
            "Add a preference or winner value for each judged example, using the "
            "model id or Preference.TIE."
        ),
        details=payload,
    )


def _tallies(data: EvalData, model_a: str, model_b: str):
    wins_a = wins_b = ties = 0
    decisive: list[float] = []
    judged_examples = 0
    per_judge: "OrderedDict[str, dict[str, int]]" = OrderedDict()

    for example in data.examples:
        vote_a = vote_b = vote_tie = 0
        for judge, preference in (example.preferences or {}).items():
            if preference is Preference.TIE:
                bucket = "ties"
                vote_tie += 1
            elif preference == model_a:
                bucket = "wins_a"
                vote_a += 1
            elif preference == model_b:
                bucket = "wins_b"
                vote_b += 1
            else:
                continue
            counts = per_judge.setdefault(
                judge, {"wins_a": 0, "wins_b": 0, "ties": 0})
            counts[bucket] += 1

        total = vote_a + vote_b + vote_tie
        if total == 0:
            continue
        judged_examples += 1
        if vote_a > total / 2:
            wins_a += 1
            decisive.append(1.0)
        elif vote_b > total / 2:
            wins_b += 1
            decisive.append(0.0)
        else:
            ties += 1

    return wins_a, wins_b, ties, decisive, judged_examples, dict(per_judge)


def audit_preferences(
    data: EvalData,
    model_a: str,
    model_b: str,
    alpha: float = 0.05,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
    *,
    significant: bool | None = None,
) -> list[Finding]:
    """Run an exact sign test and a seeded CI on example-level majority outcomes."""
    if not data.has_preferences:
        return [_skip("The results contain no pairwise preference judgments.")]

    wins_a, wins_b, ties, decisive, judged, per_judge = _tallies(
        data, model_a, model_b)
    common = {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "n_decisive": wins_a + wins_b,
        "n_judged_examples": judged,
        "per_judge": per_judge,
        "aggregation": "strict_majority_per_example",
    }
    if judged == 0:
        return [_skip(
            f"No preference judgment names '{model_a}', '{model_b}', or a tie.",
            common,
        )]

    p_value = mcnemar_exact(wins_b, wins_a)
    if not decisive:
        return [_skip(
            f"All {ties} judged examples were ties, leaving zero decisive pairs.",
            {**common, "p_value": p_value},
        )]

    if significant is None:
        significant = p_value < alpha
    leader = model_a if wins_a >= wins_b else model_b
    trailer = model_b if leader == model_a else model_a
    significance = Finding(
        pillar=PILLAR,
        title=(
            f"{leader} is preferred significantly more often than {trailer}"
            if significant else
            f"Preference between {model_a} and {model_b} is inconclusive"
        ),
        status=Status.PASS if significant else Status.FAIL,
        why=(
            "A raw preference count can be sampling noise. The exact sign test "
            "checks whether decisive example-level outcomes split beyond chance."
        ),
        how_detected=(
            f"The exact two-sided sign test over {wins_a + wins_b} decisive examples "
            f"gave p = {p_value:.4f} against alpha {alpha}; {ties} examples tied."
        ),
        how_to_fix=(
            "The preference difference is strong enough to act on."
            if significant else
            "Do not call a winner yet. Collect more independently judged examples."
        ),
        details={
            "check": "preference_significance",
            **common,
            "p_value": p_value,
            "alpha": alpha,
            "significant": significant,
            "outcome": "significant" if significant else "inconclusive",
            "test": "exact sign test",
        },
    )

    outcomes = np.asarray(decisive, dtype=float)
    ci_low, ci_high = bootstrap_ci(
        outcomes,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    win_rate_a = wins_a / (wins_a + wins_b)
    effect = Finding(
        pillar=PILLAR,
        title=f"{model_a} won {win_rate_a:.1%} of decisive preferences",
        status=Status.PASS,
        why=(
            "The p-value says whether the split is distinguishable from chance. "
            "The win rate and interval show its direction and practical size."
        ),
        how_detected=(
            f"{model_a} won {wins_a} of {wins_a + wins_b} decisive examples "
            f"({win_rate_a:.1%}); the {confidence:.0%} bootstrap interval was "
            f"[{ci_low:.1%}, {ci_high:.1%}]."
        ),
        how_to_fix="Report the interval with the win rate so its uncertainty stays visible.",
        details={
            "check": "preference_effect",
            **common,
            "win_rate_a": win_rate_a,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence": confidence,
            "seed": seed,
        },
    )
    return [significance, effect]
