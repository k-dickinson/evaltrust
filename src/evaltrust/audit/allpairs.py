"""Optional comparison of every model pair in one corrected family."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..config import AuditConfig
from ..core.schema import EvalData, Finding, Status
from ..stats.multiplicity import holm_bonferroni
from ..stats.paired import mcnemar_exact
from ..stats.resampling import permutation_test
from .statistical import _discordant_counts, _is_binary

PILLAR = "All-pairs Comparison"
_METHODS = ("bonferroni", "holm", "none")


def _mean_score(data: EvalData, model: str) -> float:
    values = [
        example.scores[model]
        for example in data.examples
        if model in example.scores
    ]
    return float(np.mean(values)) if values else float("-inf")


def _ranked_models(data: EvalData) -> list[str]:
    """Rank declared models by mean score, using model id to break ties."""
    unique = list(dict.fromkeys(data.models))
    return sorted(unique, key=lambda model: (-_mean_score(data, model), model))


def _pair_pvalue(
    data: EvalData,
    model_a: str,
    model_b: str,
    n_resamples: int,
    seed: int,
) -> float:
    """Return the p-value from the same test used by the primary comparison."""
    differences = data.differences(model_a, model_b)
    mean_difference = float(differences.mean())

    if _is_binary(data, model_a, model_b):
        if mean_difference >= 0:
            leader, trailer = model_b, model_a
        else:
            leader, trailer = model_a, model_b
        leader_only, trailer_only = _discordant_counts(data, leader, trailer)
        return float(mcnemar_exact(leader_only, trailer_only))

    oriented = differences if mean_difference >= 0 else -differences
    return float(permutation_test(
        oriented, n_resamples=n_resamples, seed=seed))


def _correct_pvalues(
    pvalues: list[float], alpha: float, method: str
) -> tuple[list[bool], list[float]]:
    """Correct one tested pair family, mirroring the metric-suite rules."""
    if method not in _METHODS:
        raise ValueError(
            f"correction must be one of {_METHODS}, got {method!r}")

    raw = [float(p) for p in pvalues]
    k = len(raw)
    if k == 0:
        return [], []
    if k == 1 or method == "none":
        return [p < alpha for p in raw], raw
    if method == "bonferroni":
        corrected_alpha = alpha / k
        return (
            [p < corrected_alpha for p in raw],
            [min(k * p, 1.0) for p in raw],
        )
    return holm_bonferroni(raw, alpha)


def audit_all_pairs(data: EvalData, cfg: AuditConfig) -> list[Finding]:
    """Test every declared model pair and correct across tests performed.

    Models are ranked by mean score, with lexical tie-breaking. Pairs with no
    shared scores stay in the report but do not enter the corrected family.
    Each tested pair uses ``seed + i`` in tested-pair order so permutation
    Monte Carlo error is not shared across the family.
    """
    models = _ranked_models(data)
    candidate_pairs = list(combinations(models, 2))
    n_pairs_total = len(candidate_pairs)
    method = cfg.correction

    # Validate even when no pair is testable, matching the suite's early check.
    _correct_pvalues([], cfg.alpha, method)

    preference_only = data.has_preferences and not any(
        example.scores for example in data.examples)
    if preference_only:
        pairs = [
            {
                "model_a": model_a,
                "model_b": model_b,
                "n": 0,
                "assessed": False,
                "reason": "preference_only",
            }
            for model_a, model_b in candidate_pairs
        ]
        return [_skip(
            title="All-pairs: not assessed for preference-only data",
            detected=(
                "Preference-only data has no paired model scores for an "
                "all-pairs score comparison."
            ),
            fix="Add paired per-model scores to compare every model pair.",
            reason="preference_only",
            method=method,
            alpha=cfg.alpha,
            pairs=pairs,
        )]

    candidates = []
    for model_a, model_b in candidate_pairs:
        n_paired = int(data.differences(model_a, model_b).size)
        candidates.append((model_a, model_b, n_paired))
    tested = [candidate for candidate in candidates if candidate[2] > 0]
    k = len(tested)

    if k == 0:
        pairs = [
            {
                "model_a": model_a,
                "model_b": model_b,
                "n": n_paired,
                "assessed": False,
                "reason": "no_paired_scores",
            }
            for model_a, model_b, n_paired in candidates
        ]
        return [_skip(
            title="All-pairs: no model pairs had paired scores",
            detected=(
                f"None of the {n_pairs_total} declared model pairs had scores "
                "on the same examples."
            ),
            fix="Evaluate at least two models on the same examples.",
            reason=("fewer_than_two_models" if n_pairs_total == 0
                    else "no_paired_scores"),
            method=method,
            alpha=cfg.alpha,
            pairs=pairs,
        )]

    raw_p = [
        _pair_pvalue(
            data,
            model_a,
            model_b,
            n_resamples=cfg.n_resamples,
            seed=cfg.seed + i,
        )
        for i, (model_a, model_b, _n_paired) in enumerate(tested)
    ]
    rejected, adjusted = _correct_pvalues(raw_p, cfg.alpha, method)
    corrected = {
        (model_a, model_b): (raw_p[i], adjusted[i], rejected[i])
        for i, (model_a, model_b, _n_paired) in enumerate(tested)
    }

    pairs = []
    for model_a, model_b, n_paired in candidates:
        result = corrected.get((model_a, model_b))
        if result is None:
            pairs.append({
                "model_a": model_a,
                "model_b": model_b,
                "n": n_paired,
                "assessed": False,
                "reason": "no_paired_scores",
            })
            continue
        p_value, adjusted_p, reject = result
        pairs.append({
            "model_a": model_a,
            "model_b": model_b,
            "n": n_paired,
            "assessed": True,
            "p_value": float(p_value),
            "adjusted_p": float(adjusted_p),
            "reject": bool(reject),
        })

    n_separable = int(sum(rejected))
    return [Finding(
        pillar=PILLAR,
        title=(f"All-pairs: {n_separable} of {k} pairs separable after "
               f"{method} (alpha {cfg.alpha:g})"),
        status=Status.PASS,
        why=(
            "The top-two comparison hides whether the rest of the field is "
            "distinguishable. Testing every pair without one family-wide "
            "correction also inflates false positives."
        ),
        how_detected=(
            f"Tested {k} of {n_pairs_total} declared model pairs on paired "
            f"scores and applied {method} across the tested family at alpha "
            f"{cfg.alpha:g}. A pair is separable only when significant after "
            "correction. An underpowered pair is listed but not separable."
        ),
        how_to_fix=(
            "Use the separable pairs as supported distinctions. Treat each "
            "inseparable pair as an ordering this data does not support."
        ),
        details={
            "check": "all_pairs",
            "assessed": True,
            "method": method,
            "alpha": float(cfg.alpha),
            "k": int(k),
            "n_pairs_total": int(n_pairs_total),
            "n_separable": n_separable,
            "pairs": pairs,
        },
    )]


def _skip(
    *,
    title: str,
    detected: str,
    fix: str,
    reason: str,
    method: str,
    alpha: float,
    pairs: list[dict],
) -> Finding:
    return Finding(
        pillar=PILLAR,
        title=title,
        status=Status.SKIP,
        why=(
            "The top-two comparison can hide whether other models are "
            "distinguishable, but every pair needs shared scores to test it."
        ),
        how_detected=detected,
        how_to_fix=fix,
        details={
            "check": "all_pairs",
            "assessed": False,
            "reason": reason,
            "method": method,
            "alpha": float(alpha),
            "k": 0,
            "n_pairs_total": len(pairs),
            "n_separable": 0,
            "pairs": pairs,
        },
    )
