"""Advisory rank-stability check for multi-model mean-score leaderboards."""

from __future__ import annotations

import numpy as np

from ..config import AuditConfig
from ..core.schema import EvalData, Finding, Status
from ..stats.rank_stability import bootstrap_rank_distribution
from .allpairs import _ranked_models

PILLAR = "All-pairs Comparison"


def audit_rank_stability(data: EvalData, cfg: AuditConfig) -> list[Finding]:
    """Bootstrap the mean-score ranking and report which positions hold.

    Gated on the same ``all_pairs`` path as the pair family. Emits one advisory
    finding: PASS when assessed, SKIP otherwise. Never WARN or FAIL.
    """
    models = list(dict.fromkeys(data.models))
    n_models = len(models)
    score_counts = {
        model: int(sum(1 for ex in data.examples if model in ex.scores))
        for model in models
    }
    preference_only = data.has_preferences and not any(
        example.scores for example in data.examples
    )

    common = {
        "check": "rank_stability",
        "alpha": float(cfg.alpha),
        "threshold": float(1.0 - cfg.alpha),
        "seed": int(cfg.seed),
        "n_resamples": int(cfg.n_resamples),
        "n_models": int(n_models),
        "n_examples": int(data.n_examples),
        "score_counts": score_counts,
    }

    why = (
        "A point ordering of models by mean score can be noise. Positions that "
        "swap under resampling are not supported distinctions."
    )

    if preference_only:
        return [_skip(
            title="Rank stability: not assessed for preference-only data",
            detected=(
                "Preference-only data has no per-model scores for a mean-score "
                "ranking."
            ),
            fix="Add paired per-model scores to assess rank stability.",
            reason="preference_only",
            common=common,
            why=why,
        )]

    if data.n_examples == 0:
        return [_skip(
            title="Rank stability: no examples to resample",
            detected="The file has no examples.",
            fix="Provide scored examples before assessing rank stability.",
            reason="no_examples",
            common=common,
            why=why,
        )]

    if n_models < 3:
        return [_skip(
            title="Rank stability: needs at least three scored models",
            detected=(
                f"This file declares {n_models} model(s). Rank stability is "
                "reported only for three or more models; two-model files keep "
                "the paired comparison."
            ),
            fix="Add a third scored model, or use the paired comparison and "
                "optional Bayesian view for two models.",
            reason="fewer_than_three_models",
            common=common,
            why=why,
        )]

    if any(count == 0 for count in score_counts.values()):
        zeroed = [m for m, c in score_counts.items() if c == 0]
        return [_skip(
            title="Rank stability: a declared model has no scores",
            detected=(
                "At least one declared model has zero scores "
                f"({', '.join(zeroed)}), so a joint ranking is undefined."
            ),
            fix="Score every declared model on at least one example, or drop "
                "unscored models from the file.",
            reason="zero_score_model",
            common=common,
            why=why,
        )]

    # Build score matrix in declared model order (columns).
    score_rows = np.full((data.n_examples, n_models), np.nan, dtype=float)
    for i, ex in enumerate(data.examples):
        for j, model in enumerate(models):
            if model in ex.scores:
                score_rows[i, j] = float(ex.scores[model])

    clustered = data.has_clusters
    cluster_ids = None
    if clustered:
        # Match schema: missing group_id -> singleton; sorted unique later.
        labels = []
        for ex in data.examples:
            if ex.group_id is not None:
                labels.append(("g", ex.group_id))
            else:
                labels.append(("i", ex.id))
        # Encode as ints for the primitive.
        encode: dict[tuple, int] = {}
        cluster_ids = np.empty(data.n_examples, dtype=int)
        for i, key in enumerate(labels):
            if key not in encode:
                encode[key] = len(encode)
            cluster_ids[i] = encode[key]
        n_units = len(encode)
    else:
        n_units = data.n_examples

    if n_units < 2:
        return [_skip(
            title="Rank stability: fewer than two resampling units",
            detected=(
                f"Only {n_units} resampling unit(s) "
                f"({'cluster' if clustered else 'example'}). Bootstrap "
                "rank occupancy needs at least two."
            ),
            fix=(
                "Add more independent clusters (group_id) or more examples."
                if clustered else
                "Add more scored examples."
            ),
            reason="fewer_than_two_units",
            common={**common, "n_resampling_units": int(n_units),
                    "clustered": bool(clustered)},
            why=why,
        )]

    # Observed order: same ranking as all-pairs (mean, lexical model id).
    observed_models = _ranked_models(data)
    # Map to column indices in `models`.
    model_index = {model: i for i, model in enumerate(models)}
    observed_order = np.array(
        [model_index[m] for m in observed_models], dtype=int
    )

    result = bootstrap_rank_distribution(
        score_rows,
        n_resamples=cfg.n_resamples,
        seed=cfg.seed,
        cluster_ids=cluster_ids,
        observed_order=observed_order,
    )

    threshold = 1.0 - cfg.alpha
    stable_mask = result.position_retention >= threshold
    stable_positions = [
        int(i + 1) for i, ok in enumerate(stable_mask) if ok
    ]
    n_stable = len(stable_positions)
    full_pct = 100.0 * result.full_order_retention

    # Occupancy rows follow observed_order; columns are ranks 0..m-1 (best first).
    occ_rows = [
        [float(result.occupancy[model_index[m], r]) for r in range(n_models)]
        for m in observed_models
    ]
    position_retention = [
        float(result.position_retention[s]) for s in range(n_models)
    ]

    unit_word = "clusters" if result.clustered else "examples"
    independence = (
        f"Whole group_id clusters were resampled with replacement ({n_units} "
        f"units); rows within a selected cluster were pooled."
        if result.clustered else
        f"Examples were resampled with replacement ({n_units} units) and "
        "treated as independent. Supply a group_id per example to enable "
        "cluster-aware resampling."
    )

    return [Finding(
        pillar=PILLAR,
        title=(
            f"Rank stability: {n_stable} of {n_models} positions stable, "
            f"full order retained {full_pct:.1f}%"
        ),
        status=Status.PASS,
        why=why,
        how_detected=(
            f"Joint bootstrap of the mean-score ranking over {cfg.n_resamples} "
            f"resamples of {unit_word} (seed {cfg.seed}). {independence} "
            "Exact mean ties split rank-occupancy credit evenly, so model-name "
            "order is never counted as stability. A position is stable when "
            f"its observed model holds it in at least {threshold:g} of "
            "resamples (1 - alpha)."
        ),
        how_to_fix=(
            "Treat unstable adjacent positions as tied. Gather more independent "
            "examples or clusters before asserting the order."
        ),
        details={
            **common,
            "assessed": True,
            "criterion": (
                "position stable when observed model holds it in at least "
                "1 - alpha of bootstrap resamples; exact ties split occupancy"
            ),
            "n_resampling_units": int(result.n_resampling_units),
            "clustered": bool(result.clustered),
            "observed_order": list(observed_models),
            "rank_occupancy": {
                model: occ_rows[i] for i, model in enumerate(observed_models)
            },
            "position_retention": position_retention,
            "stable_positions": stable_positions,
            "top1_retention": float(result.top1_retention),
            "full_order_retention": float(result.full_order_retention),
            "tie_resamples": int(result.tie_resamples),
        },
    )]


def _skip(
    *,
    title: str,
    detected: str,
    fix: str,
    reason: str,
    common: dict,
    why: str,
) -> Finding:
    return Finding(
        pillar=PILLAR,
        title=title,
        status=Status.SKIP,
        why=why,
        how_detected=detected,
        how_to_fix=fix,
        details={
            **common,
            "assessed": False,
            "reason": reason,
        },
    )
