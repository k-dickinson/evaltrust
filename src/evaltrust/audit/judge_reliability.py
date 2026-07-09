"""Judge Reliability audit.

Would a different evaluator reach the same verdict? Checks consensus (do judges
agree on the winner) and item-level agreement beyond chance. Requires multi-judge
data; absent that, we SKIP.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..core.schema import EvalData, Finding, Status
from ..stats.agreement import fleiss_kappa, percent_agreement

PILLAR = "Judge Reliability"


def _skip(reason: str) -> Finding:
    return Finding(
        pillar=PILLAR,
        title="Not assessed",
        status=Status.SKIP,
        why=(
            "A single judge's verdict could be its own bias. Without a second "
            "opinion we can't tell whether the ranking would survive a different "
            "evaluator."
        ),
        how_detected=reason,
        how_to_fix=(
            "Score the outputs with a second judge (another model or a human)."
        ),
        details={"check": "judge_reliability", "assessed": False},
    )


def _judge_names(data: EvalData) -> list[str]:
    names: list[str] = []
    for ex in data.examples:
        if ex.judges:
            for j in ex.judges:
                if j not in names:
                    names.append(j)
    return names


def audit_judge_reliability(
    data: EvalData, model_a: str, model_b: str, agreement_threshold: float = 0.8
) -> list[Finding]:
    if not data.has_judges:
        return [_skip("The results file contains no per-judge scores.")]
    judges = _judge_names(data)
    if len(judges) < 2:
        return [_skip("Only one judge is present in the results file.")]

    consensus = _consensus(data, judges, model_a, model_b)
    ratings = _rating_matrix(data, judges, model_a, model_b)
    if ratings is None:
        return [consensus,
                _skip("Judges did not score a common set of items.")]
    return [consensus, _agreement(ratings, judges, agreement_threshold)]


def _consensus(data, judges, model_a, model_b) -> Finding:
    winners = {}
    for j in judges:
        a_vals = [ex.judges[j][model_a] for ex in data.examples
                  if ex.judges and j in ex.judges and model_a in ex.judges[j]]
        b_vals = [ex.judges[j][model_b] for ex in data.examples
                  if ex.judges and j in ex.judges and model_b in ex.judges[j]]
        winners[j] = model_b if np.mean(b_vals) >= np.mean(a_vals) else model_a

    unique = set(winners.values())
    unanimous = len(unique) == 1
    winner = next(iter(unique)) if unanimous else None

    return Finding(
        pillar=PILLAR,
        title=("Judges agree on the winner" if unanimous
               else "Judges disagree on the winner"),
        status=Status.PASS if unanimous else Status.FAIL,
        why=(
            "If different judges crown different winners, the ranking says more "
            "about the choice of judge than about the models."
        ),
        how_detected=(
            f"Each judge's preferred model: "
            + ", ".join(f"{j}->{winners[j]}" for j in judges) + "."
        ),
        how_to_fix=(
            f"Every judge preferred {winner}; the verdict is judge-independent."
            if unanimous else
            "Don't report a single winner. Find why the judges differ and "
            "reconcile them first."
        ),
        details={"check": "judge_consensus", "per_judge_winner": winners,
                 "unanimous": unanimous},
    )


def _rating_matrix(data, judges, model_a, model_b) -> np.ndarray | None:
    """Rows = (example, model) items every judge scored; columns = judges."""
    rows = []
    for ex in data.examples:
        if not ex.judges:
            continue
        for m in (model_a, model_b):
            if all(j in ex.judges and m in ex.judges[j] for j in judges):
                rows.append([ex.judges[j][m] for j in judges])
    if not rows:
        return None
    return np.array(rows, dtype=float)


def _agreement(ratings: np.ndarray, judges: list[str],
               agreement_threshold: float = 0.8) -> Finding:
    agree = percent_agreement(ratings)

    # Outlier = judge with the lowest mean pairwise agreement with the others.
    n = len(judges)
    mean_pair = []
    for k in range(n):
        others = [float(np.mean(ratings[:, k] == ratings[:, j]))
                  for j in range(n) if j != k]
        mean_pair.append(float(np.mean(others)))
    outlier = judges[int(np.argmin(mean_pair))]

    kappa = _maybe_fleiss(ratings)
    good = agree >= agreement_threshold
    kappa_str = "n/a (non-categorical scores)" if kappa is None else f"{kappa:.3f}"

    return Finding(
        pillar=PILLAR,
        title=("Judges agree at the item level" if good
               else "Judges frequently disagree"),
        status=Status.PASS if good else Status.WARN,
        why=(
            "Low item-level agreement means the judges are applying different "
            "standards, so any single judge's scores carry hidden noise."
        ),
        how_detected=(
            f"Mean pairwise agreement across {n} judges was {agree:.0%} "
            f"(Fleiss' kappa {kappa_str}); {outlier} agreed least with the rest."
        ),
        how_to_fix=(
            "The judges are consistent with each other."
            if good else
            f"Tighten the rubric and review {outlier}; it's the odd one out."
        ),
        details={"check": "inter_judge_agreement", "percent_agreement": agree,
                 "fleiss_kappa": kappa, "outlier_judge": outlier},
    )


def _maybe_fleiss(ratings: np.ndarray) -> float | None:
    """Fleiss' kappa if the scores are categorical (few integer-valued levels)."""
    values = np.unique(ratings)
    if values.size > 10 or not np.allclose(values, np.round(values)):
        return None
    cats = {v: i for i, v in enumerate(values)}
    table = np.zeros((ratings.shape[0], len(cats)), dtype=int)
    for r, row in enumerate(ratings):
        for v in row:
            table[r, cats[v]] += 1
    return fleiss_kappa(table)
