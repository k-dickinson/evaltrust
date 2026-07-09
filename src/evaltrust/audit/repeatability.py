"""Repeatability audit.

If you ran this evaluation again tomorrow, would you reach the same conclusion?
If the winner changes from one run to the next, the ranking is noise dressed up
as a result. Requires repeated-run data; when it's absent we emit a SKIP that
tells the user how to generate it.
"""

from __future__ import annotations

import numpy as np

from ..core.schema import EvalData, Finding, Status

PILLAR = "Repeatability"


def _skip(reason: str) -> Finding:
    return Finding(
        pillar=PILLAR,
        title="Not assessed",
        status=Status.SKIP,
        why=(
            "A conclusion you can't reproduce isn't a conclusion. Without "
            "repeated runs we cannot tell whether the ranking is stable or a "
            "lucky draw."
        ),
        how_detected=reason,
        how_to_fix="Run each model 3-5 times and include the per-run scores.",
        details={"check": "repeatability", "assessed": False},
    )


def _run_gaps(data: EvalData, model_a: str, model_b: str) -> np.ndarray | None:
    """Per-run mean gap (B - A), or None if there aren't >=2 aligned runs."""
    a_runs, b_runs = [], []
    for ex in data.examples:
        if not ex.runs or model_a not in ex.runs or model_b not in ex.runs:
            continue
        a_runs.append(ex.runs[model_a])
        b_runs.append(ex.runs[model_b])
    if not a_runs:
        return None

    r = min(min(len(x) for x in a_runs), min(len(x) for x in b_runs))
    if r < 2:
        return None

    a_mat = np.array([x[:r] for x in a_runs], dtype=float)  # examples x runs
    b_mat = np.array([x[:r] for x in b_runs], dtype=float)
    return b_mat.mean(axis=0) - a_mat.mean(axis=0)          # length r


def audit_repeatability(
    data: EvalData, model_a: str, model_b: str
) -> list[Finding]:
    if not data.has_runs:
        return [_skip("The results file contains no repeated runs.")]

    gaps = _run_gaps(data, model_a, model_b)
    if gaps is None:
        return [_skip("Fewer than two repeated runs are available per model.")]

    r = gaps.size
    overall = float(gaps.mean())
    overall_sign = np.sign(overall)
    flips = int(np.count_nonzero(np.sign(gaps) != overall_sign))
    stability = 1.0 - flips / r
    gap_std = float(gaps.std(ddof=1)) if r > 1 else 0.0

    return [
        _stability(flips, r, stability, overall, model_a, model_b),
        _variance(gap_std, overall, model_a, model_b),
    ]


def _stability(flips, r, stability, overall, model_a, model_b) -> Finding:
    leader = model_b if overall >= 0 else model_a
    if flips == 0:
        status = Status.PASS
    elif flips < r / 2:
        status = Status.WARN
    else:
        status = Status.FAIL

    return Finding(
        pillar=PILLAR,
        title=("Ranking is stable across reruns" if status is Status.PASS
               else "Ranking changes across reruns"),
        status=status,
        why=(
            "If the winner flips from one run to the next, the reported ranking "
            "is driven by run-to-run randomness, not a real difference. Deciding "
            "on it is a coin toss."
        ),
        how_detected=(
            f"Across {r} reruns the winner was {leader} in {r - flips} of them "
            f"and reversed in {flips} (stability {stability:.0%})."
        ),
        how_to_fix=(
            f"{leader} wins consistently across reruns; the ranking is reliable."
            if status is Status.PASS else
            "Don't rely on this ranking. Average more runs or fix the seed until "
            "the winner stops changing."
        ),
        details={"check": "rerun_stability", "runs": r, "flips": flips,
                 "stability": stability, "mean_gap": overall},
    )


def _variance(gap_std, overall, model_a, model_b) -> Finding:
    signal = abs(overall)
    noisy = gap_std >= signal and gap_std > 0
    snr = (signal / gap_std) if gap_std > 0 else float("inf")

    return Finding(
        pillar=PILLAR,
        title=("Rerun measurement noise is high" if noisy
               else "Rerun measurement noise is acceptable"),
        status=Status.WARN if noisy else Status.PASS,
        why=(
            "Even without a full flip, a gap that swings as much as its own size "
            "between runs is barely distinguishable from noise and won't "
            "reproduce reliably."
        ),
        how_detected=(
            f"The run-to-run gap had mean {overall:+.4f} and standard deviation "
            f"{gap_std:.4f} (signal-to-noise {snr:.2f})."
        ),
        how_to_fix=(
            "The gap is stable relative to its run-to-run noise."
            if not noisy else
            "Average more runs to shrink the noise before trusting the gap size."
        ),
        details={"check": "measurement_variance", "gap_std": gap_std,
                 "mean_gap": overall, "snr": snr},
    )


# --- single-model repeatability -------------------------------------------
# With one model there's no gap to track, so we ask whether the model's own score
# holds still across reruns, via the std of the per-run mean score.

def _skip_single(reason: str) -> Finding:
    return Finding(
        pillar=PILLAR,
        title="Not assessed",
        status=Status.SKIP,
        why=(
            "A score you can't reproduce isn't one you can trust. Without "
            "repeated runs we cannot tell whether it is stable or a lucky draw."
        ),
        how_detected=reason,
        how_to_fix="Run the model 3-5 times and include the per-run scores.",
        details={"check": "single_repeatability", "assessed": False},
    )


def _run_means(data: EvalData, model: str) -> np.ndarray | None:
    """Per-run mean score across examples, or None if there aren't >=2 aligned runs."""
    runs = []
    for ex in data.examples:
        if not ex.runs or model not in ex.runs:
            continue
        runs.append(ex.runs[model])
    if not runs:
        return None

    r = min(len(x) for x in runs)
    if r < 2:
        return None

    mat = np.array([x[:r] for x in runs], dtype=float)  # examples x runs
    return mat.mean(axis=0)                               # length r: mean per run


def audit_single_repeatability(data: EvalData, model: str) -> list[Finding]:
    if not data.has_runs:
        return [_skip_single("The results file contains no repeated runs.")]

    run_means = _run_means(data, model)
    if run_means is None:
        return [_skip_single("Fewer than two repeated runs are available.")]

    r = run_means.size
    mean_score = float(run_means.mean())
    score_std = float(run_means.std(ddof=1))
    return [_score_stability(score_std, mean_score, r, model)]


def _score_stability(score_std, mean_score, r, model) -> Finding:
    signal = abs(mean_score)
    noisy = score_std >= signal and score_std > 0
    snr = (signal / score_std) if score_std > 0 else float("inf")

    return Finding(
        pillar=PILLAR,
        title=("Score swings across reruns" if noisy
               else "Score is stable across reruns"),
        status=Status.WARN if noisy else Status.PASS,
        why=(
            "A score that swings as much as its own size between runs is barely "
            "distinguishable from noise. You can't ship on a number that won't "
            "hold still."
        ),
        how_detected=(
            f"Across {r} reruns {model} averaged {mean_score:.4f} with a "
            f"run-to-run standard deviation of {score_std:.4f} "
            f"(signal-to-noise {snr:.2f})."
        ),
        how_to_fix=(
            "The score is stable relative to its run-to-run noise."
            if not noisy else
            "Average more runs to pin the score down before trusting it."
        ),
        details={"check": "single_repeatability", "score_std": score_std,
                 "mean_score": mean_score, "runs": r, "snr": snr},
    )
