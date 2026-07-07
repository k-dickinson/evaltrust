"""Repeatability audit.

The question: if you ran this evaluation again tomorrow, would you reach the same
conclusion? Stochastic judges and sampling make eval scores wobble; if the winner
changes from one run to the next, the ranking is noise dressed up as a result.

Requires repeated-run data in the file. When it is absent we don't guess — we
emit a SKIP that tells the user how to generate the evidence.
"""

from __future__ import annotations

import numpy as np

from ..core.schema import EvalData, Finding, Status

PILLAR = "Repeatability"


def _skip(reason: str) -> Finding:
    return Finding(
        pillar=PILLAR,
        title="Repeatability not assessed",
        status=Status.SKIP,
        why=(
            "A conclusion you can't reproduce isn't a conclusion. Without "
            "repeated runs we cannot tell whether the ranking is stable or a "
            "lucky draw."
        ),
        how_detected=reason,
        how_to_fix=(
            "Evaluate each model 3-5 times (varying the judge/sampling seed) and "
            "include the per-run scores so EvalLab can measure rerun stability."
        ),
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
            f"{leader} wins consistently across reruns — the ranking is reliable."
            if status is Status.PASS else
            "Do not rely on this ranking. Average more runs, reduce judge "
            "temperature, or fix seeds until the winner stops changing."
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
            "Average more reruns per model to shrink the measurement noise "
            "before trusting the size of the gap."
        ),
        details={"check": "measurement_variance", "gap_std": gap_std,
                 "mean_gap": overall, "snr": snr},
    )
