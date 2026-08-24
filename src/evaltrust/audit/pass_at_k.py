"""Interpret the paired pass@k comparison as an advisory finding (issue #164).

Reads per-task sampled outcomes from ``Example.runs`` (each run value is one
sampled completion; a pass is a run scoring exactly ``1.0``) and reports each
model's ``pass@k`` with a bootstrap interval plus the paired difference. Only
tasks where both models have at least ``k`` runs are used; the rest are skipped
and counted so the reader sees exactly what the estimate is built on.
"""

from __future__ import annotations

from ..core.schema import EvalData, Finding, Status
from ..stats.pass_at_k import paired_pass_at_k

PILLAR = "Statistical Validity"
CHECK = "pass_at_k"
METHOD = "passk-paired-percentile-bootstrap-v1"
CONFIDENCE = 0.95


def _counts(runs: list[float]) -> tuple[int, int, int]:
    """Return (n, c, n_nonbinary): samples, passes (==1.0), and off-{0,1} values."""
    n = len(runs)
    c = sum(1 for v in runs if v == 1.0)
    nonbinary = sum(1 for v in runs if v not in (0.0, 1.0))
    return n, c, nonbinary


def _details(**kw) -> dict:
    base = {"check": CHECK, "method": METHOD, "confidence": CONFIDENCE}
    base.update(kw)
    return base


def _skip(model_a: str, model_b: str, k: int, *, detected: str, reason: str,
          n_skipped: int) -> Finding:
    return Finding(
        pillar=PILLAR,
        title=f"pass@{k}: {model_a} vs {model_b} not assessed",
        status=Status.SKIP,
        why=(
            "pass@k needs at least k sampled completions per task for both "
            "models. Without them the k-sampling structure a code eval reports "
            "cannot be estimated."
        ),
        how_detected=detected,
        how_to_fix=(
            "Provide per-task repeated runs (0/1 per completion) for both models "
            f"with at least k={k} samples each, or lower --k."
        ),
        details=_details(assessed=False, k=int(k), model_a=model_a,
                         model_b=model_b, reason=reason, n_tasks=0,
                         n_skipped=int(n_skipped), n_nonbinary=0),
    )


def audit_pass_at_k(
    data: EvalData,
    model_a: str,
    model_b: str,
    k: int,
    *,
    confidence: float = CONFIDENCE,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> list[Finding]:
    """Report one advisory paired pass@k finding for two models."""
    per_task: list[tuple[int, int, int, int]] = []
    n_skipped = 0
    n_nonbinary = 0
    for ex in data.examples:
        runs = ex.runs or {}
        runs_a = runs.get(model_a)
        runs_b = runs.get(model_b)
        if not runs_a or not runs_b:
            continue
        n_a, c_a, nb_a = _counts(list(runs_a))
        n_b, c_b, nb_b = _counts(list(runs_b))
        if n_a < k or n_b < k:
            n_skipped += 1
            continue
        n_nonbinary += nb_a + nb_b
        per_task.append((n_a, c_a, n_b, c_b))

    if not per_task:
        detected = (
            "No example carries repeated runs for both models, so pass@k has no "
            "sampled completions to read."
            if n_skipped == 0 else
            f"Every eligible task had fewer than k={k} runs for at least one "
            f"model; {n_skipped} task(s) were skipped."
        )
        reason = "no_runs" if n_skipped == 0 else "all_below_k"
        return [_skip(model_a, model_b, k, detected=detected, reason=reason,
                      n_skipped=n_skipped)]

    result = paired_pass_at_k(
        per_task, k, confidence=confidence, n_resamples=n_resamples, seed=seed)

    decisive = result.delta_low > 0.0 or result.delta_high < 0.0
    status = Status.PASS if decisive else Status.WARN
    leader = model_b if result.delta > 0 else model_a
    direction = (
        f"{leader} leads by {abs(result.delta):.1%} "
        f"[{result.delta_low:.1%}, {result.delta_high:.1%}]"
        if decisive else
        f"the {result.delta:+.1%} gap's interval "
        f"[{result.delta_low:.1%}, {result.delta_high:.1%}] includes 0"
    )
    nonbinary_note = (
        f" {n_nonbinary} run value(s) outside {{0, 1}} were counted as failures."
        if n_nonbinary else ""
    )

    return [Finding(
        pillar=PILLAR,
        title=(
            f"pass@{k}: {model_a} {result.pass_at_k_a:.1%} vs "
            f"{model_b} {result.pass_at_k_b:.1%} "
            f"(Δ {result.delta:+.1%} [{result.delta_low:.1%}, "
            f"{result.delta_high:.1%}], {confidence:.0%} bootstrap)"
        ),
        status=status,
        why=(
            "A flat pass/fail rate throws away the completion-sampling variance "
            "that pass@k is built on. This uses the unbiased estimator and a "
            "paired bootstrap over tasks so the gap carries an honest interval."
        ),
        how_detected=(
            f"Estimated unbiased pass@{k} on {result.n_tasks} task(s) with at "
            f"least {k} runs for both models ({n_skipped} skipped for n < k); "
            f"{direction}. Interval resampled {result.n_tasks} tasks with "
            f"replacement (seed {seed})." + nonbinary_note
        ),
        how_to_fix=(
            "Read this as an advisory ranking at this k. To sharpen it, add more "
            "sampled completions per task or compare at the k your benchmark "
            "reports (e.g. pass@1, pass@10)."
        ),
        details=_details(
            assessed=True,
            k=result.k,
            model_a=model_a,
            model_b=model_b,
            pass_at_k_a=result.pass_at_k_a,
            pass_at_k_b=result.pass_at_k_b,
            delta=result.delta,
            delta_low=result.delta_low,
            delta_high=result.delta_high,
            a_low=result.a_low,
            a_high=result.a_high,
            b_low=result.b_low,
            b_high=result.b_high,
            n_tasks=result.n_tasks,
            n_skipped=int(n_skipped),
            n_nonbinary=int(n_nonbinary),
            seed=int(seed),
        ),
    )]
