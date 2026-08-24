"""Unbiased pass@k estimator and a paired A-vs-B comparison (issue #164).

Code benchmarks (HumanEval, MBPP, SWE-bench) sample ``n`` completions per task
and report ``pass@k`` with the unbiased estimator ``1 - C(n-c, k) / C(n, k)``
rather than a plain pass rate. This module computes that estimator per task in a
numerically stable product form, then compares two models' mean pass@k over the
shared tasks with a seeded percentile bootstrap that resamples whole tasks, so
the interval reflects task sampling on top of the estimator's own completion
sampling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .resampling import bootstrap_statistic_ci

_METHOD = "passk-paired-percentile-bootstrap-v1"


@dataclass(frozen=True)
class PassAtKResult:
    """Plain-scalar result for the paired pass@k estimand (B - A convention)."""

    k: int
    pass_at_k_a: float
    pass_at_k_b: float
    delta: float
    delta_low: float
    delta_high: float
    a_low: float
    a_high: float
    b_low: float
    b_high: float
    confidence: float
    n_tasks: int
    method: str


def _pass_at_k_single(n: int, c: int, k: int) -> float:
    """Unbiased ``pass@k`` for a single task: ``1 - C(n-c, k) / C(n, k)``.

    Uses the numerically stable product form from the HumanEval paper
    (``1 - prod_{i=n-c+1}^{n} (1 - k / i)``) so there are no factorials to
    overflow at large ``n``. When ``n - c < k`` every set of ``k`` draws must
    contain a correct sample, so the value is exactly ``1.0``.
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _mean(sample: np.ndarray) -> np.ndarray:
    """Vectorized mean over the last axis (the ``bootstrap_statistic_ci`` contract)."""
    return sample.mean(axis=-1)


def paired_pass_at_k(
    per_task: Sequence[tuple[int, int, int, int]],
    k: int,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PassAtKResult:
    """Compare two models' mean ``pass@k`` over paired tasks.

    ``per_task`` is one ``(n_a, c_a, n_b, c_b)`` tuple per task that both models
    sampled at least ``k`` times (``n samples``, ``c correct``). The point
    estimates are the mean of the per-task unbiased ``pass@k``; ``delta`` follows
    EvalTrust's ``B - A`` convention. The three intervals are seeded percentile
    bootstraps over tasks, all drawn with the same ``seed`` and task count, so the
    per-model and difference intervals come from identical resamples.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("pass@k requires an integer k >= 1")
    tasks = list(per_task)
    if not tasks:
        raise ValueError("paired_pass_at_k requires at least one task")
    if (
        isinstance(confidence, (bool, np.bool_))
        or not np.isscalar(confidence)
        or not np.isfinite(confidence)
        or confidence <= 0
        or confidence >= 1
    ):
        raise ValueError("confidence must be between 0 and 1")
    if (
        isinstance(n_resamples, (bool, np.bool_))
        or not isinstance(n_resamples, (int, np.integer))
        or n_resamples < 1
    ):
        raise ValueError("n_resamples must be a positive integer")

    passk_a = np.empty(len(tasks), dtype=float)
    passk_b = np.empty(len(tasks), dtype=float)
    for i, task in enumerate(tasks):
        n_a, c_a, n_b, c_b = (int(x) for x in task)
        for n, c in ((n_a, c_a), (n_b, c_b)):
            if n < 1 or c < 0 or c > n:
                raise ValueError(
                    f"each task needs 0 <= c <= n with n >= 1, got n={n}, c={c}"
                )
            if n < k:
                raise ValueError(
                    f"each task must have n >= k for both models, got n={n}, k={k}"
                )
        passk_a[i] = _pass_at_k_single(n_a, c_a, k)
        passk_b[i] = _pass_at_k_single(n_b, c_b, k)

    confidence = float(confidence)
    n_resamples = int(n_resamples)
    a_low, a_high = bootstrap_statistic_ci(
        passk_a, _mean, confidence=confidence, n_resamples=n_resamples, seed=seed)
    b_low, b_high = bootstrap_statistic_ci(
        passk_b, _mean, confidence=confidence, n_resamples=n_resamples, seed=seed)
    delta_low, delta_high = bootstrap_statistic_ci(
        passk_b - passk_a, _mean, confidence=confidence,
        n_resamples=n_resamples, seed=seed)

    return PassAtKResult(
        k=int(k),
        pass_at_k_a=float(passk_a.mean()),
        pass_at_k_b=float(passk_b.mean()),
        delta=float(passk_b.mean() - passk_a.mean()),
        delta_low=delta_low,
        delta_high=delta_high,
        a_low=a_low,
        a_high=a_high,
        b_low=b_low,
        b_high=b_high,
        confidence=confidence,
        n_tasks=len(tasks),
        method=_METHOD,
    )
