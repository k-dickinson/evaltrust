"""Power analysis for a paired (one-sample-on-differences) t-test.

Distinguishes "no effect" from "too small a sample to see it". Uses the exact
noncentral-t formulation (matching statsmodels).
"""

from __future__ import annotations

import math

from scipy import stats as _sp


def achieved_power(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """Power of a two-sided paired t-test to detect ``effect_size`` at ``n`` pairs.

    ``effect_size`` is Cohen's d (sign-agnostic).
    """
    if n < 2:
        return 0.0

    d = abs(effect_size)
    df = n - 1
    ncp = d * math.sqrt(n)  # noncentrality parameter
    crit = _sp.t.ppf(1.0 - alpha / 2.0, df)

    # P(reject) = P(T > crit) + P(T < -crit) under the noncentral alternative.
    upper = _sp.nct.sf(crit, df, ncp)
    lower = _sp.nct.cdf(-crit, df, ncp)
    power = float(upper + lower)
    return min(1.0, max(0.0, power))


def required_n(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    max_n: int = 10_000_000,
) -> int:
    """Smallest number of paired examples to reach ``power`` for this effect.

    Returns ``max_n`` as a sentinel if the target power is unreachable within
    the cap (e.g. a zero effect can never be detected above the alpha rate).
    """
    d = abs(effect_size)
    if d == 0.0:
        return max_n
    if math.isinf(d):
        return 2

    # Monotone in n, so binary-search the smallest sufficient sample.
    lo, hi = 2, 2
    while achieved_power(d, hi, alpha) < power:
        hi *= 2
        if hi > max_n:
            return max_n
    while lo < hi:
        mid = (lo + hi) // 2
        if achieved_power(d, mid, alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def minimum_detectable_effect(
    n: int, power: float = 0.8, alpha: float = 0.05, tol: float = 1e-4
) -> float:
    """Smallest true effect size (Cohen's d) detectable at ``power`` given ``n``.

    Prospective (a property of the sample size), avoiding post-hoc power's
    circularity. Bisects on d.
    """
    if n < 2:
        return float("inf")

    lo, hi = 0.0, 0.5
    while achieved_power(hi, n, alpha) < power:
        hi *= 2.0
        if hi > 1e6:
            return float("inf")
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if achieved_power(mid, n, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi
