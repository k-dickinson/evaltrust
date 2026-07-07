"""Power analysis for a paired (one-sample-on-differences) t-test.

This is what turns "the difference wasn't significant" into an actionable
recommendation. Either the effect is genuinely absent, or the sample was too
small to see it — power analysis distinguishes the two and tells the user how
many more examples would be needed.

Uses the exact noncentral-t formulation (matching statsmodels), not a normal
approximation.
"""

from __future__ import annotations

import math

from scipy import stats as _sp


def achieved_power(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """Power of a two-sided paired t-test to detect ``effect_size`` at ``n`` pairs.

    Power is the probability of correctly rejecting the null when an effect of
    this size truly exists. ``effect_size`` is Cohen's d (sign-agnostic).
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
