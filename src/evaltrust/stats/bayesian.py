"""Closed-form Bayesian statistics for paired win counts."""

from __future__ import annotations

import numpy as np
from scipy import stats as _sp

_PRIOR_ALPHA = _PRIOR_BETA = 0.5


def bayesian_win_probability(
    wins: int,
    losses: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return P(win rate > 0.5) and an equal-tail interval for the win rate.

    Uses a Jeffreys Beta(0.5, 0.5) prior and an analytic Beta posterior.
    The caller excludes ties before passing the decisive win and loss counts.
    """
    for name, count in (("wins", wins), ("losses", losses)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(
            count, (int, np.integer)
        ):
            raise ValueError(f"{name} must be a non-negative integer")
        if count < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if wins + losses == 0:
        raise ValueError("at least one decisive win or loss is required")
    if (
        isinstance(confidence, (bool, np.bool_))
        or not isinstance(confidence, (int, float, np.integer, np.floating))
        or not np.isfinite(confidence)
        or confidence <= 0
        or confidence >= 1
    ):
        raise ValueError("confidence must be between 0 and 1")

    posterior_alpha = wins + _PRIOR_ALPHA
    posterior_beta = losses + _PRIOR_BETA
    probability = float(_sp.beta.sf(0.5, posterior_alpha, posterior_beta))
    tail = (1.0 - confidence) / 2.0
    low, high = _sp.beta.ppf(
        [tail, 1.0 - tail], posterior_alpha, posterior_beta
    )
    return probability, float(low), float(high)
