"""Inter-rater agreement metrics for judge reliability.

Percent agreement is intuitive but doesn't correct for chance agreement; kappa
does. EvalTrust reports both.
"""

from __future__ import annotations

import itertools

import numpy as np


def percent_agreement(ratings: np.ndarray) -> float:
    """Mean fraction of items on which a pair of judges agree, over all pairs.

    ``ratings`` has shape (n_items, n_judges). For two judges this is simply the
    fraction of items they score identically.
    """
    ratings = np.asarray(ratings)
    n_judges = ratings.shape[1]
    if n_judges < 2:
        raise ValueError("percent_agreement requires at least two judges")

    pair_scores = [
        float(np.mean(ratings[:, i] == ratings[:, j]))
        for i, j in itertools.combinations(range(n_judges), 2)
    ]
    return float(np.mean(pair_scores))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two raters' label vectors.

    kappa = (p_observed - p_expected) / (1 - p_expected). When both raters use a
    single category for everything they agree perfectly, so kappa is 1.0.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("cohen_kappa requires equal-length rating vectors")

    n = a.size
    categories = np.unique(np.concatenate([a, b]))
    po = float(np.mean(a == b))

    pe = 0.0
    for c in categories:
        pe += (np.mean(a == c)) * (np.mean(b == c))
    pe = float(pe)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' kappa for many raters per item.

    ``table`` has shape (n_items, n_categories); each cell is the number of
    raters who assigned that category to that item. Every row must sum to the
    same number of raters.
    """
    table = np.asarray(table, dtype=float)
    n_items, n_cats = table.shape
    n_raters = table.sum(axis=1)
    if not np.allclose(n_raters, n_raters[0]):
        raise ValueError("fleiss_kappa requires the same rater count per item")
    m = float(n_raters[0])

    # Per-item agreement.
    p_item = (np.sum(table**2, axis=1) - m) / (m * (m - 1))
    p_bar = float(np.mean(p_item))

    # Per-category proportion of all assignments.
    p_cat = table.sum(axis=0) / (n_items * m)
    p_bar_e = float(np.sum(p_cat**2))

    if p_bar_e == 1.0:
        return 1.0
    return (p_bar - p_bar_e) / (1.0 - p_bar_e)
