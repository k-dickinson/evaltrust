"""Tests for inter-judge agreement metrics.

Judge Reliability asks: if a different evaluator looked, would it reach the same
verdict? These metrics quantify how much the judges actually agree beyond chance.
Validated against statsmodels' inter_rater reference implementations.
"""

import numpy as np
import pytest
from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa

from evallab.stats.agreement import (
    cohen_kappa,
    fleiss_kappa as our_fleiss,
    percent_agreement,
)


# ---------------------------------------------------------------------------
# percent_agreement: mean fraction of items on which a pair of judges agree
# ---------------------------------------------------------------------------

def test_percent_agreement_is_one_when_all_judges_identical():
    ratings = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    assert percent_agreement(ratings) == pytest.approx(1.0)


def test_percent_agreement_two_judges_half_disagree():
    ratings = np.array([[1, 1], [1, 0], [0, 0], [0, 1]])
    assert percent_agreement(ratings) == pytest.approx(0.5)


def test_percent_agreement_averages_over_all_pairs():
    # cols A=[1,1,0,0] B=[1,1,0,0] C=[1,0,0,1]
    # AB=1.0, AC=0.5, BC=0.5 -> mean 2/3
    ratings = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1]])
    assert percent_agreement(ratings) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# cohen_kappa: two raters, chance-corrected
# ---------------------------------------------------------------------------

def test_cohen_kappa_perfect_agreement_is_one():
    a = np.array([1, 0, 1, 0, 1])
    assert cohen_kappa(a, a) == pytest.approx(1.0)


def test_cohen_kappa_all_same_single_category_is_one():
    a = np.array([1, 1, 1, 1])
    b = np.array([1, 1, 1, 1])
    assert cohen_kappa(a, b) == pytest.approx(1.0)


def test_cohen_kappa_known_value():
    # Confusion matrix [[20,5],[10,15]]: po=0.7, pe=0.5, kappa=0.4
    a = np.array([0] * 25 + [1] * 25)
    b = np.array([0] * 20 + [1] * 5 + [0] * 10 + [1] * 15)
    assert cohen_kappa(a, b) == pytest.approx(0.4, abs=1e-9)


def test_cohen_kappa_matches_statsmodels():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, size=200)
    b = a.copy()
    flip = rng.random(200) < 0.3
    b[flip] = rng.integers(0, 3, size=flip.sum())

    cats = [0, 1, 2]
    table = np.array([[np.sum((a == i) & (b == j)) for j in cats] for i in cats])
    ref = cohens_kappa(table).kappa
    assert cohen_kappa(a, b) == pytest.approx(ref, abs=1e-9)


# ---------------------------------------------------------------------------
# fleiss_kappa: many raters per item
# ---------------------------------------------------------------------------

def test_fleiss_kappa_perfect_agreement_is_one():
    # every item unanimously category 0 or 1
    table = np.array([[5, 0], [0, 5], [5, 0], [0, 5]])
    assert our_fleiss(table) == pytest.approx(1.0)


def test_fleiss_kappa_matches_statsmodels():
    rng = np.random.default_rng(1)
    n_items, n_raters, n_cats = 40, 5, 3
    table = np.zeros((n_items, n_cats), dtype=int)
    for i in range(n_items):
        votes = rng.integers(0, n_cats, size=n_raters)
        for v in votes:
            table[i, v] += 1
    assert our_fleiss(table) == pytest.approx(fleiss_kappa(table), abs=1e-9)
