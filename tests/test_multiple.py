"""Tests for multiple-comparison corrections."""

import pytest
from statsmodels.stats.multitest import multipletests

from evaltrust.stats.multiple import holm_bonferroni


def test_holm_bonferroni_matches_statsmodels():
    p_values = [0.001, 0.021, 0.029, 0.20]
    expected_reject, expected_adjusted, _, _ = multipletests(
        p_values, alpha=0.05, method="holm")

    got = holm_bonferroni(p_values, alpha=0.05)

    assert got.reject == [bool(x) for x in expected_reject]
    assert got.adjusted_pvalues == pytest.approx(expected_adjusted)


def test_holm_bonferroni_reports_thresholds_in_original_order():
    got = holm_bonferroni([0.03, 0.001, 0.20], alpha=0.05)
    assert got.thresholds == pytest.approx([0.025, 0.05 / 3, 0.05])


def test_holm_bonferroni_uses_the_audit_strict_alpha_boundary():
    got = holm_bonferroni([0.05, 0.001], alpha=0.05)
    assert got.reject == [False, True]


def test_holm_bonferroni_step_down_stops_at_the_first_failure():
    """A p-value below its own threshold is still retained once an earlier one fails.

    0.031 misses alpha/2, so 0.032 is retained even though it clears its own
    threshold of alpha/1. Callers must not re-derive rejections from thresholds.
    """
    p_values = [0.031, 0.032]
    got = holm_bonferroni(p_values, alpha=0.05)
    assert got.thresholds == pytest.approx([0.025, 0.05])
    assert got.reject == [False, False]

    naive = [p < t for p, t in zip(p_values, got.thresholds)]
    assert naive == [False, True]  # the second would reject on its threshold alone


def test_holm_bonferroni_breaks_ties_by_input_order():
    """Tied p-values must not have their thresholds assigned by sort luck.

    Thresholds are reported per metric, so a tie has to resolve deterministically:
    the earlier metric takes the stricter threshold. An unstable sort permutes
    ties once the array is long enough to leave numpy's insertion-sort cutoff,
    which silently reshuffles which metric is shown which alpha.
    """
    p_values = [0.2, 0.04, 0.04, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01,
                0.2, 0.04, 0.2, 0.04, 0.04, 0.2, 0.04, 0.04]
    got = holm_bonferroni(p_values, alpha=0.05)

    for value in set(p_values):
        tied = [i for i, p in enumerate(p_values) if p == value]
        thresholds = [got.thresholds[i] for i in tied]
        assert thresholds == sorted(thresholds), (
            f"thresholds for tied p={value} are not in input order")

    # ties share an adjusted p-value regardless of order
    assert got.adjusted_pvalues[1] == pytest.approx(got.adjusted_pvalues[2])


def test_holm_bonferroni_is_deterministic_under_ties():
    first = holm_bonferroni([0.02, 0.02, 0.02], alpha=0.05)
    assert first.reject == [False, False, False]
    assert first.adjusted_pvalues == pytest.approx([0.06, 0.06, 0.06])


def test_holm_bonferroni_matches_statsmodels_under_ties():
    p_values = [0.01, 0.01, 0.04, 0.04]
    expected_reject, expected_adjusted, _, _ = multipletests(
        p_values, alpha=0.05, method="holm")
    got = holm_bonferroni(p_values, alpha=0.05)
    assert got.reject == [bool(x) for x in expected_reject]
    assert got.adjusted_pvalues == pytest.approx(expected_adjusted)


def test_holm_bonferroni_rejects_bad_p_values():
    with pytest.raises(ValueError):
        holm_bonferroni([0.1, 1.2])
    with pytest.raises(ValueError):
        holm_bonferroni([0.1, -0.01])
    with pytest.raises(ValueError):
        holm_bonferroni([])


def test_holm_bonferroni_rejects_non_finite_p_values():
    with pytest.raises(ValueError, match="finite"):
        holm_bonferroni([float("nan"), 0.01])
    with pytest.raises(ValueError, match="finite"):
        holm_bonferroni([float("inf"), 0.01])


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, -0.1])
def test_holm_bonferroni_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        holm_bonferroni([0.01, 0.2], alpha=alpha)
