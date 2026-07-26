"""Tests for the fixed-example predictive rerun primitive."""

from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest
from scipy import stats as sp

from evaltrust.stats import predictive_rerun as pr


def test_public_api_is_model_specific_and_has_no_generic_confidence_surface():
    assert hasattr(pr, "NormalTheoryPredictiveRerunResult")
    assert hasattr(pr, "predictive_rerun_normal_theory")
    assert not hasattr(pr, "PredictiveRerunResult")
    assert not hasattr(pr, "predictive_rerun_gap")

    signature = inspect.signature(pr.predictive_rerun_normal_theory)
    assert "central_mass" in signature.parameters
    assert signature.parameters["central_mass"].default == pytest.approx(0.95)
    assert "confidence" not in signature.parameters

    field_names = {
        field.name for field in fields(pr.NormalTheoryPredictiveRerunResult)
    }
    assert field_names == {
        "point_estimate_b_minus_a",
        "normal_theory_probability_a_better",
        "normal_theory_range_low",
        "normal_theory_range_high",
        "central_mass",
        "prediction_variance",
        "degrees_of_freedom",
        "n_examples",
        "future_runs",
        "method",
        "degenerate_reason",
    }
    assert not any(name.startswith("interval_") for name in field_names)
    assert not any(name.startswith("predictive_probability_") for name in field_names)
    assert "confidence" not in field_names


def test_hand_calculated_welch_satterthwaite_fixture():
    runs_a = [[1.0, 3.0], [2.0, 4.0, 6.0]]
    runs_b = [[2.0, 6.0, 4.0, 8.0], [1.0, 5.0]]
    future_runs = 3

    result = pr.predictive_rerun_normal_theory(
        runs_a, runs_b, future_runs=future_runs
    )

    # Means are A=(2, 4), B=(5, 3), so equal example weighting gives 1.
    assert result.point_estimate_b_minus_a == pytest.approx(1.0)

    # Unbiased variances are A=(2, 4), B=(20/3, 8).
    # Contributions before the 1/m^2 aggregate factor are listed explicitly.
    contributions = [
        2.0 * (1.0 / 2.0 + 1.0 / 3.0),
        (20.0 / 3.0) * (1.0 / 4.0 + 1.0 / 3.0),
        4.0 * (1.0 / 3.0 + 1.0 / 3.0),
        8.0 * (1.0 / 2.0 + 1.0 / 3.0),
    ]
    assert contributions == pytest.approx(
        [5.0 / 3.0, 35.0 / 9.0, 8.0 / 3.0, 20.0 / 3.0]
    )
    expected_variance = sum(contributions) / 4.0
    assert expected_variance == pytest.approx(67.0 / 18.0)
    assert result.prediction_variance == pytest.approx(expected_variance)

    q = [value / 4.0 for value in contributions]
    denominator = (
        q[0] ** 2 / (2 - 1)
        + q[1] ** 2 / (4 - 1)
        + q[2] ** 2 / (3 - 1)
        + q[3] ** 2 / (2 - 1)
    )
    expected_df = expected_variance**2 / denominator
    assert result.degrees_of_freedom == pytest.approx(expected_df)

    critical = float(sp.t.ppf(0.975, expected_df))
    half_width = critical * expected_variance**0.5
    assert result.normal_theory_range_low == pytest.approx(1.0 - half_width)
    assert result.normal_theory_range_high == pytest.approx(1.0 + half_width)
    assert result.normal_theory_probability_a_better == pytest.approx(
        float(sp.t.cdf(-1.0 / expected_variance**0.5, expected_df))
    )
    assert result.central_mass == pytest.approx(0.95)
    assert result.method == "welch-satterthwaite-normal-theory-v1"


def test_same_data_counterexample_matches_strict_t_cdf_event():
    scale = 3.0**0.5
    centered = [scale, -scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    shifted = [value + 1.0 for value in centered]

    result = pr.predictive_rerun_normal_theory(
        [centered, centered],
        [shifted, shifted],
        future_runs=3,
    )

    assert result.point_estimate_b_minus_a == pytest.approx(1.0)
    assert result.prediction_variance == pytest.approx(1.0 / 3.0)
    assert result.degrees_of_freedom == pytest.approx(32.0)
    assert result.normal_theory_probability_a_better == pytest.approx(
        float(sp.t.cdf(-(3.0**0.5), 32.0))
    )


def test_reduces_to_one_stream_future_mean_prediction_formula():
    runs_a = [[1.0, 2.0, 4.0, 5.0]]
    runs_b = [[7.0, 7.0, 7.0, 7.0]]
    future_runs = 5

    result = pr.predictive_rerun_normal_theory(
        runs_a, runs_b, future_runs=future_runs
    )

    observed_mean_a = 3.0
    unbiased_variance_a = 10.0 / 3.0
    expected_point = 7.0 - observed_mean_a
    expected_variance = unbiased_variance_a * (1.0 / 4.0 + 1.0 / 5.0)
    expected_df = 3.0
    half_width = float(sp.t.ppf(0.975, expected_df)) * expected_variance**0.5

    assert result.point_estimate_b_minus_a == pytest.approx(expected_point)
    assert result.prediction_variance == pytest.approx(expected_variance)
    assert result.degrees_of_freedom == pytest.approx(expected_df)
    assert result.normal_theory_range_low == pytest.approx(expected_point - half_width)
    assert result.normal_theory_range_high == pytest.approx(expected_point + half_width)
    assert result.normal_theory_probability_a_better is not None
    assert result.degenerate_reason is None


def test_range_and_probability_match_independent_scipy_t_oracles():
    result = pr.predictive_rerun_normal_theory(
        [[0.2, 0.5, 0.8], [0.0, 0.4]],
        [[0.1, 0.3, 0.9, 1.1], [0.3, 0.7, 0.8]],
        future_runs=7,
        central_mass=0.925,
    )

    assert result.prediction_variance is not None
    assert result.degrees_of_freedom is not None
    standard_error = result.prediction_variance**0.5
    alpha = 1.0 - 0.925
    critical = float(
        sp.t.ppf(1.0 - alpha / 2.0, result.degrees_of_freedom)
    )
    assert result.normal_theory_range_low == pytest.approx(
        result.point_estimate_b_minus_a - critical * standard_error
    )
    assert result.normal_theory_range_high == pytest.approx(
        result.point_estimate_b_minus_a + critical * standard_error
    )
    assert result.normal_theory_probability_a_better == pytest.approx(
        float(
            sp.t.cdf(
                -result.point_estimate_b_minus_a / standard_error,
                result.degrees_of_freedom,
            )
        )
    )


def test_negative_b_minus_a_gap_means_a_probability_above_half():
    result = pr.predictive_rerun_normal_theory(
        [[0.8, 1.0, 1.2]],
        [[0.1, 0.4, 0.7]],
        future_runs=3,
    )
    assert result.point_estimate_b_minus_a < 0.0
    assert result.normal_theory_probability_a_better is not None
    assert result.normal_theory_probability_a_better > 0.5


def test_prediction_variance_contains_observed_center_and_future_noise():
    runs_a = [[1.0, 3.0, 5.0, 7.0]]
    runs_b = [[2.0, 2.0, 2.0, 2.0]]
    future_runs = 8

    result = pr.predictive_rerun_normal_theory(
        runs_a, runs_b, future_runs=future_runs
    )

    sample_variance = 20.0 / 3.0
    observed_center = sample_variance / 4.0
    future_noise = sample_variance / future_runs
    assert result.prediction_variance == pytest.approx(
        observed_center + future_noise
    )
    assert result.prediction_variance != pytest.approx(observed_center)
    assert result.prediction_variance != pytest.approx(future_noise)


def test_example_weights_stay_equal_under_strongly_unequal_counts():
    short_a = [-1.0, 1.0]
    short_b = [-1.0, 1.0]
    long_a = [-1.0, 1.0] * 50
    long_b = [9.0, 11.0] * 50

    result = pr.predictive_rerun_normal_theory(
        [short_a, long_a],
        [short_b, long_b],
        future_runs=4,
    )

    pooled_gap = (
        (sum(short_b) + sum(long_b)) / 102.0
        - (sum(short_a) + sum(long_a)) / 102.0
    )
    assert result.point_estimate_b_minus_a == pytest.approx(5.0)
    assert result.point_estimate_b_minus_a != pytest.approx(pooled_gap)


def test_unequal_a_and_b_counts_are_accepted_without_truncation():
    result = pr.predictive_rerun_normal_theory(
        [[0.0, 2.0], [3.0, 4.0, 5.0, 6.0, 7.0]],
        [[1.0, 3.0, 5.0, 7.0], [2.0, 8.0, 5.0]],
        future_runs=6,
    )
    expected = ((4.0 - 1.0) + (5.0 - 5.0)) / 2.0
    assert result.point_estimate_b_minus_a == pytest.approx(expected)
    assert result.prediction_variance is not None


def test_value_order_within_each_stream_has_no_meaning():
    runs_a = [[0.0, 1.0, 4.0], [2.0, 5.0, 9.0, 11.0]]
    runs_b = [[3.0, 2.0, 8.0, 1.0], [7.0, 4.0, 6.0]]
    expected = pr.predictive_rerun_normal_theory(runs_a, runs_b, future_runs=5)
    permuted = pr.predictive_rerun_normal_theory(
        [list(reversed(stream)) for stream in runs_a],
        [list(reversed(stream)) for stream in runs_b],
        future_runs=5,
    )
    assert permuted == expected


def test_joint_example_permutation_changes_nothing():
    runs_a = [[0.0, 1.0], [2.0, 5.0, 7.0], [9.0, 11.0, 13.0, 15.0]]
    runs_b = [[1.0, 4.0, 6.0], [3.0, 8.0], [8.0, 12.0, 14.0]]
    expected = pr.predictive_rerun_normal_theory(runs_a, runs_b, future_runs=4)
    order = [2, 0, 1]
    permuted = pr.predictive_rerun_normal_theory(
        [runs_a[index] for index in order],
        [runs_b[index] for index in order],
        future_runs=4,
    )
    assert permuted == expected


def test_all_singleton_input_returns_structural_degradation():
    result = pr.predictive_rerun_normal_theory(
        [[1.0], [4.0]],
        [[3.0], [2.0]],
        future_runs=5,
        central_mass=0.9,
    )
    assert result.point_estimate_b_minus_a == pytest.approx(0.0)
    assert result.normal_theory_range_low == pytest.approx(0.0)
    assert result.normal_theory_range_high == pytest.approx(0.0)
    assert result.normal_theory_probability_a_better is None
    assert result.prediction_variance is None
    assert result.degrees_of_freedom is None
    assert result.central_mass == pytest.approx(0.9)
    assert result.n_examples == 2
    assert result.future_runs == 5
    assert result.method == "welch-satterthwaite-normal-theory-v1"
    assert result.degenerate_reason == "single_run"


@pytest.mark.parametrize(
    "runs_a, runs_b",
    [
        ([[1.0], [2.0, 3.0]], [[4.0], [5.0, 6.0]]),
        ([[1.0, 2.0]], [[3.0]]),
        ([[1.0]], [[2.0, 3.0]]),
    ],
)
def test_mixed_singleton_and_identified_streams_direct_caller_to_skip(runs_a, runs_b):
    with pytest.raises(ValueError, match="SKIP policy"):
        pr.predictive_rerun_normal_theory(runs_a, runs_b, future_runs=3)


def test_observed_zero_variance_returns_point_fallback_without_probability():
    result = pr.predictive_rerun_normal_theory(
        [[2.0, 2.0], [5.0, 5.0, 5.0]],
        [[1.0, 1.0, 1.0], [7.0, 7.0]],
        future_runs=4,
    )
    assert result.point_estimate_b_minus_a == pytest.approx(0.5)
    assert result.normal_theory_range_low == pytest.approx(0.5)
    assert result.normal_theory_range_high == pytest.approx(0.5)
    assert result.prediction_variance is None
    assert result.degrees_of_freedom is None
    assert result.normal_theory_probability_a_better is None
    assert result.method == "welch-satterthwaite-normal-theory-v1"
    assert result.degenerate_reason == "observed_zero_variance"


def test_observed_constancy_never_fabricates_a_probability():
    tied = pr.predictive_rerun_normal_theory(
        [[1.0, 1.0], [4.0, 4.0]],
        [[2.0, 2.0], [3.0, 3.0]],
        future_runs=2,
    )
    favors_a = pr.predictive_rerun_normal_theory(
        [[2.0, 2.0]],
        [[1.0, 1.0]],
        future_runs=2,
    )
    singleton_tie = pr.predictive_rerun_normal_theory(
        [[1.0]],
        [[1.0]],
        future_runs=2,
    )

    assert tied.point_estimate_b_minus_a == 0.0
    assert tied.normal_theory_probability_a_better is None
    assert tied.prediction_variance is None
    assert favors_a.point_estimate_b_minus_a == -1.0
    assert favors_a.normal_theory_probability_a_better is None
    assert favors_a.prediction_variance is None
    assert singleton_tie.normal_theory_probability_a_better is None


def test_nonconstant_streams_cannot_be_labeled_observed_zero_variance():
    with pytest.raises(ValueError, match="nonconstant streams"):
        pr.predictive_rerun_normal_theory(
            [[0.0, 1e-200]],
            [[0.0, 0.0]],
            future_runs=2,
        )


@pytest.mark.parametrize(
    "runs_a, runs_b, message",
    [
        ([], [], "at least one example"),
        ([[1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]], "same number"),
        ([[]], [[1.0, 2.0]], "nonempty"),
        ([[1.0, 2.0]], [[]], "nonempty"),
        ([np.array([[1.0, 2.0], [3.0, 4.0]])], [[1.0, 2.0]], "one-dimensional"),
        ([[1.0, 2.0]], [np.array([[1.0], [2.0]])], "one-dimensional"),
        ([[1.0, np.nan]], [[1.0, 2.0]], "finite"),
        ([[1.0, 2.0]], [[1.0, np.inf]], "finite"),
        ([[True, 1.0]], [[1.0, 2.0]], "boolean"),
        ([[1.0, 2.0]], [np.array([False, True])], "boolean"),
        ([[1.0, object()]], [[1.0, 2.0]], "numeric"),
    ],
)
def test_rejects_invalid_run_arrays(runs_a, runs_b, message):
    with pytest.raises(ValueError, match=message):
        pr.predictive_rerun_normal_theory(runs_a, runs_b, future_runs=3)


@pytest.mark.parametrize(
    "future_runs",
    [0, -1, 1.5, np.float64(2.0), True, np.bool_(False), "3"],
)
def test_rejects_invalid_future_run_count(future_runs):
    with pytest.raises(ValueError, match="future_runs"):
        pr.predictive_rerun_normal_theory(
            [[1.0, 2.0]], [[2.0, 3.0]], future_runs=future_runs
        )


@pytest.mark.parametrize(
    "central_mass",
    [0.0, 1.0, -0.1, 1.1, np.nan, np.inf, True, np.bool_(False), "0.95"],
)
def test_rejects_invalid_central_mass(central_mass):
    with pytest.raises(ValueError, match="central_mass"):
        pr.predictive_rerun_normal_theory(
            [[1.0, 2.0]],
            [[2.0, 3.0]],
            future_runs=3,
            central_mass=central_mass,
        )


def test_result_is_frozen_and_uses_plain_python_scalars():
    result = pr.predictive_rerun_normal_theory(
        [np.array([0.0, 1.0, 2.0])],
        [np.array([1.0, 3.0, 4.0, 8.0])],
        future_runs=np.int64(3),
        central_mass=np.float32(0.9),
    )

    float_fields = (
        result.point_estimate_b_minus_a,
        result.normal_theory_range_low,
        result.normal_theory_range_high,
        result.normal_theory_probability_a_better,
        result.prediction_variance,
        result.degrees_of_freedom,
        result.central_mass,
    )
    assert all(type(value) is float for value in float_fields)
    assert type(result.n_examples) is int
    assert type(result.future_runs) is int
    assert type(result.method) is str
    assert result.method == "welch-satterthwaite-normal-theory-v1"
    assert result.degenerate_reason is None
    with pytest.raises(FrozenInstanceError):
        result.future_runs = 4


def test_does_not_read_or_change_global_numpy_rng_state():
    np.random.seed(130)
    before = np.random.get_state()
    first = pr.predictive_rerun_normal_theory(
        [[0.0, 1.0, 3.0]], [[1.0, 2.0, 5.0]], future_runs=6
    )
    after = np.random.get_state()
    second = pr.predictive_rerun_normal_theory(
        [[0.0, 1.0, 3.0]], [[1.0, 2.0, 5.0]], future_runs=6
    )

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert first == second


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_source_has_no_rng_or_resample_sized_allocation_path():
    signature = inspect.signature(pr.predictive_rerun_normal_theory)
    assert "seed" not in signature.parameters
    assert "n_resamples" not in signature.parameters

    tree = ast.parse(inspect.getsource(pr))
    called = {
        _dotted_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not any(name.startswith("np.random") for name in called)
    assert not any("resample" in name.lower() for name in called)
    assert called.isdisjoint(
        {"np.empty", "np.zeros", "np.ones", "np.full", "np.tile"}
    )


def test_module_docs_define_model_probability_and_range_limits():
    docs = inspect.getdoc(pr)
    assert docs is not None
    normalized = " ".join(docs.lower().split())
    assert "working normal-theory" in normalized
    assert "future b - a < 0" in normalized
    assert "not a posterior probability" in normalized
    assert "not a calibrated" in normalized
