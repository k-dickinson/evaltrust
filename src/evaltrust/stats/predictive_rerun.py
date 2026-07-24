"""Working normal-theory statistics for rerunning the same fixed examples.

The fitted probability estimates the strict event ``future B - A < 0`` under a
Welch-Satterthwaite normal-theory approximation. It is not a posterior
probability of latent model superiority, an exact fixed-law probability, or a
distribution-free result. The range contains ``central_mass`` of the fitted
approximation. It is not a calibrated coverage interval for arbitrary run
distributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy import stats as _sp


_METHOD: Literal["welch-satterthwaite-normal-theory-v1"] = (
    "welch-satterthwaite-normal-theory-v1"
)


@dataclass(frozen=True)
class NormalTheoryPredictiveRerunResult:
    """Model-specific prediction for one fixed-example rerun."""

    point_estimate_b_minus_a: float
    normal_theory_probability_a_better: float | None
    normal_theory_range_low: float
    normal_theory_range_high: float
    central_mass: float
    prediction_variance: float | None
    degrees_of_freedom: float | None
    n_examples: int
    future_runs: int
    method: Literal["welch-satterthwaite-normal-theory-v1"]
    degenerate_reason: Literal[
        "single_run",
        "observed_zero_variance",
    ] | None


def _validate_stream(
    stream: Sequence[float],
    *,
    model: str,
    example_index: int,
) -> np.ndarray:
    """Return one finite, nonempty, one-dimensional float stream."""
    try:
        if any(isinstance(value, (bool, np.bool_)) for value in stream):
            raise ValueError(
                f"{model} run stream {example_index} contains a boolean value"
            )
    except TypeError as exc:
        raise ValueError(
            f"{model} run stream {example_index} must be one-dimensional"
        ) from exc

    try:
        values = np.asarray(stream, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{model} run stream {example_index} must contain numeric values"
        ) from exc
    if values.ndim != 1:
        raise ValueError(
            f"{model} run stream {example_index} must be one-dimensional"
        )
    if values.size == 0:
        raise ValueError(
            f"{model} run stream {example_index} must be nonempty"
        )
    if not bool(np.isfinite(values).all()):
        raise ValueError(
            f"{model} run stream {example_index} must contain only finite values"
        )
    return values


def _point_result(
    point_estimate: float,
    *,
    central_mass: float,
    n_examples: int,
    future_runs: int,
    reason: Literal["single_run", "observed_zero_variance"],
) -> NormalTheoryPredictiveRerunResult:
    """Return a point fallback when predictive variance is unidentified."""
    return NormalTheoryPredictiveRerunResult(
        point_estimate_b_minus_a=float(point_estimate),
        normal_theory_probability_a_better=None,
        normal_theory_range_low=float(point_estimate),
        normal_theory_range_high=float(point_estimate),
        central_mass=float(central_mass),
        prediction_variance=None,
        degrees_of_freedom=None,
        n_examples=int(n_examples),
        future_runs=int(future_runs),
        method=_METHOD,
        degenerate_reason=reason,
    )


def predictive_rerun_normal_theory(
    runs_a: Sequence[Sequence[float]],
    runs_b: Sequence[Sequence[float]],
    *,
    future_runs: int,
    central_mass: float = 0.95,
) -> NormalTheoryPredictiveRerunResult:
    """Estimate ``B - A`` for rerunning fixed, equal-weight examples.

    The result uses separate independent A and B run streams and one explicit
    future run count. The probability and range come from a working
    Welch-Satterthwaite normal-theory approximation. They do not change or
    replace inference about the observed mean gap.
    """
    if isinstance(future_runs, (bool, np.bool_)) or not isinstance(
        future_runs, (int, np.integer)
    ):
        raise ValueError("future_runs must be an integer greater than zero")
    if future_runs <= 0:
        raise ValueError("future_runs must be an integer greater than zero")
    if (
        isinstance(central_mass, (bool, np.bool_))
        or not isinstance(central_mass, (int, float, np.integer, np.floating))
        or not np.isfinite(central_mass)
        or central_mass <= 0
        or central_mass >= 1
    ):
        raise ValueError("central_mass must be finite and between zero and one")

    try:
        n_a = len(runs_a)
        n_b = len(runs_b)
    except TypeError as exc:
        raise ValueError("runs_a and runs_b must be sequences") from exc
    if n_a != n_b:
        raise ValueError("runs_a and runs_b must contain the same number of examples")
    if n_a < 1:
        raise ValueError("at least one example is required")

    validated_a = [
        _validate_stream(stream, model="A", example_index=index)
        for index, stream in enumerate(runs_a)
    ]
    validated_b = [
        _validate_stream(stream, model="B", example_index=index)
        for index, stream in enumerate(runs_b)
    ]
    counts = [int(stream.size) for stream in (*validated_a, *validated_b)]
    all_singleton = all(count == 1 for count in counts)
    all_observed_constant = all(
        bool((values == values[0]).all())
        for values in (*validated_a, *validated_b)
    )
    if not all_singleton and any(count == 1 for count in counts):
        raise ValueError(
            "singleton and identified run streams cannot be mixed; "
            "apply and count the explicit SKIP policy before calling "
            "predictive_rerun_normal_theory"
        )

    point_components = [
        float(values_b.mean()) - float(values_a.mean())
        for values_a, values_b in zip(validated_a, validated_b, strict=True)
    ]
    point_estimate = math.fsum(point_components) / n_a
    if not math.isfinite(point_estimate):
        raise ValueError("run values produced a non-finite point estimate")

    central_mass_value = float(central_mass)
    future_runs_value = int(future_runs)
    if all_singleton:
        return _point_result(
            point_estimate,
            central_mass=central_mass_value,
            n_examples=n_a,
            future_runs=future_runs_value,
            reason="single_run",
        )
    if all_observed_constant:
        return _point_result(
            point_estimate,
            central_mass=central_mass_value,
            n_examples=n_a,
            future_runs=future_runs_value,
            reason="observed_zero_variance",
        )

    aggregate_factor = float(n_a * n_a)
    q_and_df: list[tuple[float, int]] = []
    for values_a, values_b in zip(validated_a, validated_b, strict=True):
        for values in (values_a, values_b):
            observed_runs = int(values.size)
            sample_variance = float(values.var(ddof=1))
            q = (
                sample_variance
                * (1.0 / observed_runs + 1.0 / future_runs_value)
                / aggregate_factor
            )
            if not math.isfinite(q):
                raise ValueError("run values produced a non-finite prediction variance")
            q_and_df.append((q, observed_runs - 1))

    prediction_variance = math.fsum(q for q, _ in q_and_df)
    if not math.isfinite(prediction_variance):
        raise ValueError("run values produced a non-finite prediction variance")
    if prediction_variance == 0.0:
        raise ValueError(
            "run values produced zero prediction variance from nonconstant streams"
        )

    max_q = max(q for q, _ in q_and_df)
    scaled_denominator = math.fsum(
        (q / max_q) ** 2 / stream_df for q, stream_df in q_and_df
    )
    degrees_of_freedom = (
        (prediction_variance / max_q) ** 2 / scaled_denominator
    )
    standard_error = math.sqrt(prediction_variance)
    tail_mass = 1.0 - central_mass_value
    critical = float(
        _sp.t.ppf(1.0 - tail_mass / 2.0, degrees_of_freedom)
    )
    probability = float(
        _sp.t.cdf(
            -point_estimate / standard_error,
            degrees_of_freedom,
        )
    )
    range_low = point_estimate - critical * standard_error
    range_high = point_estimate + critical * standard_error
    if not all(
        math.isfinite(value)
        for value in (
            degrees_of_freedom,
            critical,
            probability,
            range_low,
            range_high,
        )
    ):
        raise ValueError("run values produced non-finite predictive statistics")

    return NormalTheoryPredictiveRerunResult(
        point_estimate_b_minus_a=float(point_estimate),
        normal_theory_probability_a_better=float(probability),
        normal_theory_range_low=float(range_low),
        normal_theory_range_high=float(range_high),
        central_mass=central_mass_value,
        prediction_variance=float(prediction_variance),
        degrees_of_freedom=float(degrees_of_freedom),
        n_examples=int(n_a),
        future_runs=future_runs_value,
        method=_METHOD,
        degenerate_reason=None,
    )
