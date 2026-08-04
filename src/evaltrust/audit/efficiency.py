"""Efficiency audit -- cost and latency alongside quality.

Advisory-only: the findings never change the verdict level (status is always
PASS or SKIP).  The pillar is silently absent when no cost or latency data is
present, so default output for quality-only files is unchanged.

The findings make the quality/cost tradeoff explicit:

  Efficiency
  ok   model_b uses 3.2x the tokens of model_a for a +4.1 pt quality gain

rather than leaving it as vague advice text ("decide on cost or speed").

Data flow
---------
The caller passes separate ``EvalData`` objects for token counts and for
latency, each keyed by model -- the same shape as the quality ``EvalData``.
This keeps the quality audit's ``EvalData`` clean and requires no schema
changes to ``Example``.

The adapter layer (e.g. the MLflow adapter) already reads ``token_count``
and ``latency`` as named metric columns; the CLI or the Python API can pass
those datasets in alongside the quality dataset.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.schema import EvalData, Finding, Status

PILLAR = "Efficiency"


def audit_efficiency(
    quality_data: EvalData,
    model_a: str,
    model_b: str,
    *,
    token_count_data: EvalData | None = None,
    latency_data: EvalData | None = None,
    latency_unit: str = "ms",
) -> list[Finding]:
    """Return advisory efficiency findings when cost or latency data is present.

    Parameters
    ----------
    quality_data:
        The quality ``EvalData`` already used by the other audit checks.  Used
        only to read the mean quality scores for each model so the quality delta
        can be shown alongside the cost/latency delta.
    model_a, model_b:
        The two models being compared (same labels as the quality audit).
    token_count_data:
        Optional ``EvalData`` whose ``scores`` carry per-example token counts
        for the same two models.  When ``None`` the token-count finding is
        skipped.
    latency_data:
        Optional ``EvalData`` whose ``scores`` carry per-example latency
        values for the same two models.  When ``None`` the latency finding is
        skipped.
    latency_unit:
        The unit label for latency values (default ``"ms"``).  Pass ``"s"``
        if the caller stores seconds, ``"us"`` for microseconds, etc.  The
        value is used verbatim in the ``how_detected`` text so the finding
        always reflects the actual unit of the data.

    Returns
    -------
    list[Finding]
        Zero findings when neither dataset is supplied.  One finding per
        dataset that is supplied.  All findings have ``status = Status.PASS``
        (informational) so they never lower the confidence verdict.
    """
    findings: list[Finding] = []

    # Mean quality scores (for the tradeoff narrative).
    mean_q_a = _mean(quality_data, model_a)
    mean_q_b = _mean(quality_data, model_b)
    quality_delta = mean_q_b - mean_q_a  # positive means B is better

    # Guard: if quality_delta is NaN (e.g. preference-only data where _mean
    # finds no scores for a model), we cannot produce a meaningful tradeoff
    # narrative and must not emit NaN into details (not JSON-serialisable by
    # strict parsers).  Treat as "quality data unavailable" and skip both
    # findings -- the quality audit itself will flag the missing scores.
    if math.isnan(quality_delta):
        return findings

    if token_count_data is not None:
        f = _token_finding(token_count_data, model_a, model_b,
                           quality_delta)
        if f is not None:
            findings.append(f)

    if latency_data is not None:
        f = _latency_finding(latency_data, model_a, model_b,
                             quality_delta, latency_unit)
        if f is not None:
            findings.append(f)

    return findings


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _mean(data: EvalData, model: str) -> float:
    vals = [ex.scores[model] for ex in data.examples if model in ex.scores]
    return float(np.mean(vals)) if vals else float("nan")


def _ratio_str(numerator: float, denominator: float) -> str:
    """Return 'Nx' where N = numerator / denominator, formatted compactly."""
    if denominator == 0:
        return "infinitely"
    return f"{numerator / denominator:.2g}x"


def _quality_delta_str(delta: float) -> str:
    """'+4.1 pt quality gain' or '-2.0 pt quality drop' or 'no quality change'.

    Uses the signed delta directly so drops are negative and gains are positive.
    """
    if abs(delta) < 1e-9:
        return "no quality change"
    direction = "gain" if delta > 0 else "drop"
    sign = "+" if delta > 0 else "-"
    return f"{sign}{abs(delta) * 100:.1f} pt quality {direction}"


def _token_finding(
    token_data: EvalData,
    model_a: str,
    model_b: str,
    quality_delta: float,
) -> Finding | None:
    mean_tok_a = _mean(token_data, model_a)
    mean_tok_b = _mean(token_data, model_b)

    if np.isnan(mean_tok_a) or np.isnan(mean_tok_b):
        return None  # data present but no overlap with these models; skip silently

    q_str = _quality_delta_str(quality_delta)

    # ratio_b_over_a: how many times more tokens does B use vs A?
    # > 1 means B is more expensive; < 1 means B is cheaper.
    ratio_b_over_a = _ratio_str(mean_tok_b, mean_tok_a)  # mean_tok_b / mean_tok_a
    ratio_a_over_b = _ratio_str(mean_tok_a, mean_tok_b)  # mean_tok_a / mean_tok_b (>1 when B cheaper)

    tok_equal = math.isclose(mean_tok_a, mean_tok_b, rel_tol=1e-9)

    if tok_equal:
        comparison = f"{model_b} uses the same number of tokens as {model_a}"
    elif mean_tok_b > mean_tok_a:
        # B costs more tokens
        comparison = f"{model_b} uses {ratio_b_over_a} the tokens of {model_a}"
    else:
        # B is cheaper -- show how much cheaper B is relative to A (ratio > 1)
        comparison = f"{model_b} uses {ratio_a_over_b} fewer tokens than {model_a}"

    title = f"Token cost: {comparison}"
    how = (
        f"{model_a} averaged {mean_tok_a:.1f} tokens/example; "
        f"{model_b} averaged {mean_tok_b:.1f} tokens/example. "
        f"Quality delta: {q_str}."
    )

    if tok_equal:
        # Cost is equal -- recommendation driven purely by quality delta.
        if abs(quality_delta) < 1e-9:
            fix = (
                f"The models are equal on both quality and token cost. "
                f"Either model is a valid choice."
            )
        elif quality_delta > 0:
            fix = (
                f"The models cost the same number of tokens. {model_b} is "
                f"better on quality ({q_str}) -- prefer {model_b}."
            )
        else:
            fix = (
                f"The models cost the same number of tokens. {model_b} is "
                f"worse on quality ({q_str}) -- prefer {model_a}."
            )
    elif abs(quality_delta) < 1e-9:
        if mean_tok_b > mean_tok_a:
            fix = (
                f"The models are equivalent on quality. {model_b} costs "
                f"{ratio_b_over_a} the tokens -- prefer {model_a}."
            )
        else:
            fix = (
                f"The models are equivalent on quality. {model_b} uses "
                f"{ratio_a_over_b} fewer tokens -- prefer {model_b}."
            )
    elif quality_delta > 0:
        # B is better on quality
        if mean_tok_b > mean_tok_a:
            fix = (
                f"{model_b} is better on quality ({q_str}) but costs "
                f"{ratio_b_over_a} the tokens. Decide whether the quality "
                f"gain justifies the extra cost."
            )
        else:
            fix = (
                f"{model_b} is both better on quality ({q_str}) and uses "
                f"{ratio_a_over_b} fewer tokens. Prefer {model_b}."
            )
    else:
        # B is worse on quality
        if mean_tok_b > mean_tok_a:
            fix = (
                f"{model_b} is worse on quality ({q_str}) and costs "
                f"{ratio_b_over_a} the tokens. Prefer {model_a}."
            )
        else:
            fix = (
                f"{model_b} is worse on quality ({q_str}) but uses "
                f"{ratio_a_over_b} fewer tokens. Decide whether the cost "
                f"saving justifies the quality drop."
            )

    tok_ratio = mean_tok_b / mean_tok_a if mean_tok_a != 0 else None
    return Finding(
        pillar=PILLAR,
        title=title,
        status=Status.PASS,  # advisory -- never lowers the verdict
        why=(
            "A model that is marginally better on quality but uses significantly "
            "more tokens may not be worth the extra cost in production. Showing "
            "the ratio makes the tradeoff concrete."
        ),
        how_detected=how,
        how_to_fix=fix,
        details={
            "check": "efficiency_tokens",
            "model_a": model_a,
            "model_b": model_b,
            "mean_tokens_a": mean_tok_a,
            "mean_tokens_b": mean_tok_b,
            "token_ratio_b_over_a": tok_ratio,
            "quality_delta": quality_delta,
        },
    )


def _latency_finding(
    latency_data: EvalData,
    model_a: str,
    model_b: str,
    quality_delta: float,
    latency_unit: str = "ms",
) -> Finding | None:
    mean_lat_a = _mean(latency_data, model_a)
    mean_lat_b = _mean(latency_data, model_b)

    if np.isnan(mean_lat_a) or np.isnan(mean_lat_b):
        return None

    q_str = _quality_delta_str(quality_delta)

    # ratio_b_over_a: how many times slower is B vs A?
    # > 1 means B is slower; < 1 means B is faster.
    ratio_b_over_a = _ratio_str(mean_lat_b, mean_lat_a)  # mean_lat_b / mean_lat_a
    ratio_a_over_b = _ratio_str(mean_lat_a, mean_lat_b)  # mean_lat_a / mean_lat_b (>1 when B faster)

    lat_equal = math.isclose(mean_lat_a, mean_lat_b, rel_tol=1e-9)

    if lat_equal:
        comparison = f"{model_b} has the same latency as {model_a}"
    elif mean_lat_b > mean_lat_a:
        # B is slower
        comparison = f"{model_b} is {ratio_b_over_a} slower than {model_a}"
    else:
        # B is faster -- show how much faster B is relative to A (ratio > 1)
        comparison = f"{model_b} is {ratio_a_over_b} faster than {model_a}"

    title = f"Latency: {comparison}"
    # Use caller-supplied unit so the label matches the data (ms, s, us, etc.)
    how = (
        f"{model_a} averaged {mean_lat_a:.1f} {latency_unit}/example; "
        f"{model_b} averaged {mean_lat_b:.1f} {latency_unit}/example. "
        f"Quality delta: {q_str}."
    )

    if lat_equal:
        # Latency is equal -- recommendation driven purely by quality delta.
        if abs(quality_delta) < 1e-9:
            fix = (
                f"The models are equal on both quality and latency. "
                f"Either model is a valid choice."
            )
        elif quality_delta > 0:
            fix = (
                f"The models have the same latency. {model_b} is better on "
                f"quality ({q_str}) -- prefer {model_b}."
            )
        else:
            fix = (
                f"The models have the same latency. {model_b} is worse on "
                f"quality ({q_str}) -- prefer {model_a}."
            )
    elif abs(quality_delta) < 1e-9:
        if mean_lat_b > mean_lat_a:
            fix = (
                f"The models are equivalent on quality. {model_b} is "
                f"{ratio_b_over_a} slower -- prefer {model_a}."
            )
        else:
            fix = (
                f"The models are equivalent on quality. {model_b} is "
                f"{ratio_a_over_b} faster -- prefer {model_b}."
            )
    elif quality_delta > 0:
        # B is better on quality
        if mean_lat_b > mean_lat_a:
            fix = (
                f"{model_b} is better on quality ({q_str}) but {ratio_b_over_a} "
                f"slower. Decide whether the latency increase is acceptable "
                f"for the quality gain."
            )
        else:
            fix = (
                f"{model_b} is both better on quality ({q_str}) and "
                f"{ratio_a_over_b} faster. Prefer {model_b}."
            )
    else:
        # B is worse on quality
        if mean_lat_b > mean_lat_a:
            fix = (
                f"{model_b} is worse on quality ({q_str}) and {ratio_b_over_a} "
                f"slower. Prefer {model_a}."
            )
        else:
            fix = (
                f"{model_b} is worse on quality ({q_str}) but {ratio_a_over_b} "
                f"faster. Decide whether the speed gain justifies the quality drop."
            )

    lat_ratio = mean_lat_b / mean_lat_a if mean_lat_a != 0 else None
    return Finding(
        pillar=PILLAR,
        title=title,
        status=Status.PASS,  # advisory -- never lowers the verdict
        why=(
            "A model that is marginally better on quality but noticeably slower "
            "may hurt user experience in latency-sensitive deployments. Showing "
            "the ratio makes the tradeoff concrete."
        ),
        how_detected=how,
        how_to_fix=fix,
        details={
            "check": "efficiency_latency",
            "model_a": model_a,
            "model_b": model_b,
            "mean_latency_a": mean_lat_a,
            "mean_latency_b": mean_lat_b,
            "latency_ratio_b_over_a": lat_ratio,
            "quality_delta": quality_delta,
        },
    )
