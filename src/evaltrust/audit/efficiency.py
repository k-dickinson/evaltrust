"""Efficiency audit — cost and latency alongside quality.

Advisory-only: the findings never change the verdict level (status is always
PASS or SKIP).  The pillar is silently absent when no cost or latency data is
present, so default output for quality-only files is unchanged.

The findings make the quality/cost tradeoff explicit:

  Efficiency
  ✓  model_b uses 3.2x the tokens for a +4.1 pt quality gain

rather than leaving it as vague advice text ("decide on cost or speed").

Data flow
---------
The caller passes separate ``EvalData`` objects for token counts and for
latency, each keyed by model — the same shape as the quality ``EvalData``.
This keeps the quality audit's ``EvalData`` clean and requires no schema
changes to ``Example``.

The adapter layer (e.g. the MLflow adapter) already reads ``token_count``
and ``latency`` as named metric columns; the CLI or the Python API can pass
those datasets in alongside the quality dataset.
"""

from __future__ import annotations

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
        values (milliseconds or any consistent unit) for the same two models.
        When ``None`` the latency finding is skipped.

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

    if token_count_data is not None:
        f = _token_finding(token_count_data, model_a, model_b,
                           mean_q_a, mean_q_b, quality_delta)
        if f is not None:
            findings.append(f)

    if latency_data is not None:
        f = _latency_finding(latency_data, model_a, model_b,
                             mean_q_a, mean_q_b, quality_delta)
        if f is not None:
            findings.append(f)

    return findings


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _mean(data: EvalData, model: str) -> float:
    vals = [ex.scores[model] for ex in data.examples if model in ex.scores]
    return float(np.mean(vals)) if vals else float("nan")


def _ratio_str(a: float, b: float) -> str:
    """'B uses 3.2x the tokens of A' or 'B uses 0.4x the tokens of A'."""
    if a == 0:
        return "∞x"
    return f"{b / a:.2g}x"


def _quality_delta_str(delta: float) -> str:
    """'+4.1 pt quality gain' or '−2.0 pt quality drop' or 'no quality change'."""
    if abs(delta) < 1e-9:
        return "no quality change"
    direction = "gain" if delta > 0 else "drop"
    return f"{abs(delta) * 100:+.1f} pt quality {direction}".replace("+", "+")


def _token_finding(
    token_data: EvalData,
    model_a: str,
    model_b: str,
    mean_q_a: float,
    mean_q_b: float,
    quality_delta: float,
) -> Finding | None:
    mean_tok_a = _mean(token_data, model_a)
    mean_tok_b = _mean(token_data, model_b)

    if np.isnan(mean_tok_a) or np.isnan(mean_tok_b):
        return None  # data present but no overlap with these models; skip silently

    ratio = _ratio_str(mean_tok_a, mean_tok_b)
    q_str = _quality_delta_str(quality_delta)

    # Pick the direction that reads most naturally.
    if mean_tok_b >= mean_tok_a:
        comparison = f"{model_b} uses {ratio} the tokens of {model_a}"
    else:
        inv = _ratio_str(mean_tok_b, mean_tok_a)
        comparison = f"{model_b} uses {inv} the tokens of {model_a} (cheaper)"

    title = f"Token cost: {comparison}"
    how = (
        f"{model_a} averaged {mean_tok_a:.1f} tokens/example; "
        f"{model_b} averaged {mean_tok_b:.1f} tokens/example "
        f"({ratio} ratio). Quality delta: {q_str}."
    )

    if abs(quality_delta) < 1e-9:
        fix = (
            f"The models are equivalent on quality. {model_b} costs "
            f"{ratio} the tokens — prefer the cheaper one."
        )
    elif quality_delta > 0:
        # B is better quality
        if mean_tok_b > mean_tok_a:
            fix = (
                f"{model_b} is better on quality ({q_str}) but costs "
                f"{ratio} the tokens. Decide whether the quality gain "
                f"justifies the extra cost."
            )
        else:
            fix = (
                f"{model_b} is both better on quality ({q_str}) and "
                f"cheaper on tokens. Prefer {model_b}."
            )
    else:
        # B is worse quality
        if mean_tok_b > mean_tok_a:
            fix = (
                f"{model_b} is worse on quality ({q_str}) and costs more "
                f"({ratio} the tokens). Prefer {model_a}."
            )
        else:
            fix = (
                f"{model_b} is worse on quality ({q_str}) but cheaper "
                f"({ratio} the tokens). Decide whether the cost saving "
                f"justifies the quality drop."
            )

    return Finding(
        pillar=PILLAR,
        title=title,
        status=Status.PASS,  # advisory — never lowers the verdict
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
            "token_ratio_b_over_a": (mean_tok_b / mean_tok_a
                                     if mean_tok_a != 0 else None),
            "quality_delta": quality_delta,
        },
    )


def _latency_finding(
    latency_data: EvalData,
    model_a: str,
    model_b: str,
    mean_q_a: float,
    mean_q_b: float,
    quality_delta: float,
) -> Finding | None:
    mean_lat_a = _mean(latency_data, model_a)
    mean_lat_b = _mean(latency_data, model_b)

    if np.isnan(mean_lat_a) or np.isnan(mean_lat_b):
        return None

    ratio = _ratio_str(mean_lat_a, mean_lat_b)
    q_str = _quality_delta_str(quality_delta)

    if mean_lat_b >= mean_lat_a:
        comparison = f"{model_b} is {ratio} slower than {model_a}"
    else:
        inv = _ratio_str(mean_lat_b, mean_lat_a)
        comparison = f"{model_b} is {inv} faster than {model_a}"

    title = f"Latency: {comparison}"
    how = (
        f"{model_a} averaged {mean_lat_a:.1f} ms/example; "
        f"{model_b} averaged {mean_lat_b:.1f} ms/example "
        f"({ratio} ratio). Quality delta: {q_str}."
    )

    if abs(quality_delta) < 1e-9:
        fix = (
            f"The models are equivalent on quality. {model_b} is "
            f"{ratio} {'slower' if mean_lat_b >= mean_lat_a else 'faster'} "
            f"— prefer the faster one."
        )
    elif quality_delta > 0:
        if mean_lat_b > mean_lat_a:
            fix = (
                f"{model_b} is better on quality ({q_str}) but "
                f"{ratio} slower. Decide whether the latency increase "
                f"is acceptable for the quality gain."
            )
        else:
            fix = (
                f"{model_b} is both better on quality ({q_str}) and "
                f"faster. Prefer {model_b}."
            )
    else:
        if mean_lat_b > mean_lat_a:
            fix = (
                f"{model_b} is worse on quality ({q_str}) and slower "
                f"({ratio}). Prefer {model_a}."
            )
        else:
            fix = (
                f"{model_b} is worse on quality ({q_str}) but faster "
                f"({ratio}). Decide whether the speed gain justifies "
                f"the quality drop."
            )

    return Finding(
        pillar=PILLAR,
        title=title,
        status=Status.PASS,  # advisory — never lowers the verdict
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
            "latency_ratio_b_over_a": (mean_lat_b / mean_lat_a
                                       if mean_lat_a != 0 else None),
            "quality_delta": quality_delta,
        },
    )
