"""The public Python API.

    import evaltrust

    report = evaltrust.audit("results.json")
    if report.verdict.level is evaltrust.VerdictLevel.LOW:
        raise SystemExit("Evaluation is not trustworthy enough to ship on.")

    report.to_dict()   # machine-readable, JSON-serializable

``audit`` accepts whatever you have: a path to a results file, two paths to pair
single-model files, or an already-loaded ``EvalData``.

Opt-in features (``bayesian``, ``all_pairs``, ``run_aware`` / ``run_aware_future_runs``,
``correction``) can be enabled either via the ``config=`` passthrough or as direct
keyword arguments.  Both are equivalent; ``config`` takes precedence when supplied.

Run-level comparison and contamination detection have their own top-level entry
points: :func:`audit_run_level` and :func:`audit_contamination`.
"""

from __future__ import annotations

from .audit.contamination import ContaminationResult, run_contamination_audit
from .audit.runner import AuditReport, run_audit
from .audit.suite import SuiteReport, audit_suite as _audit_suite
from .audit.two_sample import audit_two_sample
from .config import AuditConfig
from .core.ingest import load, load_comparison, load_run_level, load_suite
from .core.schema import EvalData, Finding, RunLevelData


def audit(
    source: "str | list[str] | tuple[str, ...] | EvalData",
    *,
    model_a: str | None = None,
    model_b: str | None = None,
    # --- common statistical knobs ---
    alpha: float = 0.05,
    equivalence_margin: float = 0.05,
    threshold: float | None = None,
    seed: int = 0,
    slice_by: str | None = None,
    # --- opt-in features (mirrors AuditConfig fields) ---
    bayesian: bool = False,
    all_pairs: bool = False,
    run_aware: bool = False,
    run_aware_future_runs: int | None = None,
    correction: str = "bonferroni",
    # --- config passthrough (takes precedence over loose kwargs) ---
    config: "AuditConfig | None" = None,
) -> AuditReport:
    """Audit an evaluation and return an :class:`AuditReport`.

    Parameters
    ----------
    source:
        A results file path (JSON/JSONL/CSV), a list/tuple of two single-model
        file paths to pair, or an already-loaded :class:`EvalData`.
    model_a, model_b:
        Pick or label the two models to compare.
    alpha:
        Significance level (default 0.05).
    equivalence_margin:
        Largest gap considered negligible for equivalence testing (default 0.05).
    threshold:
        For single-model audits: the target score to test against.  Ignored for
        two-model comparisons.
    seed:
        RNG seed for reproducibility (default 0).
    slice_by:
        Name of a per-example attribute to break the comparison down by slice,
        with Bonferroni correction across slices.
    bayesian:
        Enable the optional Bayesian win-probability view (default False).
    all_pairs:
        Compare every model pair in the data, not just the top two (default False).
    run_aware:
        Enable the predictive-rerun repeatability check (default False).
        Requires ``run_aware_future_runs`` to be set.
    run_aware_future_runs:
        Number of future runs to predict when ``run_aware=True``.
    correction:
        Multiple-comparison correction for suite audits: ``"bonferroni"``,
        ``"holm"``, or ``"none"`` (default ``"bonferroni"``).
    config:
        A fully built :class:`AuditConfig`.  When supplied, all loose statistical
        kwargs above are ignored and the config is used directly.

    Returns
    -------
    AuditReport
    """
    if config is None:
        config = AuditConfig(
            alpha=alpha,
            equivalence_margin=equivalence_margin,
            seed=seed,
            bayesian=bayesian,
            all_pairs=all_pairs,
            run_aware=run_aware,
            run_aware_future_runs=run_aware_future_runs if run_aware else None,
            correction=correction,
        )

    kw = dict(config=config, threshold=threshold, slice_by=slice_by)

    if isinstance(source, EvalData):
        return run_audit(source, model_a=model_a, model_b=model_b, **kw)

    if isinstance(source, (list, tuple)):
        paths = list(source)
        if len(paths) == 1:
            data = load(paths[0])
            return run_audit(data, model_a=model_a, model_b=model_b, **kw)
        data = load_comparison(paths, label_a=model_a, label_b=model_b)
        # Two-model comparison ignores threshold (single-model parameter)
        kw_comparison = {k: v for k, v in kw.items() if k != "threshold"}
        return run_audit(data, **kw_comparison)

    data = load(source)
    return run_audit(data, model_a=model_a, model_b=model_b, **kw)


def audit_run_level(
    path: str,
    *,
    model_a: str | None = None,
    model_b: str | None = None,
    alpha: float = 0.05,
    seed: int = 0,
    n_resamples: int = 10_000,
    config: "AuditConfig | None" = None,
) -> "list[Finding]":
    """Audit run-level (aggregate) scores and return a list of :class:`Finding` objects.

    Use this when your harness emits a single total score per run rather than
    per-example scores, making per-example pairing impossible.  The underlying
    analysis uses an unpaired two-sample approach: a Mann-Whitney U test together
    with a bootstrap estimate of P(model_a > model_b).

    Parameters
    ----------
    path:
        Path to a CSV or JSON file with run-level scores.  See
        :func:`~evaltrust.core.ingest.load_run_level` for the accepted formats.
    model_a, model_b:
        Labels that identify the two models in the file.
    alpha:
        Significance level (default 0.05).
    seed:
        RNG seed for reproducibility (default 0).
    n_resamples:
        Bootstrap iterations (default 10 000).
    config:
        A fully built :class:`AuditConfig`.  When supplied, ``alpha``, ``seed``,
        and ``n_resamples`` are taken from it instead.

    Returns
    -------
    list[Finding]
        Three findings: ``decision``, ``effect_size``, ``precision``.

    Example
    -------
    ::

        import evaltrust
        findings = evaltrust.audit_run_level(
            "run_scores.csv", model_a="gpt-4", model_b="claude-3"
        )
        for f in findings:
            print(f.status.name, f.title)
    """
    if config is not None:
        alpha = config.alpha
        seed = config.seed
        n_resamples = config.n_resamples

    data: RunLevelData = load_run_level(path, model_a=model_a, model_b=model_b)
    return audit_two_sample(data, alpha=alpha, seed=seed, n_resamples=n_resamples)


def audit_contamination(
    benchmark: "list[str]",
    reference: "list[str]",
    *,
    threshold: float = 0.85,
) -> ContaminationResult:
    """Check a benchmark for contamination against a reference corpus.

    Detects examples in ``benchmark`` that appear verbatim or near-verbatim in
    ``reference`` (e.g. a training set or a known-leaked set).  Returns a
    :class:`~evaltrust.audit.contamination.ContaminationResult` with exact-match
    counts, near-match counts, and the overall contamination fraction.

    Parameters
    ----------
    benchmark:
        The benchmark prompts or examples to audit (list of strings).
    reference:
        The reference corpus to check against (list of strings).
    threshold:
        Similarity threshold for near-match detection, in [0, 1] (default 0.85).
        A value of 1.0 only flags exact matches.

    Returns
    -------
    ContaminationResult
        Dataclass with ``exact_matches``, ``near_matches``, ``total_items``, and
        ``contamination_fraction``.

    Example
    -------
    ::

        import evaltrust
        result = evaltrust.audit_contamination(
            benchmark=my_benchmark_prompts,
            reference=training_corpus,
        )
        print(f"Contamination: {result.contamination_fraction:.1%}")
    """
    return run_contamination_audit(benchmark, reference, threshold=threshold)


def audit_suite(
    source: "str | dict[str, EvalData]",
    *,
    model_a: str | None = None,
    model_b: str | None = None,
    alpha: float = 0.05,
    equivalence_margin: float = 0.05,
    seed: int = 0,
    correction: str = "bonferroni",
) -> SuiteReport:
    """Audit a multi-metric suite and return a :class:`SuiteReport`.

    ``source`` is a file with a ``metric`` column or a ``{metric: EvalData}`` map.
    ``correction`` is ``{"bonferroni", "holm", "none"}``.
    """
    suite = load_suite(source) if isinstance(source, str) else source
    return _audit_suite(suite, model_a=model_a, model_b=model_b, alpha=alpha,
                        equivalence_margin=equivalence_margin, seed=seed,
                        correction=correction)
