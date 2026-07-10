"""The public Python API.

    import evaltrust

    report = evaltrust.audit("results.json")
    if report.verdict.level is evaltrust.VerdictLevel.LOW:
        raise SystemExit("Evaluation is not trustworthy enough to ship on.")

    report.to_dict()   # machine-readable, JSON-serializable

``audit`` accepts whatever you have: a path to a results file, two paths to pair
single-model files, or an already-loaded ``EvalData``.
"""

from __future__ import annotations

from .audit.runner import AuditReport, run_audit
from .audit.suite import SuiteReport, audit_suite as _audit_suite
from .core.ingest import load, load_comparison, load_suite
from .core.schema import EvalData


def audit(
    source: "str | list[str] | tuple[str, ...] | EvalData",
    *,
    model_a: str | None = None,
    model_b: str | None = None,
    alpha: float = 0.05,
    equivalence_margin: float = 0.05,
    threshold: float | None = None,
    seed: int = 0,
) -> AuditReport:
    """Audit an evaluation and return an :class:`AuditReport`.

    ``source`` may be:
      - a path to a results file (JSON or CSV),
      - a list/tuple of two paths to single-model files, paired by example id,
      - an :class:`EvalData` you built yourself.

    ``model_a`` / ``model_b`` choose which two models to compare (or, for the
    two-file form, label the two files).
    """
    kw = dict(alpha=alpha, equivalence_margin=equivalence_margin,
              threshold=threshold, seed=seed)

    if isinstance(source, EvalData):
        return run_audit(source, model_a=model_a, model_b=model_b, **kw)

    if isinstance(source, (list, tuple)):
        paths = list(source)
        if len(paths) == 1:
            data = load(paths[0])
            return run_audit(data, model_a=model_a, model_b=model_b, **kw)
        data = load_comparison(paths, label_a=model_a, label_b=model_b)
        return run_audit(data, **kw)

    data = load(source)
    return run_audit(data, model_a=model_a, model_b=model_b, **kw)


def audit_suite(
    source: "str | dict[str, EvalData]",
    *,
    model_a: str | None = None,
    model_b: str | None = None,
    alpha: float = 0.05,
    equivalence_margin: float = 0.05,
    seed: int = 0,
    correction: str | None = None,
) -> SuiteReport:
    """Audit a multi-metric suite and return a :class:`SuiteReport`.

    ``source`` is a path to a file with a ``metric`` column, or a ready-made
    ``{metric: EvalData}`` mapping. Every metric is audited for the same model
    pair, with the significance threshold corrected for the number of metrics.

    ``correction`` is ``"bonferroni"``, ``"holm"``, or ``"none"``; when omitted
    it falls back to :class:`AuditConfig`'s default so the default lives in one
    place.
    """
    suite = load_suite(source) if isinstance(source, str) else source
    return _audit_suite(suite, model_a=model_a, model_b=model_b, alpha=alpha,
                        equivalence_margin=equivalence_margin, seed=seed,
                        correction=correction)
