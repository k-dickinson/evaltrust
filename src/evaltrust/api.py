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

    ``source`` is a results file (JSON/JSONL/CSV), two single-model files to pair,
    or an :class:`EvalData`. ``model_a`` / ``model_b`` pick or label the two models.
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
