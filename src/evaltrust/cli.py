"""The `evaltrust` command-line interface.

One command, no configuration:

    evaltrust audit results.json

Reads your existing eval output, audits whether its conclusion is trustworthy,
and prints the verdict. ``--strict`` makes a Low-Confidence verdict fail the
process, so an audit can gate CI the way tests do.
"""

from __future__ import annotations

import json
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from .adapters.registry import UnknownFormatError
from .versions import SCHEMA_VERSION
from .audit.runner import run_audit
from .audit.suite import audit_suite
from .audit.verdict import _LEVEL_RANK, VerdictLevel, coerce_level
from .config import AuditConfig
from .core.ingest import load_comparison, load_suite
from .diff import compare
from .report.html import render_html
from .report.terminal import (
    print_diff,
    print_report,
    print_suite,
    render_diff_plain,
    render_markdown,
    render_plain,
    render_suite_markdown,
    render_suite_plain,
)

app = typer.Typer(
    add_completion=False,
    help="Check whether an eval's result is real or just noise.")
_err = Console(stderr=True)   # errors go to stderr, not stdout (#51)
# Diagnostics that must never mix into machine-readable stdout (e.g. a warning
# emitted alongside --json output) go here, on real stderr.
_warn = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if not value:
        return
    try:
        typer.echo(package_version("evaltrust"))
    except PackageNotFoundError:
        typer.echo("unknown")
    raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed evaltrust version and exit.",
    ),
) -> None:
    """EvalTrust: you ran an eval and got a score gap between two models. This
    tells you whether that gap is a real improvement or just luck, before you
    ship on it."""


@app.command()
def audit(
    results: List[str] = typer.Argument(
        ..., help="One results file (JSON/CSV), or two single-model files to compare."),
    model_a: Optional[str] = typer.Option(
        None, "--model-a", help="Model to compare (or label for the first file)."),
    model_b: Optional[str] = typer.Option(
        None, "--model-b", help="Model to compare (or label for the second file)."),
    alpha: Optional[float] = typer.Option(
        None, "--alpha", help="Significance level (overrides config; default 0.05)."),
    equivalence_margin: Optional[float] = typer.Option(
        None, "--equivalence-margin",
        help="Largest score gap considered practically negligible (for equivalence)."),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Seed for reproducible resampling."),
    correction: Optional[str] = typer.Option(
        None, "--correction",
        help="Multi-metric correction: bonferroni (default), holm, or none."),
    config_path: Optional[str] = typer.Option(
        None, "--config", help="Path to a config TOML (default: .evaltrust.toml or pyproject)."),
    reference_judge: Optional[str] = typer.Option(
        None, "--reference-judge",
        help="Name of the human/gold judge to calibrate the AI judges against."),
    threshold: Optional[float] = typer.Option(
        None, "--threshold",
        help="For a single-model eval, the target score to test against (e.g. 0.8)."),
    slice_by: Optional[str] = typer.Option(
        None, "--slice-by",
        help="Break the comparison down by this per-example attribute "
             "(e.g. category, difficulty, language)."),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero if confidence is Low."),
    fail_under: Optional[str] = typer.Option(
        None, "--fail-under",
        help="Exit non-zero if confidence is below this level (high/moderate/low)."),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the audit as JSON (for CI and tooling)."),
    html_out: Optional[str] = typer.Option(
        None, "--html", help="Write the audit as a standalone HTML file to this path."),
    plain: bool = typer.Option(
        False, "--plain", help="Plain ASCII output (no colour or Unicode)."),
    md: bool = typer.Option(
        False, "--md", help="Emit the audit as Markdown (for PR comments and docs)."),
    explain: bool = typer.Option(
        False, "--explain", help="Also show why each flag matters and how it was measured."),
) -> None:
    """Audit an evaluation and print a confidence verdict.

    Pass one file that already contains multiple models, or two single-model
    files (e.g. two DeepEval runs) to pair into an A-vs-B comparison.
    """
    if len(results) > 2:
        _err.print("[red]Provide at most two files (one, or two to compare).[/red]")
        raise typer.Exit(code=2)

    # Config: a file (explicit --config, or .evaltrust.toml / pyproject) provides
    # the team's policy; any flag the user passed overrides it.
    try:
        cfg = AuditConfig.load(path=config_path)
    except (OSError, ValueError) as e:
        _err.print(f"[red]Could not read config: {e}[/red]")
        raise typer.Exit(code=2)
    overrides = {k: v for k, v in (("alpha", alpha),
                                   ("equivalence_margin", equivalence_margin),
                                   ("seed", seed),
                                   ("reference_judge", reference_judge),
                                   ("correction", correction))
                 if v is not None}
    cfg = replace(cfg, **overrides)

    suite_report = None
    report = None
    try:
        if len(results) == 2:
            data = load_comparison(results, label_a=model_a, label_b=model_b)
            report = run_audit(data, config=cfg, slice_by=slice_by)
        else:
            suite = load_suite(results[0])
            if len(suite) > 1:
                suite_report = audit_suite(
                    suite, model_a=model_a, model_b=model_b, config=cfg)
            else:
                data = next(iter(suite.values()))
                report = run_audit(data, model_a=model_a, model_b=model_b,
                                   threshold=threshold, config=cfg,
                                   slice_by=slice_by)
    except OSError as e:  # missing, unreadable, or a directory given as a file
        _err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2)
    except UnknownFormatError as e:
        _err.print(f"[red]Unrecognised evaluation format.[/red]\n{e}")
        raise typer.Exit(code=2)
    except ValueError as e:
        _err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2)

    if suite_report is not None:
        if as_json:
            typer.echo(json.dumps(suite_report.to_dict(), indent=2))
        elif md:
            typer.echo(render_suite_markdown(suite_report, explain=explain), nl=False)
        elif plain:
            typer.echo(render_suite_plain(suite_report, explain=explain), nl=False)
        else:
            print_suite(suite_report, explain=explain)
        level = suite_report.overall_level
    else:
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2))
        elif md:
            typer.echo(render_markdown(report, explain=explain), nl=False)
        elif plain:
            typer.echo(render_plain(report, explain=explain), nl=False)
        else:
            print_report(report, explain=explain)
        level = report.verdict.level
    if html_out is not None:
        if suite_report is not None:
            # stderr, not stdout: in --json mode this must not trail the JSON body.
            _warn.print("[yellow]--html is not yet supported for multi-metric suites; "
                        "run a single metric to get an HTML report.[/yellow]")
        elif report is not None:
            Path(html_out).write_text(render_html(report, explain=explain), encoding="utf-8")

    threshold = fail_under if fail_under is not None else ("moderate" if strict else None)
    if threshold is not None:
        try:
            minimum = coerce_level(threshold)
        except ValueError as e:
            _err.print(f"[red]{e}[/red]")
            raise typer.Exit(code=2)
        if _LEVEL_RANK[level] < _LEVEL_RANK[minimum]:
            raise typer.Exit(code=1)


@app.command()
def diff(
    old: str = typer.Argument(..., help="Earlier audit JSON (from `audit --json`)."),
    new: str = typer.Argument(..., help="Newer audit JSON to compare against it."),
    fail_on_regression: bool = typer.Option(
        True, "--fail-on-regression/--no-fail-on-regression",
        help="Exit non-zero if the newer audit regressed."),
    as_json: bool = typer.Option(False, "--json", help="Emit the diff as JSON."),
    plain: bool = typer.Option(False, "--plain", help="Plain ASCII output."),
) -> None:
    """Compare two saved audits and flag regressions between runs."""
    try:
        old_data = json.loads(Path(old).read_text(encoding="utf-8"))
        new_data = json.loads(Path(new).read_text(encoding="utf-8"))
    except OSError as e:  # missing, unreadable, or a directory given as a file
        _err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2)
    except json.JSONDecodeError as e:
        _err.print(f"[red]Not valid audit JSON: {e}[/red]")
        raise typer.Exit(code=2)

    try:
        result = compare(old_data, new_data)
    except ValueError as e:
        _err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2)

    if as_json:
        typer.echo(json.dumps(
            {"schema_version": SCHEMA_VERSION,
             "regression": result.has_regression,
             "changes": [vars(c) for c in result.changes]}, indent=2))
    elif plain:
        typer.echo(render_diff_plain(result), nl=False)
    else:
        print_diff(result)

    if fail_on_regression and result.has_regression:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
