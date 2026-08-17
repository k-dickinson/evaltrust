"""The `evaltrust` command-line interface.

One command, no configuration:

    evaltrust audit results.json

Reads your existing eval output, audits whether its conclusion is trustworthy,
and prints the verdict. ``--strict`` makes a Low-Confidence verdict fail the
process, so an audit can gate CI the way tests do.
"""

from __future__ import annotations

import json
from collections import OrderedDict
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
from .audit.two_sample import audit_two_sample
from .core.ingest import load_comparison, load_run_level, load_suite
from .diff import compare
from .report.html import render_html, render_html_from_parts
from .report.terminal import (
    print_diff,
    print_report,
    print_suite,
    render_diff_plain,
    render_markdown,
    render_markdown_from_parts,
    render_plain,
    render_plain_from_parts,
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
    bayesian: Optional[bool] = typer.Option(
        None,
        "--bayesian/--no-bayesian",
        help="Add the Bayesian decisive-pair win probability."),
    win_rate: Optional[bool] = typer.Option(
        None,
        "--win-rate/--no-win-rate",
        help=(
            "Add how often the first model scores higher on paired examples, "
            "with ties half credit and a bootstrap interval."
        )),
    run_aware: Optional[bool] = typer.Option(
        None,
        "--run-aware/--no-run-aware",
        help=(
            "Add the fixed-example normal-theory predictive rerun finding. "
            "Requires --future-runs. Approximate-range coverage fell to ~0.933 "
            "with 8 examples and 3 observed runs, and ~0.937 with 25 examples "
            "and 6 observed runs, for measured skewed-stream cases."
        )),
    future_runs: Optional[int] = typer.Option(
        None,
        "--future-runs",
        help="Future runs per model and example for the run-aware prediction."),
    correction: Optional[str] = typer.Option(
        None, "--correction",
        help="Multiple-comparison correction: bonferroni (default), holm, or none."),
    all_pairs: Optional[bool] = typer.Option(
        None, "--all-pairs/--no-all-pairs",
        help="Also compare every model pair with one family-wide correction."),
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
    run_level: bool = typer.Option(
        False, "--run-level",
        help=(
            "Treat the input as run-level (aggregate) scores — one total score "
            "per run, no per-example breakdown.  Runs an unpaired two-sample "
            "comparison (bootstrap P(A>B) + Mann-Whitney U) instead of the "
            "standard paired path.  Requires exactly one file and both "
            "--model-a and --model-b (or two-column wide CSV)."
        )),
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
                                   ("bayesian", bayesian),
                                   ("win_rate", win_rate),
                                   ("run_aware", run_aware),
                                   ("run_aware_future_runs", future_runs),
                                   ("reference_judge", reference_judge),
                                   ("correction", correction),
                                   ("all_pairs", all_pairs))
                 if v is not None}
    try:
        cfg = replace(cfg, **overrides)
    except ValueError as e:
        _err.print(f"[red]Could not apply config: {e}[/red]")
        raise typer.Exit(code=2)

    # ---- Run-level (unpaired) path ----
    if run_level:
        if len(results) != 1:
            _err.print("[red]--run-level requires exactly one input file.[/red]")
            raise typer.Exit(code=2)
        try:
            rl_data = load_run_level(results[0], model_a=model_a, model_b=model_b)
        except (FileNotFoundError, ValueError) as e:
            _err.print(f"[red]{e}[/red]")
            raise typer.Exit(code=2)
        findings = audit_two_sample(
            rl_data,
            alpha=cfg.alpha,
            confidence=1.0 - cfg.alpha,
            n_resamples=10_000,
            seed=cfg.seed,
        )
        from .core.schema import Status
        from .versions import METHODOLOGY_VERSION

        # Build a lightweight report dict for JSON / plain output.
        report_dict = {
            "schema_version": SCHEMA_VERSION,
            "methodology_version": METHODOLOGY_VERSION,
            "comparison_path": "unpaired_two_sample",
            "model_a": rl_data.model_a,
            "model_b": rl_data.model_b,
            "n_runs_a": rl_data.n_a,
            "n_runs_b": rl_data.n_b,
            "source_format": rl_data.source_format,
            "findings": [f.to_dict() for f in findings],
        }

        # Build the subtitle string shared by all three text formats.
        rl_subtitle = (
            f"{rl_data.model_a} vs {rl_data.model_b} · "
            f"{rl_data.n_a} runs / {rl_data.n_b} runs"
        )

        if as_json:
            typer.echo(json.dumps(report_dict, indent=2))
        elif md:
            # Delegate to the shared renderer (no verdict → verdict args omitted).
            typer.echo(
                render_markdown_from_parts(rl_subtitle, findings, explain=explain),
                nl=False,
            )
        elif plain:
            # Delegate to the shared renderer (no verdict → verdict args omitted).
            typer.echo(
                render_plain_from_parts(rl_subtitle, findings, explain=explain),
                nl=False,
            )
        else:
            # Rich colour terminal — model names may contain square brackets that
            # Rich interprets as markup, so escape before passing to print.
            from rich.markup import escape as _escape
            from rich.text import Text as _Text
            from .report.terminal import (
                _display_title, _display_how_detected,
                _grouped, _SYMBOL, _bullets,
            )
            _con = Console()
            _con.print(
                f"\n[bold]EvalTrust[/bold]  "
                f"[dim]{_escape(rl_subtitle)}[/dim]\n"
            )
            for pillar, items in _grouped(findings).items():
                _con.print(f"[bold]{_escape(pillar)}[/bold]")
                for f in items:
                    sym, color = _SYMBOL[f.status]
                    _title = _escape(_display_title(f))
                    _con.print(
                        f"  [{color}]{sym}[/{color}] "
                        + (_title if f.status is not Status.SKIP
                           else f"[dim]{_title}[/dim]")
                    )
                    if explain:
                        _con.print(
                            f"    [dim]{_escape(_display_how_detected(f))}[/dim]"
                        )
            _t = _Text()
            _bullets(_t, "What to do",
                     [f.how_to_fix for f in findings
                      if f.status in (Status.WARN, Status.FAIL)])
            _bullets(_t, "To check more",
                     [f.how_to_fix for f in findings if f.status is Status.SKIP],
                     style="dim")
            if _t:
                _con.print(_t)
            else:
                _con.print()

        if html_out is not None:
            Path(html_out).write_text(
                render_html_from_parts(
                    rl_subtitle,
                    findings,
                    title_suffix=(
                        f"Run-level: {rl_data.model_a} vs {rl_data.model_b}"
                    ),
                    explain=explain,
                ),
                encoding="utf-8",
            )

        # Exit code: fail if --strict or --fail-under and worst finding is bad.
        worst_status = max(
            (f.status for f in findings),
            key=lambda s: {
                Status.FAIL: 3, Status.WARN: 2,
                Status.PASS: 1, Status.SKIP: 0
            }[s],
        )
        level_str = {
            Status.PASS: "high",
            Status.WARN: "moderate",
            Status.FAIL: "low",
            Status.SKIP: "low",
        }[worst_status]
        _run_level_level = coerce_level(level_str)
        threshold = fail_under if fail_under is not None else ("moderate" if strict else None)
        if threshold is not None:
            try:
                minimum = coerce_level(threshold)
            except ValueError as e:
                _err.print(f"[red]{e}[/red]")
                raise typer.Exit(code=2)
            if _LEVEL_RANK[_run_level_level] < _LEVEL_RANK[minimum]:
                raise typer.Exit(code=1)
        return

    suite_report = None
    report = None
    try:
        _EFFICIENCY_KEYS = {"token_count", "latency"}

        if len(results) == 2:
            data = load_comparison(results, label_a=model_a, label_b=model_b)
            # Two input files already define one pair. Keep all-pairs scoped to
            # a single file that declares the model family.
            # Also try to extract token_count and latency from each file's suite
            # so they can be passed as efficiency side-channels.
            token_count_data = None
            latency_data = None
            try:
                suite_a = load_suite(results[0])
                suite_b = load_suite(results[1])
                # Merge token_count from both files if present in both.
                if "token_count" in suite_a and "token_count" in suite_b:
                    from .core.pairing import primary_model
                    la = primary_model(suite_a["token_count"])
                    lb = primary_model(suite_b["token_count"])
                    lbl_a = (model_a or data.models[0])
                    lbl_b = (model_b or data.models[1])
                    from .core.pairing import merge_two as _merge
                    token_count_data = _merge(
                        suite_a["token_count"], suite_b["token_count"],
                        lbl_a, lbl_b)
                if "latency" in suite_a and "latency" in suite_b:
                    lbl_a = (model_a or data.models[0])
                    lbl_b = (model_b or data.models[1])
                    from .core.pairing import merge_two as _merge
                    latency_data = _merge(
                        suite_a["latency"], suite_b["latency"],
                        lbl_a, lbl_b)
            except Exception:
                # Efficiency extraction is best-effort: if anything goes wrong
                # (format doesn't support suites, pairing fails, etc.), just
                # proceed without efficiency findings.
                token_count_data = None
                latency_data = None

            report = run_audit(
                data, config=replace(cfg, all_pairs=False),
                slice_by=slice_by,
                token_count_data=token_count_data,
                latency_data=latency_data)
        else:
            suite = load_suite(results[0])
            if len(suite) > 1:
                # Extract operational columns (token_count, latency) before
                # deciding whether this is a multi-metric suite or a single
                # quality metric with cost/latency side-channels.
                # The MLflow adapter (and any future adapter that follows the
                # same convention) puts these in the suite keyed by their bare
                # names.  We pull them out here so run_audit receives them as
                # first-class efficiency datasets rather than treating them as
                # quality metrics.
                token_count_data = suite.get("token_count")
                latency_data = suite.get("latency")
                # Quality suite: everything except the operational columns.
                quality_suite = OrderedDict(
                    (k, v) for k, v in suite.items() if k not in _EFFICIENCY_KEYS
                )
                if len(quality_suite) > 1:
                    # Genuine multi-metric suite: audit as a suite.
                    # Efficiency data is advisory; the suite path doesn't yet
                    # thread it through, so we skip it silently here and let
                    # the suite report stand on its own.
                    suite_report = audit_suite(
                        quality_suite, model_a=model_a, model_b=model_b, config=cfg)
                elif len(quality_suite) == 1:
                    # Single quality metric + optional cost/latency side-channels.
                    data = next(iter(quality_suite.values()))
                    report = run_audit(data, model_a=model_a, model_b=model_b,
                                       threshold=threshold, config=cfg,
                                       slice_by=slice_by,
                                       token_count_data=token_count_data,
                                       latency_data=latency_data)
                else:
                    # Suite was only operational columns (no quality metric).
                    # Fall back to the original suite path so the user gets a
                    # useful error rather than a silent empty report.
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


@app.command()
def contamination(
    benchmark: str = typer.Argument(..., help="Path to the benchmark dataset file."),
    reference: str = typer.Argument(..., help="Path to the reference/training dataset file."),
    column: str = typer.Option("prompt", "--column", help="The column or key containing the text (e.g. 'prompt', 'text', 'input')."),
    fail_over: Optional[float] = typer.Option(
        None, "--fail-over",
        help="Exit non-zero when contamination fraction exceeds this value (0–1). Defaults to 0.15.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON (for CI and tooling)."),
) -> None:
    """Audit a benchmark dataset against a reference dataset for contamination.

    Note: Near-match detection uses a quadratic O(N*M) SequenceMatcher approach intended
    for modest dataset sizes. For very large reference corpora, a scalable n-gram
    or MinHash approach should be used instead.
    """
    from .audit.contamination import run_contamination_audit
    import csv

    def load_texts(path: str, col: str) -> list[str]:
        p = Path(path)
        if not p.exists():
            _err.print(f"[red]File not found: {path}[/red]")
            raise typer.Exit(code=2)

        text = p.read_text(encoding="utf-8")
        suffix = p.suffix.lower()

        texts: list[str] = []
        skipped = 0
        if suffix == ".csv":
            import io
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                if col not in row:
                    skipped += 1
                    continue
                val = row[col]
                if val is None or not isinstance(val, str):
                    _err.print(f"[red]Column '{col}' in CSV has a non-string value.[/red]")
                    raise typer.Exit(code=2)
                texts.append(val)
        elif suffix == ".jsonl":
            for i, line in enumerate(text.split('\n')):
                line = line.rstrip('\r')
                if not line.strip():
                    continue
                row = json.loads(line)
                if col not in row:
                    skipped += 1
                    continue
                val = row[col]
                if not isinstance(val, str):
                    _err.print(f"[red]Key '{col}' in JSONL line {i+1} is not a string.[/red]")
                    raise typer.Exit(code=2)
                texts.append(val)
        elif suffix == ".json":
            data = json.loads(text)
            if not isinstance(data, list):
                _err.print("[red]JSON file must contain an array of objects.[/red]")
                raise typer.Exit(code=2)
            for row in data:
                if not isinstance(row, dict) or col not in row:
                    skipped += 1
                    continue
                val = row[col]
                if not isinstance(val, str):
                    _err.print(f"[red]Key '{col}' in JSON array item is not a string.[/red]")
                    raise typer.Exit(code=2)
                texts.append(val)
        else:
            _err.print(f"[red]Unsupported file format: {suffix}[/red]")
            raise typer.Exit(code=2)

        if skipped:
            _warn.print(
                f"[yellow]{skipped} row(s) missing column '{col}' in '{path}' — skipped.[/yellow]"
            )
        if not texts:
            _err.print(f"[red]No usable rows found in '{path}' for column '{col}'.[/red]")
            raise typer.Exit(code=2)

        return texts

    try:
        bench_texts = load_texts(benchmark, column)
        ref_texts = load_texts(reference, column)
    except Exception as e:
        if isinstance(e, typer.Exit):
            raise
        _err.print(f"[red]Error parsing files: {e}[/red]")
        raise typer.Exit(code=2) from None

    result = run_contamination_audit(bench_texts, ref_texts)

    # Validate --fail-over range.
    if fail_over is not None and not (0.0 <= fail_over <= 1.0):
        _err.print(f"[red]--fail-over must be between 0 and 1 (got {fail_over})[/red]")
        raise typer.Exit(code=2)

    # Determine the effective fail threshold (default 15 %).
    threshold_frac = fail_over if fail_over is not None else 0.15

    if as_json:
        typer.echo(json.dumps({
            "total_items": result.total_items,
            "exact_matches": result.exact_matches,
            "near_matches": result.near_matches,
            "contamination_fraction": result.contamination_fraction,
        }, indent=2))
    else:
        frac = result.contamination_fraction * 100
        color = typer.colors.GREEN if frac < 5 else typer.colors.YELLOW if frac < 15 else typer.colors.RED
        typer.echo("\nEvalTrust Benchmark Contamination Audit")
        typer.echo("=======================================")
        typer.echo(f"Total benchmark items: {result.total_items}")
        typer.echo(f"Exact matches found:   {result.exact_matches}")
        typer.echo(f"Near matches found:    {result.near_matches}")
        typer.echo("Contamination level:   ", nl=False)
        typer.secho(f"{frac:.1f}%", fg=color, bold=True)
        typer.echo("")
        if frac > 0:
            typer.secho(
                "\nWarning: Overlap detected between benchmark and reference sets.",
                fg=typer.colors.YELLOW,
            )

    if result.contamination_fraction >= threshold_frac:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
