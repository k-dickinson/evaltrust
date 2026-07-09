# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `evaltrust --version` prints the installed version and exits.
- Read line-delimited JSON (`.jsonl`) results — one record per line — through the
  existing record pipeline, with the same `metric`-column suite handling and
  skipped-row reporting as CSV. A malformed line fails with its line number.
- **Holm-Bonferroni correction for multi-metric suites.** `audit_suite` (and
  `--correction` / the `correction` config key) now accept `bonferroni`
  (default, unchanged), `holm`, or `none`. Holm is a step-down refinement that
  rejects at least as many metrics as Bonferroni at the same family-wise error
  rate; each metric is re-run at its Holm-effective alpha so its verdict, prose,
  and equivalence interval stay consistent. `SuiteReport` gains `metric_alphas`
  and `adjusted_p`.
- **BCa bootstrap intervals.** `stats.bootstrap_ci` gains a `method` option
  (`"percentile"`, the default, or `"bca"`) for bias-corrected and accelerated
  confidence intervals — second-order accurate and noticeably more faithful on
  skewed data. The percentile interval stays the default everywhere; BCa is
  validated against `scipy.stats.bootstrap(method="BCa")` and falls back to the
  percentile interval (never a silent `NaN`) on degenerate samples.
- **Single-model repeatability.** A single-model audit whose file has repeated
  runs now reports how stable the score is across reruns (the standard deviation
  of the per-run mean score), under the Repeatability pillar. It degrades to a
  SKIP when the file has no repeated runs.
- **Inspect (UK AISI) adapter:** read an Inspect `.json` eval log directly — model
  from `eval.model`, per-sample scorer grades (`C`/`I`/`P`/`N`) or numbers as the
  score. One model per log; compare two runs.

### Fixed

- **Binary effect size is now computed on the paired sample.** The pass-rate
  effect size (risk difference and Cohen's *h*) for pass/fail data was measured
  over every example each model scored, while the significance test (McNemar) and
  the confidence interval used only the paired examples. On data where the two
  models scored different example sets the effect size — and the PASS/WARN
  verdict it drives — could disagree with the significance test; both now use the
  same paired sample.

### Changed

- **Judge calibration** now handles continuous judge scores (e.g. a 1–5 rubric):
  it switches from exact-match agreement to a Spearman rank correlation against
  the reference judge, and names the metric in the finding so a correlation is
  never mistaken for an agreement rate. Binary pass/fail judges are unchanged.

## [0.5.0] — 2026-07-08

### Added

- **Single-model auditing.** Point EvalTrust at one model's scores (no comparison)
  and it reports whether you can trust the number: a confidence interval on the
  score (precision), and — with `--threshold` — whether the model really clears a
  target. A single-model file now audits instead of erroring.

### Changed

- README and docs refreshed to cover the full feature set (single-model,
  multi-metric, judge calibration, config, regression `diff`, integrations).

## [0.4.0] — 2026-07-08

### Added

- **Multi-metric suites.** A file with a `metric` column is audited one metric at
  a time (same model pair throughout), with the significance threshold
  **Bonferroni-corrected** for the number of metrics so testing many metrics
  doesn't manufacture false wins. The suite's confidence is its weakest metric.
- `evaltrust.audit_suite(...)` Python API and `SuiteReport` (with `to_dict()`);
  the CLI auto-detects multi-metric files and renders a per-metric summary.
- **Embed it in your own eval/tests:** `report.raise_if_below(level)` raises an
  `UntrustworthyError` (an `AssertionError`) so it fails a pytest cleanly.
- **Config file:** set a team's thresholds once in `.evaltrust.toml` or
  `[tool.evaltrust]`; `AuditConfig` bundles every threshold and all are now
  configurable. CLI `--config`, plus flag overrides.
- **CI gate:** `--fail-under high|moderate|low` and a bundled GitHub Action.
- **Judge calibration:** when the file has a human/gold judge, measure how well
  each AI judge agrees with it (`--reference-judge`, or auto-detected names).
- **Regression detection:** `evaltrust diff old.json new.json` flags where
  confidence dropped or a real win was lost between two runs.

### Changed

- Ingestion skips rows with missing/unreadable scores instead of crashing, and
  reports the count as a Data Quality finding. A model with no comparable scores
  now errors cleanly.

## [0.3.0] — 2026-07-07

### Added

- `--explain` flag to show why each flag matters and how it was measured.

### Changed

- Redesigned the report to be scannable at a glance: verdict and one line, checks
  grouped by pillar, a short "What to do" list, and optional "To check more". The
  full reasoning for each flag now lives behind `--explain`.
- Tightened all finding text to be terse and plain; `--plain` output is now
  guaranteed ASCII.
- Rewrote the README and docs to explain, plainly, what the tool does and why its
  numbers are trustworthy.

## [0.2.0] — 2026-07-07

### Added

- **Python API**: `evaltrust.audit(source)` accepts a path, two paths, or an
  `EvalData`; `report.to_dict()` is JSON-serializable.
- **CLI output modes**: `--json` for CI and tooling, `--plain` for ASCII-only
  output that is safe on Windows terminals, CI logs, and pipes.
- **Equivalence testing**: the statistical audit can now conclude two models are
  *statistically equivalent* within a configurable `--equivalence-margin`, instead
  of forcing every non-significant result to look like a failure.
- Proportion-appropriate statistics for pass/fail data: **McNemar's exact test**
  and **Cohen's h** / risk difference.
- **DeepEval adapter**: reads DeepEval's test-results export directly.
- Reports name the models that weren't compared when a file has more than two.

### Changed

- The Statistical Validity pillar now reports **decision / effect size /
  precision**. "Not significant" is reported as *inconclusive* or *equivalent*
  rather than a blunt failure.
- Replaced post-hoc (observed-effect) power with the prospective **minimum
  detectable effect**.
- Verdict summaries reflect the actual outcome (equivalent / inconclusive), not a
  generic "improvement is probably real".
- Documented the tool's assumptions and limitations, and made adapter-coverage
  claims precise.

## [0.1.0] — 2026-07-07

Initial release.

### Added

- `evaltrust audit` command: reads an evaluation results file and prints a
  High/Moderate/Low confidence verdict.
- Four audit pillars:
  - **Statistical Validity** — paired permutation test, bootstrap confidence
    interval, Cohen's *d*, and power analysis.
  - **Benchmark Health** — saturation and discrimination checks.
  - **Repeatability** — rerun stability and measurement noise (when the file
    contains repeated runs).
  - **Judge Reliability** — inter-judge consensus, agreement, and outlier
    detection (when the file contains multiple judges).
- Format auto-detection with adapters for Promptfoo, nested JSON, generic record
  lists, and CSV (long and wide).
- Two-file comparison for single-model tools (e.g. two DeepEval runs), paired by
  example id.
- `--strict` flag to fail CI on a Low-Confidence verdict.
- Deterministic, seeded resampling so audits are reproducible.

[Unreleased]: https://github.com/k-dickinson/evaltrust/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/k-dickinson/evaltrust/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/k-dickinson/evaltrust/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/k-dickinson/evaltrust/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/k-dickinson/evaltrust/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/k-dickinson/evaltrust/releases/tag/v0.1.0
