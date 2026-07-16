# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Friendly enum string representations.** `Status` and `VerdictLevel` now render their friendly values when converted with `str(...)`.
- **Per-slice / subgroup comparison (`--slice-by`).** The audit can now break
  the two-model comparison down by an optional per-example attribute (category,
  difficulty, language, ...) so a subgroup regression can't hide inside an
  overall improvement. Each slice's significance is tested at a
  Bonferroni-corrected threshold across the slices that were actually tested
  (`alpha / k_tested`); slices below `min_slice_size` are still listed in
  `details.slices` but don't size the family. A slice is flagged as a
  regression only when it is *significantly opposite* to the overall direction,
  so an underpowered slice never trips the flag. `Example` gains an optional
  `attributes: dict[str, str]` field, and the native nested JSON adapter reads
  an `attributes` key per example — CSV and generic record lists don't carry
  attributes yet. Available via `evaltrust audit --slice-by <attr>` and
  `evaltrust.audit(..., slice_by=...)`. Closes #84.
- **lm-eval model names from sibling results files.** When a `results_*.json`
  sits next to a samples log, the adapter uses its top-level `model_name`
  instead of inferring the name from the samples filename.
- **Versioned JSON output.** Every machine-readable payload (`audit --json`,
  `audit_suite`, `diff --json`, and `to_dict()`) now carries a `schema_version`
  (the output shape) and, for audits, a `methodology_version` (the audit methods
  and thresholds that produced the verdict), also exposed as
  `evaltrust.SCHEMA_VERSION` / `evaltrust.METHODOLOGY_VERSION`. Downstream tooling
  can pin to these instead of guessing.
- **Documented exit-code contract.** The CLI's exit codes are now a documented,
  test-locked contract for CI: `0` = ran and met the gate, `1` = ran but the gate
  failed (`--strict` / `--fail-under` / a `diff` regression), `2` = could not run
  (bad input/usage/config). See the README.
- **Line-format adapters.** JSONL ingest can detect specific row formats before
  falling through to the existing generic record path; lm-eval sample logs are
  the first supported format.
- **Pairwise preference judgments:** audit judge-level A/B/tie votes with an exact sign test and seeded win-rate interval.

- **Judge calibration thresholds are independently tunable.** A new
  `judge_correlation_threshold` config key (default `0.8`) sets the Spearman
  rank-correlation floor for continuous judge scores, separate from
  `judge_agreement_threshold` (the fraction-agreed floor for binary judges) — a
  rank correlation of 0.80 and 80% agreement are not the same bar. Both default to
  `0.8`, so output is unchanged until you set them apart.

- **Confidence interval on the effect size.** The effect-size finding now reports
  a bootstrap CI alongside the point estimate — on Cohen's *d* for continuous
  scores, on the risk difference for pass/fail — via a new seeded
  `stats.bootstrap_statistic_ci` primitive. The PASS/WARN rule is unchanged
  (still driven by the effect-size magnitude).

### Fixed
- **Multi-scorer Inspect and OpenEvals logs fan out into a full metric suite.** On
  the suite path (`load_suite` / `audit_suite`), every scorer in an Inspect or
  OpenEvals log now becomes its own metric instead of silently keeping only the
  first and dropping the rest — matching how generic files with a `metric` column
  already behave. The single-audit path is unchanged (still the first scorer). As
  part of this, an Inspect sample with no usable scores is now counted as a
  skipped row, consistent with the OpenEvals adapter. Fixes #48.
- **`_err` now writes to stderr, not stdout (#51).** `Console(stderr=False)`
  caused all CLI error messages (load failures, config errors, unrecognised
  format, non-zero exits) to pollute stdout — corrupting `--json` output and
  hiding failures from anything scraping stderr. Fixed to `Console(stderr=True)`.
  Three CLI tests updated to assert on `result.stderr` instead of `result.stdout`.

- **Resampling no longer exhausts memory on large evaluations.** `bootstrap_ci`
  and `permutation_test` built an `(n_resamples, n)` matrix, which OOMs on large
  inputs (e.g. n = 1M at 10k resamples needed ~160 GB). Resamples are now drawn in
  memory-bounded blocks, capping peak memory regardless of n. Because each block
  is drawn from the same generator, output is byte-identical to before at every
  size (verified against the previous results and scipy). Fixes #79.
- **Judge consensus no longer nan-compares when a judge scored only one model.**
  `_consensus()` called `np.mean([])` → `nan` when a judge had results for only
  one of the two models; `nan >= nan` is `False`, silently defaulting the winner
  to `model_a`. Judges missing scores for either model are now skipped and listed
  in the finding's `how_detected` and `details.skipped_judges`. When all judges
  are skipped the finding returns `Status.SKIP` instead of a spurious disagree.
  When only one judge survives the skip the finding also returns `Status.SKIP`
  (one judge is not consensus). Fixes #53.

- **Saturation check no longer false-warns on continuous or rubric scales.**
  `_saturation()` previously divided the top model mean by the highest *observed*
  score, so a top mean of 4.0 on a 0–5 rubric (observed max 4.2) read as 95% of
  ceiling and triggered a spurious WARN. A new `score_ceiling` config key (default
  `None`) lets teams declare the true upper bound; when set, it is used as the
  denominator instead of the observed max. The default path is unchanged.


- **Config typos are no longer silently ignored.** An unknown key in the config
  (`alpah`, or `equivalence-margin` with a dash) previously reverted the
  intended setting to its default with no signal. An explicit `--config` file
  now fails with the unknown key named and a did-you-mean suggestion; a
  discovered `.evaltrust.toml` / `[tool.evaltrust]` warns and ignores it.

- **Two-file pairing no longer hides dropped data.** Pairing two single-model
  files now carries both files' `skipped_rows` counts forward and counts every
  example that appears in only one file (or lacks a score) as
  `unmatched_examples`; the audit reports them as a Data Quality finding
  ("N examples dropped during pairing") instead of silently auditing the
  overlap.

### Added

- **Generic adapter aliases:** recognize common real-world column names like
  `sample_id`, `item_id`, `system_name`, `hypothesis`, `accuracy`,
  `pass_rate`, `task`, and `skill`; parse percent-style scores such as `85%`;
  accept additional unambiguous pass/fail words (`accept`/`reject`,
  `positive`/`negative`, `good`/`bad`, `won`/`lost`); and list available model
  names when explicit comparison models do not match the data.

- **LangSmith adapter:** read a LangSmith run export directly — one
  experiment/model per file, scored from each run's `feedback_stats.<key>.avg`
  (averaged across metrics). One model per run; compare two runs.

- **Ragas adapter:** read a Ragas result export (`EvaluationResult.to_pandas()`)
  directly, scored from its per-example metric columns (`faithfulness`,
  `answer_relevancy`, `context_precision`, ...) averaged per row. One RAG
  pipeline per run; compare two runs. Metrics are currently averaged into a
  single score rather than split into a per-metric suite — see
  [#9](https://github.com/k-dickinson/evaltrust/issues/9) for the follow-up.

## [0.6.0] — 2026-07-09

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
- **OpenEvals adapter:** read a langchain-ai/openevals results list directly. One
  model per run; compare two runs.
- **HTML report** via `--html <path>`: a self-contained, dependency-free page.
- **Markdown report** via `--md`, for PR comments and docs.
- `win` and `loss` are recognised as pass/fail score words.

### Fixed

- **Binary effect size is now computed on the paired sample.** The pass-rate
  effect size (risk difference and Cohen's *h*) for pass/fail data was measured
  over every example each model scored, while the significance test (McNemar) and
  the confidence interval used only the paired examples. On data where the two
  models scored different example sets the effect size — and the PASS/WARN
  verdict it drives — could disagree with the significance test; both now use the
  same paired sample.
- **`--json` output is always valid JSON.** Non-finite floats (an infinite
  Cohen's d from a zero-variance gap, or an infinite signal-to-noise ratio) were
  written as the non-standard `Infinity`/`NaN` tokens that strict parsers reject.
  They now serialize as `null`.
- **OpenEvals adapter is robust to bad data.** A row with a missing or unreadable
  score is skipped and counted (reported as `skipped_rows`) instead of aborting
  the whole file, and the example id no longer defaults to the free-text `input`,
  which could merge two distinct evaluations into one.
- **`--html` with a multi-metric suite no longer corrupts `--json` output.** The
  "not supported for suites" warning now goes to stderr, so stdout stays valid
  JSON.

### Changed

- **Judge calibration** now handles continuous judge scores (e.g. a 1–5 rubric):
  it switches from exact-match agreement to a Spearman rank correlation against
  the reference judge, and names the metric in the finding so a correlation is
  never mistaken for an agreement rate. Binary pass/fail judges are unchanged.
- **Holm-Bonferroni now carries its rejection decision** into each metric's audit
  instead of reconstructing an effective alpha to re-derive it under a strict
  `p < alpha`; the reported per-metric `alpha` is the exact Holm step threshold,
  with no ULP nudge on the exact-tie boundary. Verdicts are unchanged. On that
  boundary a metric can now be reported significant with `p == alpha` exactly
  (Holm rejects via `adjusted_p <= alpha`), so the significant decision prose picks
  its operator to match reality — `<`, `<=`, or `>` — instead of the previously
  hard-coded (and, without the ULP nudge, now false at the boundary) `p < alpha`.
  Every other correction path still reads `<` and its prose is byte-for-byte
  unchanged. The per-metric `alpha` in `SuiteReport.metric_alphas` is a plain
  `float` on both the Holm and Bonferroni paths.

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

[Unreleased]: https://github.com/k-dickinson/evaltrust/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/k-dickinson/evaltrust/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/k-dickinson/evaltrust/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/k-dickinson/evaltrust/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/k-dickinson/evaltrust/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/k-dickinson/evaltrust/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/k-dickinson/evaltrust/releases/tag/v0.1.0
