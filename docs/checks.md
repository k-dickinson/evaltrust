# Checks and methods

EvalTrust groups its checks into a handful of pillars. Each check returns a finding with a
status - **PASS**, **WARN**, **FAIL**, or **SKIP** - and the reasoning behind it.
This page documents the method and thresholds for each.

All comparisons are **paired**: the same examples are scored by both models, so
EvalTrust works with per-example differences rather than comparing two noisy
averages.

## Statistical Validity

The core question: is the reported gap real evidence you can act on? Three
findings.

### Decision - significant, equivalent, or inconclusive

This deliberately avoids the trap of treating "not significant" as a failure
(that would confuse *absence of evidence* with *evidence of absence*). It returns
one of three honest outcomes:

- **Significant** (**PASS**) - the leader really is ahead. Detected with
  **McNemar's exact test** for paired pass/fail data, or a **paired permutation
  test** (sign-flip, two-sided, `(count + 1) / (N + 1)` correction, no normality
  assumption) for continuous scores. Triggered when `p < alpha` (default `0.05`).
- **Equivalent** (**WARN**) - a genuine conclusion that the models are the *same*
  within a margin you set. Established by a two-one-sided-tests (TOST) style check:
  the `(1 − 2·alpha)` bootstrap interval for the gap lies entirely within
  ±`equivalence_margin`. This is what lets you answer "is my cheaper model as good
  as the expensive one?".
- **Inconclusive** (**FAIL**) - not significant *and* not equivalent: there simply
  isn't enough evidence to decide. The fix is more data, not a different claim.

A bootstrap confidence interval for the gap is reported alongside all three.

### Effect size - how big, in interpretable terms

- Continuous scores: **Cohen's *d*** on the paired differences, with a magnitude
  label (negligible `< 0.2`, small `< 0.5`, medium `< 0.8`, large `≥ 0.8`).
- Pass/fail scores: the **risk difference** in percentage points plus **Cohen's
  *h***, the effect size appropriate for proportions (Cohen's *d* assumes roughly
  continuous data and is not used for 0/1 outcomes).

**PASS** when the effect is medium or large; **WARN** when small or negligible -
because a real gap can still be too small to matter in production.

### Precision - minimum detectable effect (not post-hoc power)

Rather than the widely-criticised *observed-effect* (post-hoc) power, EvalTrust
reports the **minimum detectable effect**: the smallest true effect this sample
size could reliably detect at 80% power, computed from the exact noncentral-*t*
distribution. This is a property of the design, not of the observed result.

- **PASS** when the comparison reached a conclusion (significant or equivalent) -
  the sample was adequate.
- **WARN** when inconclusive, with a prospective recommendation for how many
  examples would be needed to detect even a small effect.

## Benchmark Health

Even a flawless comparison is worthless on a broken benchmark.

### Saturation

If the strongest model already averages within 95% of the benchmark's ceiling,
there is little headroom left to demonstrate an improvement.

- **PASS** when there is headroom.
- **WARN** when the benchmark is saturated.

### Discrimination

If the pooled standard deviation of scores is below `0.01`, the benchmark assigns
nearly the same score to everything and cannot separate any two models.

- **PASS** when scores show a healthy spread.
- **WARN** when there is almost no variation.

## Repeatability

*Requires repeated-run data (`runs`). If absent, the pillar returns a single SKIP
explaining how to add it.*

### Rerun stability

For each rerun, EvalTrust computes the mean gap between the models and checks whether
the winner is consistent.

- **PASS** when the winner never changes across reruns.
- **WARN** when it changes on a minority of reruns.
- **FAIL** when it changes on half or more (the winner is effectively a coin flip).

### Measurement noise

Compares the run-to-run standard deviation of the gap against the gap itself.

- **PASS** when the gap is stable relative to its noise.
- **WARN** when the noise is as large as the gap.

## Judge Reliability

*Requires multi-judge data (`judges`). If absent, the pillar returns a single SKIP
explaining how to add it.*

### Consensus on the winner

Each judge's preferred model is computed from its mean scores.

- **PASS** when every judge prefers the same model.
- **FAIL** when judges disagree on the winner (the ranking depends on the judge).

### Inter-judge agreement

Mean pairwise agreement across judges, plus Fleiss' kappa when the scores are
categorical, and the identity of the judge that agrees least with the rest.

- **PASS** when mean agreement is at least 80%.
- **WARN** otherwise, naming the likely outlier judge.

### Judge calibration (vs. a human/gold judge)

If the file includes a human or gold judge - treated as ground truth - EvalTrust
measures how often each AI judge agrees with it. A judge that matches humans only
70% of the time can't be trusted to rank models on its own. Include your human
labels as a judge named `gold`/`human`/`reference` (etc.), or name the reference
with `--reference-judge`.

Binary calibration uses `judge_agreement_threshold` (a fraction agreed, default
`0.8`). Continuous calibration uses `judge_correlation_threshold` (a Spearman rho,
default `0.8`). The two floors are configured independently.

- **PASS** when every AI judge clears the threshold for its score scale.
- **WARN** otherwise, naming the worst-calibrated judge and its score.

## Single model (Score Reliability)

If your file has only one model - you evaluated one system, not a comparison -
EvalTrust switches from "is B better than A?" to "can I trust this score?":

- **Precision.** A bootstrap confidence interval on the mean turns "84%" into
  "84%, 95% CI [80%, 88%]". A wide interval means too few examples to trust the
  number. **PASS** when the interval's half-width is within the target margin
  (`precision_margin`, default 5 points); **WARN** otherwise, with how many more
  examples would tighten it.
- **Threshold** (optional). Pass `--threshold 0.8` and EvalTrust tests whether the
  model really clears the bar: **PASS** if the interval is entirely above it,
  **FAIL** if entirely below, **WARN** if it straddles (too close to call).

Benchmark Health still runs. A threshold also forces single-model mode even when
the file has two models (`evaltrust audit results.json --threshold 0.8`).

## Multiple metrics (suites)

If your file scores several metrics per example (a `metric` column, see
[input formats](input-formats.md)), EvalTrust audits each metric separately with
everything above, comparing the same two models throughout, and rolls the results
into one suite report.

Testing many metrics at once inflates false positives - run 20 metrics at
`alpha = 0.05` and one will look "significant" by luck. EvalTrust corrects for this
with **Bonferroni**: it divides the significance threshold by the number of
metrics, so a metric is only called a real improvement if it clears the stricter
bar. The suite's overall confidence is the **weakest** of its metrics.

## Pairwise preference

When your file records judge votes (A wins / B wins / tie) instead of a numeric
score per model, EvalTrust runs an exact two-sided **sign test** on the decisive
votes (ties excluded) and reports a seeded win-rate interval. This answers "is the
preference real, or a coin toss?" See [input formats](input-formats.md) for the
`winner`/`preference` shape.

## Per-slice comparison (`--slice-by`)

An overall improvement can hide a regression on an important subset. With
`--slice-by <attribute>`, EvalTrust breaks the comparison down by a per-example
attribute (category, difficulty, language) and tests each slice at a
Bonferroni-corrected threshold across the slices actually tested (`alpha / k`).
A slice is flagged as a **regression** only when it is *significantly opposite* to
the overall direction, so an underpowered slice never trips the flag (it is still
listed in `details.slices`). Slice tags come from the nested-JSON `attributes`
field; see [input formats](input-formats.md).

## All-pairs comparison (`--all-pairs`)

By default a file with several models compares the two strongest. `--all-pairs`
tests **every** model pair with shared scores and corrects across the pair family
(`bonferroni` / `holm` / `none`), reporting which distinctions are statistically
separable. With three or more models it also reports an advisory **rank-stability**
finding: which leaderboard positions hold under resampling. This finding is
advisory and never changes the verdict.

## Bayesian view (`--bayesian`)

`--bayesian` adds an advisory finding: the posterior probability that one model
wins more often on decisive examples, with a credible interval, under a Jeffreys
`Beta(0.5, 0.5)` prior. It is off by default, never changes the verdict, and
answers "how confident am I that A beats B?" as a probability rather than a
p-value.

## The verdict

The overall verdict follows simple, documented rules rather than a weighted score:

- **Low Confidence** - any check FAILs (a load-bearing part of the conclusion is
  unsupported).
- **Moderate Confidence** - no failures, but at least one WARN.
- **High Confidence** - every applicable check passes.

SKIP findings never raise confidence; they represent evidence you don't yet have.

## Assumptions and limitations

We would rather you trust EvalTrust for the right reasons than oversell it. The
current release assumes:

- **Paired data.** Both models are scored on the *same* examples, matched by id.
  Unpaired comparisons (different test sets) are out of scope.
- **Default comparison is the top two models.** A single model is supported too
  (Score Reliability, above). Multi-metric suites *are* supported (a `metric`
  column, family-corrected). When a file has more than two models, the default
  still compares the two strongest by mean. Opt into `--all-pairs` to test every
  declared pair with one shared correction, and to run the advisory rank-stability
  check on three or more scored models.
- **Rank stability is advisory, not calibrated confidence.** Under `--all-pairs`,
  the tool bootstraps example rows (or whole `group_id` clusters when present),
  recomputes the mean-score ranking each time, and reports rank occupancy,
  per-position retention, top-1 retention, and full-order retention. A position
  is stable when its observed model holds it in at least `1 - alpha` of
  resamples. Exact mean ties split occupancy evenly, so model-name order is never
  counted as stability. Near ties are unstable by design. Model-dependent
  missingness can make stability reflect coverage as well as quality. Repeated
  runs (`Example.runs`) are not consumed. Two-model and multi-metric rank
  stability are out of scope.
- **Scalar scores.** Each score is a number. Pairwise-preference judgments
  (A-beats-B votes) are not yet modelled.
- **Opinionated thresholds.** `alpha` and the equivalence margin are configurable;
  the effect-size, saturation, spread, and agreement cutoffs use conventional
  defaults that may not fit every domain.
- **Monte-Carlo methods are seeded.** Results are reproducible for a given seed,
  but a p-value or interval sitting exactly on a threshold can move slightly if
  you change the seed.

These are the honest edges of the tool. Several are on the roadmap; none are
hidden.
