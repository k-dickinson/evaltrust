# Design and philosophy

## The problem

Evaluating models costs real time and money, and the output is usually a pair of
numbers:

```
Model A: 84.7
Model B: 86.2
```

From which teams conclude: ship B. But you can't tell by looking whether that
1.5-point gap is a real improvement or a lucky streak. Think of a coin: 6 heads
out of 10 flips proves nothing, but 600 out of 1,000 does - and the two look
identical on the surface. Score gaps are the same. The difference might be noise.
The sample might be too small. A second judge might disagree. The benchmark might
be saturated, so a gain near the ceiling means little.

Most evaluation tools report *what* the score is. Very few tell you whether you
should *believe* it - so people ship models and publish benchmark numbers on
differences that are really noise.

## The idea

EvalTrust is an **evaluation auditor**, not another eval framework, benchmark, or
judge. The analogy is financial accounting: companies keep their own books, and
audits exist because bookkeeping answers "what are the numbers?" while an audit
answers "can you trust these numbers?" EvalTrust plays the second role for
evaluations.

It runs *after* your existing eval tool rather than replacing it, which makes it
easy to adopt: you keep your current workflow and add one command at the end.

Every feature answers exactly one question:

> Is the evidence from this evaluation strong enough to justify the decision I'm
> about to make?

## Pillars of trust

A trustworthy evaluation is repeatable, statistically sound, robust, and
consistent across evaluators, on a healthy benchmark. EvalTrust's checks map onto
these pillars, computed from the data already in your results file:

- **Statistical Validity** - is the gap real, large enough to matter, and was the
  sample big enough to detect it? (For a single model: is the score itself precise
  enough to trust?)
- **Benchmark Health** - can the benchmark separate these models at all?
- **Repeatability** - would a rerun reach the same conclusion?
- **Judge Reliability** - would a different judge reach the same verdict, and does
  the AI judge agree with human labels?

Two further pillars - robustness to perturbation, and reproducibility provenance -
require generating new evidence (re-running the eval, calling additional judges)
rather than analyzing an existing file, and are planned as opt-in features.

## Principles

**Sit after the eval, not in place of it.** EvalTrust reads what your tool already
produced. Adoption costs one command.

**No arbitrary score.** The output is a plain-language verdict - High, Moderate,
or Low Confidence - backed by specific findings. A single opaque number would just
recreate the problem EvalTrust exists to solve.

**Every finding is actionable.** Each one answers three questions: why it matters,
how we detected it, and how to fix it. A warning you can't act on is noise.

**Missing evidence is a recommendation.** When a check needs data the file doesn't
contain, EvalTrust doesn't guess or crash - it explains how to generate that
evidence. "Add repeated runs" is itself useful advice.

**The auditor is held to its own standard.** Every statistical method is validated
against an independent reference implementation, and all resampling is seeded, so
the audit is reproducible. A tool that demands reproducibility has to be
reproducible itself.

## Scope

Deliberately offline and correct rather than broad and shaky. Today EvalTrust:

- Runs fully offline - no API keys, no network, no account.
- Reads Promptfoo, DeepEval, nested JSON, record lists, and CSV; pairs two
  single-model files, and audits a single model on its own.
- Compares two models, audits one model's score reliability, and audits
  multi-metric suites (with multiple-comparison correction).
- Covers the pillars above, plus judge calibration against human/gold labels.
- Runs as a CLI or a Python API, embeds in tests, gates CI, and compares two runs
  for regressions. Output as a terminal report, plain ASCII, or JSON.

Still out of scope, by design: anything that **calls models or orchestrates new
evaluation runs** - re-running the eval, adding judges, perturbing prompts. Those
generate *new* evidence rather than auditing what's already in the file, and would
break the offline, one-minute promise. They are a deliberate later step.
