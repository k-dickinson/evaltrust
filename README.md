<h1 align="center">EvalTrust</h1>

<p align="center">
  <strong>An auditor for LLM evaluations.</strong><br>
  It doesn't tell you how good your model is — it tells you whether you can trust the evaluation you used to decide.
</p>

<p align="center">
  <a href="#installation">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#what-it-checks">What it checks</a> ·
  <a href="docs/">Docs</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/evaltrust/"><img alt="PyPI" src="https://img.shields.io/pypi/v/evaltrust.svg"></a>
  <a href="https://pypi.org/project/evaltrust/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/evaltrust.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/pypi/l/evaltrust.svg"></a>
  <a href="https://github.com/k-dickinson/evaltrust/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/k-dickinson/evaltrust/actions/workflows/ci.yml/badge.svg"></a>
</p>

---

Teams spend real money evaluating models, then look at two numbers:

```
Model A: 84.7
Model B: 86.2   ->  ship B
```

That single comparison hides a dozen assumptions. Maybe the difference isn't
statistically significant. Maybe the sample is too small. Maybe another judge
disagrees, or the benchmark is already saturated. Most eval tools tell you *what*
your score is. **EvalTrust tells you whether you should believe it.**

It works like a financial audit: bookkeeping answers "what are the numbers?"; an
audit answers "can you trust them?" EvalTrust is the audit for evaluations. It runs
*after* your existing eval tool — it doesn't replace it.

## Example

```console
$ evaltrust audit gpt4_run.json claude_run.json
```

```
EvalTrust Audit
Comparing claude-3 vs gpt-4  · 150 examples · source: deepeval+deepeval
╭─ Verdict ────────────────────────────────────────────────────────────────────╮
│ Low Confidence                                                               │
│ The evidence does not support the conclusion. Do not ship on this result     │
│ as-is — resolve the issues below first.                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
 ✗ Improvement is not statistically significant   Statistical Validity
 ⚠ 95% confidence interval overlaps zero          Statistical Validity
 ⚠ Effect size is negligible                      Statistical Validity
 ✓ Benchmark has headroom                         Benchmark Health

  ✗ Improvement is not statistically significant
    Why it matters   A raw gap means nothing until you rule out chance.
    How we detected  A paired permutation test over 150 examples gave p = 0.41.
    How to fix       Do not claim a winner yet. Collect more examples first.
```

Two runs at 71% and 74% — a three-point "win" that is actually noise. EvalTrust
catches it before it becomes a shipping decision.

## Installation

```bash
pip install evaltrust
```

Requires Python 3.10 or newer. That's the whole setup — no API keys, no config,
no account.

<details>
<summary>Install from source (for development)</summary>

```bash
git clone https://github.com/k-dickinson/evaltrust
cd evaltrust
pip install -e ".[dev]"
pytest
```

</details>

## Quick start

1. Run your evaluation with whatever tool you already use (DeepEval, Promptfoo,
   LangSmith, OpenEvals, or a plain CSV) and save the output.
2. Point EvalTrust at it:

   ```bash
   # A file that already compares two or more models (e.g. Promptfoo):
   evaltrust audit results.json

   # Two single-model runs (e.g. two DeepEval runs), paired by example id:
   evaltrust audit gpt4_run.json claude_run.json
   ```

3. Read the verdict. Fix what it flags. Re-run.

Want to see it work before pointing it at your own data? The repo ships sample
files:

```bash
evaltrust audit examples/clean_win.json        # -> High Confidence
evaltrust audit examples/borderline.json       # -> Moderate Confidence
evaltrust audit examples/deepeval_gpt4.json examples/deepeval_claude.json
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--strict` | Exit with a non-zero status on a Low-Confidence verdict (use it to gate CI). |
| `--model-a`, `--model-b` | Choose which two models to compare, or label the two files. |
| `--alpha` | Significance level (default `0.05`). |
| `--seed` | Seed for the resampling (results are deterministic; change only to stress-test). |

## What it checks

EvalTrust audits four pillars of trust and ends in one plain-language verdict —
**High**, **Moderate**, or **Low Confidence**. There is no arbitrary aggregate
score.

| Pillar | The question it answers |
|--------|-------------------------|
| **Statistical Validity** | Is the gap real, large enough to matter, and was the sample big enough to detect it? Paired permutation test, bootstrap confidence interval, Cohen's *d*, and power analysis. |
| **Benchmark Health** | Can the benchmark even separate these models, or is it saturated / flat? |
| **Repeatability** | If you reran the evaluation, would the winner stay the winner? Uses repeated-run data when the file contains it. |
| **Judge Reliability** | Would a different judge reach the same verdict? Uses multi-judge data when the file contains it. |

Every finding follows the same rule — **why it matters**, **how we detected it**,
and **how to fix it**. Checks that need extra data (repeated runs, multiple
judges) don't guess when it's missing; they tell you how to generate it.

See [`docs/checks.md`](docs/checks.md) for the methods and thresholds behind each
one.

## Supported inputs

You never write an EvalTrust-specific format. It reads what your tool already
produced and auto-detects the shape:

- **Promptfoo** results (several providers compared across test cases)
- **Nested JSON** — `{"models": [...], "examples": [{"id", "scores": {...}}]}`
- **Record lists** — JSON like `[{"id", "model", "score"}, ...]`
- **CSV** — long (`id,model,score`) or wide (`id,gpt,claude`)

Single-model tools (DeepEval, LangSmith, OpenEvals) evaluate one model per run,
so you pass two files and EvalTrust pairs them. Details in
[`docs/input-formats.md`](docs/input-formats.md).

## How it works

```
your eval output ──▶ auto-detect + adapter ──▶ canonical model ──▶ audit ──▶ verdict
```

Adapters map every format into one internal representation, so the statistics are
written once and work everywhere. Every statistical method is validated in the
test suite against an independent reference (`scipy` and `statsmodels`), and all
resampling is seeded, so the auditor is itself reproducible. See
[`docs/architecture.md`](docs/architecture.md).

## Roadmap

- **Now:** offline CLI, four pillars, terminal report.
- **Next:** dedicated adapters for more tools, a Python API (`evaltrust.audit(...)`),
  and an optional HTML report.
- **Later:** opt-in orchestration for the pillars that need to generate evidence
  (robustness perturbations, extra judges) and a provenance/reproducibility check.

## Contributing

Contributions are welcome — new format adapters and additional checks especially.
Start with [`CONTRIBUTING.md`](CONTRIBUTING.md), and see [`RELEASING.md`](RELEASING.md)
if you maintain releases. All participants are expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md). Report security issues per the
[security policy](SECURITY.md).

## License

EvalTrust is released under the [MIT License](LICENSE) — a permissive,
OSI-approved license. Anyone, including companies and organizations, may use,
modify, and distribute it, in commercial or proprietary settings, free of charge.
There is no copyleft obligation and no contributor license agreement to sign.
