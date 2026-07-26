<h1 align="center">EvalTrust</h1>

<p align="center">
  <strong>An auditor for LLM evaluations.</strong><br>
  It doesn't tell you how good your model is - it tells you whether you can trust the evaluation you used to decide.
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

## The problem, in ten seconds

You spent a week evaluating two models. One scored 84.7%, the other 86.2%. Ship
the better one, right?

Maybe. Or maybe that 1.5-point gap is pure luck and the two models are actually
the same. **You can't tell by looking, and neither can anyone else.** People ship
models, publish benchmark numbers, and kill good ideas every day based on gaps
that are just noise.

EvalTrust does the math that tells you. Point it at your eval results and it says,
in plain English, whether the difference is **real**, whether it's **big enough to
matter**, and whether you had **enough data to know** - then gives you one verdict:
**High, Moderate, or Low confidence.** You keep using whatever eval tool you
already have. Think of it as a code reviewer for your eval's conclusion.

## Why you can't just eyeball it

Think about a coin. Flip it 10 times, get 6 heads. Rigged? No - that happens by
luck constantly. Now flip it 1,000 times and get 600 heads. *That's* real. Six
out of ten and six hundred out of a thousand look the same on the surface, but
only one is signal.

"84.7 vs 86.2" is the exact same trap. EvalTrust computes how likely that gap is
to be a lucky streak. If it's likely luck, it tells you straight: don't ship on
this yet.

## Example

```console
$ evaltrust audit gpt4_run.json claude_run.json
```

```
EvalTrust  claude-3 vs gpt-4 · 150 examples · deepeval

● Low Confidence
Not enough data to call a winner. Collect more before deciding.

Statistical Validity
  ✗ Improvement of claude-3 over gpt-4 is inconclusive
  ! Effect size is negligible
  ! Sample size may be too small
Benchmark Health
  ✓ Benchmark has headroom
  ✓ Benchmark discriminates between examples

What to do
  • Don't call a winner yet. Collect more examples first.
  • Collect ~90 more examples (~240 total) to catch a small effect.
```

Two runs at 71% and 74% - a three-point "win" that is actually noise. EvalTrust
catches it before it becomes a shipping decision. (Add `--explain` for the exact
p-values and reasoning behind each line.)

## What EvalTrust can do

One tool, whether you're comparing models, sanity-checking a single eval, or
gating CI:

- **Compare two models** - is B really better than A, or is that gap just noise?
  Significance, effect size, equivalence ("they're actually the same"), and
  whether your sample was even big enough to tell.
- **Audit a single model** - no comparison needed. It puts a confidence interval
  on your score (is 84% really `[80%, 88%]` or `[71%, 97%]`?) and, given a target,
  tests whether the model *actually* clears the bar.
- **Audit a whole metric suite** - many metrics per example at once, with
  multiple-comparison correction so testing ten metrics doesn't hand you a fake win.
- **Vet the judge** - do multiple LLM judges agree, and does the AI judge match
  human labels? The judge is where most eval trust quietly breaks.
- **Vet the benchmark** - is it saturated, or too flat to separate anything?
- **Catch regressions** - `evaltrust diff old.json new.json` flags where a metric
  got worse since last release.
- **Fit your workflow** - a CLI, a Python API, a pytest one-liner, a GitHub
  Action, a `.evaltrust.toml` for team policy, and JSON output for everything else.

## Why the numbers are real, not our opinion

Fair question for a tool that judges trust. Three things keep it honest:

- **It runs standard, decades-old statistics** - the same significance tests,
  confidence intervals, and power analysis scientists already use. Nothing invented.
- **The math is proven correct.** Every calculation is checked in the test suite
  against the libraries researchers trust (`scipy`, `statsmodels`) and must produce
  the same numbers. So it's not "trust us" - it's "our math matches the reference
  everyone already trusts."
- **It's reproducible.** Same input, same answer, every time. An opinion drifts; a
  calculation doesn't.

One honest limit: EvalTrust checks whether your *numbers* support your conclusion.
If the eval measured the wrong thing to begin with, it catches some of that
(saturated benchmarks, judges that disagree) but not all of it. It's the
statistician auditing your results, not a replacement for a well-designed eval.

## Installation

```bash
pip install evaltrust
```

Requires Python 3.10 or newer. That's the whole setup - no API keys, no config,
no account. It runs fully offline: **no network calls, no telemetry, no data
leaves your machine** (see the [security policy](SECURITY.md)).

<details>
<summary>Install from source (for development)</summary>

```bash
git clone https://github.com/k-dickinson/evaltrust
cd evaltrust
pip install -e ".[dev]"
pytest
```

The same common development tasks are available as `make install`, `make test`,
and `make audit`.

</details>

## Quick start

1. Run your evaluation with whatever tool you already use (Promptfoo, DeepEval, or
   anything that can export scores to CSV/JSON) and save the output.
2. Point EvalTrust at it:

   ```bash
   # A file that already compares two or more models (e.g. Promptfoo):
   evaltrust audit results.json

   # Two single-model runs (e.g. two DeepEval runs), paired by example id:
   evaltrust audit gpt4_run.json claude_run.json

   # Just one model? Audit whether you can trust its score, vs a target:
   evaltrust audit my_model.json --threshold 0.8
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
| `--json` | Emit the full audit as JSON, for CI logic and experiment trackers. |
| `--md` | Emit the audit as Markdown, for PR comments and docs (sample below). |
| `--html <path>` | Write a self-contained HTML report (no dependencies) to a file. |
| `--plain` | Plain ASCII output - safe for Windows terminals, CI logs, and piping to a file. |
| `--explain` | Show why each flag matters and the numbers behind it. |
| `--slice-by` | Break the comparison down by a per-example attribute (category, difficulty, language) and flag any subgroup that regresses. |
| `--all-pairs` | With more than two models, compare every pair (not just the top two), corrected across the pairs. |
| `--bayesian` | Add a Bayesian view: the probability one model wins more often, with a credible interval. |
| `--correction` | Multi-metric correction: `bonferroni` (default), `holm`, or `none`. |
| `--fail-under` | Exit non-zero if confidence is below a level (`high`/`moderate`/`low`) - gate CI. |
| `--threshold` | For a single-model eval, the target score to test against (e.g. `0.8`). |
| `--reference-judge` | Name the human/gold judge to calibrate the AI judges against. |
| `--config` | Path to a config TOML (defaults to `.evaltrust.toml` / `pyproject`). |
| `--model-a`, `--model-b` | Choose which two models to compare, or label the two files. |
| `--alpha`, `--equivalence-margin`, `--seed` | Statistical knobs (all also settable in config). |

Two saved audits can be compared for regressions with `evaltrust diff old.json new.json`.

<details>
<summary>Sample <code>--md</code> output (paste into a PR comment)</summary>

`evaltrust audit examples/clean_win.json --md` produces:

```markdown
# EvalTrust

**B vs A · 200 examples · native**

## High Confidence

The result holds up. You can act on it.

### Statistical Validity

- **[pass]** B is significantly better than A
- **[pass]** Effect size is large
- **[pass]** Sample size was sufficient

### Benchmark Health

- **[pass]** Benchmark has headroom
- **[pass]** Benchmark discriminates between examples

### Repeatability

- **[skip]** Not assessed
```

Add `--explain` to include the "why it matters" and the numbers behind each line.

</details>

## Use it standalone, or embed it in your eval

Run the CLI by hand, **or** drop the audit into your own eval/test code and fail
when the result isn't trustworthy - one line does it:

```python
import evaltrust

report = evaltrust.audit("results.json")      # path, two paths, or an EvalData
report.raise_if_below("moderate")             # raises UntrustworthyError if too low

report.to_dict()          # machine-readable JSON: log it, store it, diff it
```

Every JSON payload (`--json`, `to_dict()`, `diff --json`) carries a
`schema_version` (the output shape) and a `methodology_version` (the audit methods
and thresholds behind the verdict), also exposed as `evaltrust.SCHEMA_VERSION` /
`evaltrust.METHODOLOGY_VERSION` - pin to these instead of guessing.

In pytest, that makes "is my eval trustworthy?" a normal test:

```python
def test_new_prompt_is_a_real_improvement():
    evaltrust.audit(["old_prompt.json", "new_prompt.json"]).raise_if_below("moderate")
```

## Gate CI on it

Use the bundled GitHub Action:

```yaml
# .github/workflows/eval.yml
- uses: k-dickinson/evaltrust@v1
  with:
    results: results.json
    min-confidence: moderate     # fail the job below this level
```

...or just call the CLI with `--fail-under` (`high`, `moderate`, or `low`):

```bash
pip install evaltrust
evaltrust audit results.json --plain --fail-under moderate
```

### Exit codes

The CLI uses a stable exit-code contract you can gate on:

| Code | Meaning |
|------|---------|
| `0` | The audit ran and the verdict met the gate (or no gate was set). |
| `1` | The audit ran, but the gate failed: `--strict`, or `--fail-under` not met, or `diff` found a regression. This is the "block the build" code. |
| `2` | The audit could not run: bad usage, a missing or unreadable file, an unrecognised format, or an invalid config. This is an error, not a verdict. |

So `1` means "the evaluation isn't trustworthy enough," while `2` means "something is wrong with the inputs" — CI can tell the two apart.

More patterns in [`docs/integrations.md`](docs/integrations.md).

## What it checks

EvalTrust audits several pillars of trust and ends in one plain-language verdict -
**High**, **Moderate**, or **Low Confidence**. There is no arbitrary aggregate
score.

| Pillar | The question it answers |
|--------|-------------------------|
| **Statistical Validity** | Is the difference a real improvement, no real difference, or just too little data to tell? Significance (McNemar for pass/fail, paired permutation for continuous), equivalence testing, an effect size with a confidence interval, and the minimum detectable effect. *(For a single model: a confidence interval on the score, and an optional target test.)* |
| **Pairwise Preference** | When judges vote A-vs-B (or tie) instead of scoring each model, is the preference real? An exact sign test on the win/loss split, with a win-rate interval. |
| **Per-slice Comparison** | *(opt-in `--slice-by`)* Does the result hold within each subgroup (category, difficulty, language), or does a slice quietly regress? Corrected across the slices. |
| **All-pairs Comparison** | *(opt-in `--all-pairs`)* With more than two models, which pairs are actually distinguishable - corrected across the whole set - and is the ranking stable? |
| **Benchmark Health** | Can the benchmark even separate these models, or is it saturated / flat? |
| **Repeatability** | If you reran the evaluation, would the winner stay the winner? Uses repeated-run data when the file contains it. |
| **Judge Reliability** | Would a different judge reach the same verdict - and does the AI judge match human labels? Works for a comparison or a single model, using multi-judge and human/gold data when the file contains it. |

Correlated examples (repeated judgments, task/template groups)? Add a `group_id`
per example and the significance test and intervals resample whole clusters, so
they reflect that correlation instead of assuming independence. Add `--bayesian`
for a Bayesian view (the probability one model wins more often, with a credible
interval); it's advisory and never changes the verdict.

Every finding follows the same rule - **why it matters**, **how we detected it**,
and **how to fix it**. Checks that need extra data (repeated runs, multiple
judges) don't guess when it's missing; they tell you how to generate it.

Scoring several metrics per example (correctness, safety, tone...)? Add a `metric`
column and EvalTrust audits each one, corrects the significance threshold for the
number of metrics (so you don't get false wins by testing many), and reports the
suite's confidence as its weakest metric.

**Only evaluated one model?** Point it at a single model's scores and EvalTrust
switches to asking *can I trust this number?* - it puts a confidence interval
around your score (is 84% really `[80%, 88%]` or `[71%, 97%]`?) and, with
`--threshold 0.8`, tests whether the model actually clears your bar.

See [`docs/checks.md`](docs/checks.md) for the methods and thresholds behind each
one.

## Supported inputs

You never write an EvalTrust-specific format. It reads what your tool already
produced and auto-detects the shape. First-class adapters today:

- **Promptfoo** results (several providers compared across test cases)
- **DeepEval** test-results export (one model per run - pass two files to compare)
- **Inspect** (UK AISI) `.json` eval log (one model per log - pass two to compare)
- **OpenEvals** results list (one model per run - pass two files to compare)
- **Langfuse** Scores API export (one model per export - pass two files to compare)
- **LangSmith** run export (one experiment per run - pass two files to compare)
- **Ragas** result export (one RAG pipeline per run - pass two files to compare)
- **OpenAI Evals** (`openai/evals`) `.jsonl` log (one model per run - pass two files to compare)
- **lm-eval** (`lm-evaluation-harness`) sample logs (`.jsonl`)
- **Nested JSON** - `{"models": [...], "examples": [{"id", "scores": {...}}]}`
- **Record lists** - JSON like `[{"id", "model", "score"}, ...]`
- **CSV** - long (`id,model,score`) or wide (`id,gpt,claude`)

Tools without a dedicated adapter yet work by exporting to CSV or a record
list - usually a one-liner. More native adapters are a top roadmap item;
[contributing one](docs/adapters.md) is straightforward.
Details and single-model pairing in [`docs/input-formats.md`](docs/input-formats.md).

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

Where it's headed (individual tasks are tracked in
[issues](https://github.com/k-dickinson/evaltrust/issues)):

- **Next:** native adapters for hosted platforms (LangSmith, Braintrust, ...), an
  optional HTML report, and richer history/trend tracking.
- **Later:** opt-in orchestration for the pillars that need to *generate* evidence
  (robustness perturbations, extra judges) and a provenance/reproducibility check.

## Contributing

Contributions are welcome - new format adapters and additional checks especially.
New here? Browse the
[**good first issues**](https://github.com/k-dickinson/evaltrust/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22),
then read [`CONTRIBUTING.md`](CONTRIBUTING.md). All participants are expected to
follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report security issues per the
[security policy](SECURITY.md).

## License

EvalTrust is released under the [MIT License](LICENSE) - a permissive,
OSI-approved license. Anyone, including companies and organizations, may use,
modify, and distribute it, in commercial or proprietary settings, free of charge.
There is no copyleft obligation and no contributor license agreement to sign.
