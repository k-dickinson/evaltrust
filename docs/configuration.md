# Configuration

EvalTrust ships with sensible defaults, but a team can set its own statistical
policy once and have it applied everywhere the audit runs - locally, in CI, and
in the Python API.

## Where config comes from

In order of precedence (highest first):

1. **Command-line flags** - e.g. `--alpha`, `--equivalence-margin`, `--seed`.
2. **An explicit file** - `evaltrust audit results.json --config policy.toml`.
3. **`.evaltrust.toml`** in the current directory.
4. **`[tool.evaltrust]`** in `pyproject.toml`.
5. **Built-in defaults**.

## Example `.evaltrust.toml`

Drop this in your repo and every audit uses it:

```toml
alpha = 0.01                     # stricter significance than the 0.05 default
equivalence_margin = 0.02        # what counts as "no real difference", in score units
judge_agreement_threshold = 0.9  # require 90% inter-judge agreement to pass
score_ceiling = 5.0              # true max of the rubric (omit for 0/1 benchmarks)
```

Or the same under `pyproject.toml`:

```toml
[tool.evaltrust]
alpha = 0.01
saturation_fraction = 0.9
```

## All settings

| Key | Default | What it controls |
|-----|---------|------------------|
| `alpha` | `0.05` | Significance level. |
| `equivalence_margin` | `0.05` | Largest score gap treated as practically negligible (for equivalence). |
| `power_target` | `0.8` | Target power for the sample-size / minimum-detectable-effect advice. |
| `smallest_meaningful_effect` | `0.2` | Cohen's d worth powering for when recommending more examples. |
| `precision_margin` | `0.05` | Target confidence-interval half-width for a single-model score. |
| `saturation_fraction` | `0.95` | Mean-over-ceiling above which a benchmark is "saturated". |
| `score_ceiling` | *(observed max)* | True upper bound of the score scale (e.g. `5.0` for a 0–5 rubric). When omitted, the saturation check uses the highest observed score as the ceiling, which can produce false warnings on continuous scales where no example scores at the maximum. Set this whenever your rubric has a fixed maximum. |
| `min_spread` | `0.01` | Pooled score std below which the benchmark can't discriminate. |
| `judge_agreement_threshold` | `0.8` | Inter-judge and binary judge-vs-gold agreement floor (fraction agreed). |
| `judge_correlation_threshold` | `0.8` | Continuous judge-vs-gold Spearman rank-correlation floor (rho). |
| `reference_judge` | *(auto)* | Name of the human/gold judge to calibrate against (else auto-detected). |
| `n_resamples` | `10000` | Bootstrap / permutation resamples. |
| `seed` | `0` | RNG seed (reproducibility). |
| `correction` | `"bonferroni"` | Multiple-comparison correction for a multi-metric suite or an enabled all-pairs family: `bonferroni`, `holm`, or `none`. |
| `all_pairs` | `false` | Also test every declared model pair (`--all-pairs`). Off by default. |
| `bayesian` | `false` | Add an advisory Bayesian win-probability finding (`--bayesian`). Off by default; never changes the verdict. |
| `gated_metrics` | *(none)* | Metric names that must reach High confidence, or the whole suite is Low regardless of the other metrics. |
| `metric_weights` | *(none)* | Per-metric weights (all positive). Validated and stored; reserved for a future weighted suite rollup. |

## Multiple-comparison correction

When a file scores several metrics, testing them all at the same `alpha` inflates
false positives, so EvalTrust corrects the significance threshold for the number
of metrics. Choose the method with `correction` (or `--correction` on the CLI):

When `all_pairs = true` or `--all-pairs` is enabled for a single comparison, the
same method corrects across the model pairs that have shared scores. Pairs without
shared scores are listed but do not make the tested family larger. The same flag
also emits one advisory rank-stability finding for files with three or more
scored models: a joint bootstrap of the mean-score ranking reports which
leaderboard positions hold under resampling. Under the same flag, two-model
files still get the pair family and a rank-stability SKIP. A multi-metric suite
still compares one selected model pair per metric. It does not build a
pair-by-metric grid and suppresses both all-pairs and rank stability.

- **`bonferroni`** (default) - divide the threshold by the number of tests in
  the family (`alpha / k`). Simple and strict.
- **`holm`** - Holm-Bonferroni, a step-down refinement that rejects at least
  as many hypotheses as Bonferroni at the same family-wise error rate, so a
  genuine win is less likely to be missed.
- **`none`** - test each hypothesis at the raw `alpha` with no correction.

```toml
correction = "holm"
```

Or per run: `evaltrust audit results.json --correction holm`.

Enable the optional pair family in config:

```toml
all_pairs = true
```

Or per run: `evaltrust audit results.json --all-pairs`.

## From Python

```python
from evaltrust.config import AuditConfig
from evaltrust import run_audit

cfg = AuditConfig(alpha=0.01, equivalence_margin=0.02)
report = run_audit(data, config=cfg)
```

`AuditConfig.load()` reads the same files the CLI does, if you want to honour a
repo's policy in your own scripts.
