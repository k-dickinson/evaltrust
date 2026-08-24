# pass@k for EvalTrust — Design

Issue: #164 (`enhancement`, `priority: high`, `statistics`)

## Problem

Code benchmarks (HumanEval, MBPP, SWE-bench) sample `k` completions per task and
report `pass@k` using the unbiased estimator `1 - C(n-c, k) / C(n, k)` (n samples,
c correct) rather than a plain pass rate. That estimator carries its own sampling
variance that a flat pass/fail comparison discards. EvalTrust currently has no
notion of k-sampling: to audit a code eval you must first collapse each task to a
single pass rate, losing the structure the metric is built on and understating
uncertainty.

## Goal

Add an **opt-in** pass@k comparison between two models that:

- carries per-task sample counts (n samples, c correct) into the audit,
- reports each model's pass@k point estimate with a bootstrap interval,
- reports the paired difference `Δ = pass@k_B − pass@k_A` with an interval and a
  PASS/WARN/FAIL verdict,
- leaves the existing collapsed-score path completely unchanged.

Non-goals (deliberate, YAGNI): `pass^k` (all-k-pass reliability), multiple `k`
values in one run, a configurable pass threshold. Internals are factored so
`pass^k` is a later drop-in.

## Input / schema

No schema change. Reuse the existing `Example.runs[model]` field (already parsed
by ingest and consumed by the Repeatability pillar). Each run value is one
sampled completion:

```python
Example(
    id="task_12",
    scores={"gptX": 0.7},          # collapsed rate — unchanged, still used elsewhere
    runs={"gptX": [1, 0, 1, 1, 0]}  # n = len(runs["gptX"]), c = count of passes
)
```

- A sample counts as a **pass** iff its score `== 1.0`. pass@k inputs are
  inherently binary (a completion passes tests or not). Non-`{0.0, 1.0}` run
  values are counted as fails and surfaced in the finding's `how_detected` and
  `details` so silent misuse on continuous-score data is visible, not hidden.
- For a task and model: `n = len(runs[model])`, `c = sum(1 for v in runs[model] if v == 1.0)`.

## Estimator — `src/evaltrust/stats/pass_at_k.py`

Per-task unbiased estimator, numerically stable HumanEval product form (no
factorials, no scipy needed, no overflow at large n):

```python
def _pass_at_k_single(n: int, c: int, k: int) -> float:
    if n - c < k:          # cannot avoid a correct sample among any k drawn
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
```

A model's pass@k over a task set is the mean of the per-task estimates.

Public function:

```python
@dataclass(frozen=True)
class PassAtKResult:
    k: int
    pass_at_k_a: float
    pass_at_k_b: float
    delta: float                 # b - a, EvalTrust's B - A convention
    delta_low: float
    delta_high: float
    a_low: float
    a_high: float
    b_low: float
    b_high: float
    confidence: float
    n_tasks: int                 # tasks with n >= k for BOTH models (used)
    n_skipped: int               # tasks dropped because n < k for either model
    n_nonbinary: int             # run values outside {0,1} treated as fails
    method: str                  # "passk-paired-percentile-bootstrap-v1"


def paired_pass_at_k(
    per_task: Sequence[tuple[int, int, int, int]],  # (n_a, c_a, n_b, c_b) per used task
    k: int,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PassAtKResult: ...
```

### Interval — paired bootstrap over tasks

Precompute per-task `passk_a[i]` and `passk_b[i]` arrays over the used tasks.
Point estimates are their means; `delta = mean(passk_b) - mean(passk_a)`.

CIs come from the existing, well-tested `stats/resampling.bootstrap_statistic_ci`
called three times with the **same seed** and same task count, so all three draw
identical resample indices and the paired Δ interval is consistent with the
per-model intervals:

- `a_low/a_high  = bootstrap_statistic_ci(passk_a,          mean, seed=seed)`
- `b_low/b_high  = bootstrap_statistic_ci(passk_b,          mean, seed=seed)`
- `delta_low/high = bootstrap_statistic_ci(passk_b - passk_a, mean, seed=seed)`

where `mean = lambda m: m.mean(axis=-1)` matches the vectorized `(rows, n)`
contract. This captures task sampling; the unbiased per-task estimator already
accounts for completion sampling.

Validation mirrors `paired_win_rate`: `k >= 1`, `confidence in (0,1)`,
`n_resamples >= 1`, non-empty `per_task`, all counts finite with `0 <= c <= n`.

## Audit check — `src/evaltrust/audit/pass_at_k.py`

`audit_pass_at_k(data, model_a, model_b, k, *, confidence=0.95, n_resamples, seed)`
returns `list[Finding]`, following the `win_rate.py` template (module `PILLAR`,
`CHECK`, `METHOD`, shared `_details`, `_skip`):

- `PILLAR = "Statistical Validity"`, `CHECK = "pass_at_k"`.
- Build `per_task` from examples where **both** models have `>= k` runs. Count
  skipped tasks (n < k for either model) and non-binary values.
- If no task qualifies → single `Status.SKIP` finding explaining why (no runs, or
  every task has n < k), with `assessed=False` in `details`.
- Otherwise call `paired_pass_at_k` and emit one finding:
  - `Status.PASS` when the Δ interval excludes 0 (a real difference at this k),
  - `Status.WARN` when the Δ interval contains 0 (advisory; no decisive gap),
  - title shows `pass@k` for A and B and the Δ with its interval,
  - `how_detected` reports n_tasks used, n_skipped (n<k), n_nonbinary, seed,
  - `details` carries the full `PassAtKResult` fields for `--json`.

The check is advisory about ranking (like win_rate); it contributes WARN/PASS but
does not by itself force a LOW verdict beyond existing verdict rules.

## Wiring

- **Config** (`config.py`): add `k: int | None = None` to `AuditConfig`; add it to
  the explicit `__hash__` tuple; validate in `__post_init__` that when set it is a
  positive, non-bool integer.
- **Runner** (`audit/runner.py`): in `_comparison`, after the existing advisory
  checks, `if cfg.k is not None and data.has_runs: findings += audit_pass_at_k(...)`.
  Not run for single-model or preference-only paths (no paired runs to compare).
- **API** (`api.py`): add `k: int | None = None` kwarg to `audit(...)`, threaded
  into the constructed `AuditConfig` (respecting the `config=` precedence rule).
- **CLI** (`cli.py`): add `--k` typer option (`Optional[int]`, default None) and
  include `("k", k)` in the `overrides` dict.

## Testing (TDD)

`tests/test_pass_at_k.py` (stats) and additions to the audit/api/cli test suites:

- estimator vs hand-computed `C()` values; boundaries `c = 0`, `c = n`, `n = k`,
  and `n - c < k` (returns 1.0);
- numerical stability at large n (e.g. n = 200, k = 100) — finite, in [0, 1];
- monotonicity sanity (more correct → higher pass@k);
- `paired_pass_at_k` determinism under fixed seed; Δ interval consistency with
  per-model intervals (shared resamples);
- `n < k` tasks skipped and counted; non-binary values counted and treated as
  fails;
- check activation gating: absent when `k` is None or no runs; SKIP when every
  task has n < k; PASS vs WARN by whether Δ interval excludes 0;
- CLI `--k` and API `audit(..., k=...)` end-to-end golden run;
- existing suite stays green (the collapsed-score path is untouched).

## Backward compatibility

Entirely additive and opt-in. With no `k` set, behavior is byte-for-byte
unchanged. The `runs` field keeps its current meaning for Repeatability; pass@k
only reads it, never mutates it.
