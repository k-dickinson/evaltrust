# Run-aware predictive reruns

Run-aware mode estimates what may happen when the same examples are rerun with
fresh, independent draws. It does not estimate performance on new examples. The
finding is advisory and does not change the audit verdict.

Examples used by the finding need per-example `runs` for both models. See
[Input formats](input-formats.md#nested-json) for the nested JSON shape. For an
example to be used, each stored score must equal the mean of its run values.

## Enable the finding

Pass `--run-aware` and a positive integer future run count:

```bash
evaltrust audit results.json --run-aware --future-runs 3
```

`--future-runs 3` asks about three future runs per model and example. EvalTrust
does not infer this count from the observed runs.

To enable the same finding in `.evaltrust.toml` or `[tool.evaltrust]`, use:

```toml
run_aware = true
run_aware_future_runs = 3
```

## Read the finding

For a file with models A and B, two eligible examples, and one example without
B's run stream, the plain output includes:

```text
[ok  ] P(A better on a rerun of these examples at 3 future runs) (A=A, B=B) = 99.9%; n_predictive=2/3; predictive range (normal-theory approximation) for B - A [-0.5883, -0.2117]
```

The probability is for the strict event `future B - A < 0` on these examples.
It is not a claim that A is intrinsically better or that A will perform better
on new examples.

`n_predictive=2/3` means that two of the three examples were used. The predictive
finding's JSON `details` shows the same counts and aggregate first-match
exclusion totals:

```json
{
  "n_total": 3,
  "n_predictive": 2,
  "reason_counts": {
    "missing_runs_a": 0,
    "missing_runs_b": 1,
    "nonfinite_a": 0,
    "nonfinite_b": 0,
    "score_run_mean_disagreement": 0,
    "singleton_in_mixed": 0
  }
}
```

The reasons mean:

- `missing_runs_a` or `missing_runs_b`: that model has no stored score or no
  nonempty run stream.
- `nonfinite_a` or `nonfinite_b`: that model's run stream is invalid or contains
  a nonfinite value.
- `score_run_mean_disagreement`: a stored score does not match its run mean.
- `singleton_in_mixed`: the example has a one-run A or B stream while at least
  one admitted A or B stream has multiple runs.

If no examples remain, the finding is `SKIP` and its predictive range is
unavailable.

## Accuracy limits

The output calls the range a `predictive range (normal-theory approximation)`.
It is fitted model mass, not a calibrated coverage guarantee.

Centered, variance-scaled `LogNormal(0, 1)` simulations measured approximate-range
coverage around 0.933 with 8 examples, 3 observed runs, and 1 future run, and
around 0.937 with 25 examples, 6 observed runs, and 1 future run. Trust the
approximation less with few examples or observed runs, or with skewed run scores.

See [Normal-theory predictive reruns](predictive-rerun.md) for the method,
assumptions, degenerate cases, and simulation details.
