# Using EvalTrust: standalone or embedded

EvalTrust works two ways, and you can use either or both.

## 1. Standalone (the CLI)

Run it by hand at a decision point, or in a shell script:

```bash
evaltrust audit results.json
evaltrust audit results.json --fail-under moderate   # non-zero exit if below Moderate
```

## 2. Embedded in your own eval or tests (the Python API)

Call it inside the code that runs your evaluation and fail when the result isn't
trustworthy. One line does it:

```python
import evaltrust

report = evaltrust.audit("results.json")
report.raise_if_below("moderate")     # raises UntrustworthyError if confidence is too low
```

### In pytest

Turn "is my eval trustworthy?" into a normal test:

```python
import evaltrust

def test_new_prompt_is_a_real_improvement():
    report = evaltrust.audit(["old_prompt.json", "new_prompt.json"])
    report.raise_if_below("moderate")   # fails the test on a Low verdict
```

`UntrustworthyError` subclasses `AssertionError`, so pytest reports it as a clean
failure with the verdict in the message. `raise_if_below` returns the report, so
you can inspect it too:

```python
report = evaltrust.audit("results.json").raise_if_below()
print(report.verdict.summary)
```

Multi-metric suites have the same guard (it uses the weakest metric):

```python
evaltrust.audit_suite("suite.csv").raise_if_below("moderate")
```

## 3. In CI (GitHub Actions)

Use the bundled action to gate a pull request:

```yaml
# .github/workflows/eval.yml
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: k-dickinson/evaltrust@v1
        with:
          results: results.json
          min-confidence: moderate     # fail the job below this level
```

Or without the action, just call the CLI:

```yaml
      - run: |
          pip install evaltrust
          evaltrust audit results.json --plain --fail-under moderate
```

## Catching regressions between runs

Save each run's audit as JSON, then compare across releases to catch a metric
that got worse:

```bash
evaltrust audit results.json --json > audit-$(git rev-parse --short HEAD).json
evaltrust diff audit-last.json audit-new.json     # non-zero exit on a regression
```

```
Regressions
  ✗ safety confidence: HIGH → LOW
  ✗ safety decision: significant → inconclusive
```

Use `--no-fail-on-regression` for a report-only diff.

## Exit codes and levels

`--fail-under LEVEL` (and the action's `min-confidence`) accept `high`,
`moderate`, or `low`:

- `--fail-under high` - the strictest gate: fails on Moderate or Low.
- `--fail-under moderate` - fails only on Low. (Same as the older `--strict`.)
- `--fail-under low` - never fails on confidence; report only.

The CLI exits `0` when the gate passes, `1` when confidence is below the gate,
and `2` on a usage or input error.
