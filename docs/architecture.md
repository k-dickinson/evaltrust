# Architecture

EvalTrust turns an evaluation results file into a trust verdict through a short,
one-directional pipeline:

```
your eval output
      │
      ▼
┌──────────────┐   detect the format by structure, map it into
│   adapters   │   one internal representation
└──────────────┘
      │
      ▼
┌──────────────┐   Example / EvalData: models, per-example scores,
│  core model  │   optional repeated runs and per-judge scores
└──────────────┘
      │
      ▼
┌──────────────┐   pure numeric primitives: resampling, effect size,
│    stats     │   power, inter-rater agreement
└──────────────┘
      │
      ▼
┌──────────────┐   each check reads EvalData, calls stats, returns Findings;
│    audit     │   the verdict combines them
└──────────────┘
      │
      ▼
┌──────────────┐   render findings + verdict to the terminal
│    report    │
└──────────────┘
```

## Modules

| Package | Responsibility |
|---------|----------------|
| `core/schema.py` | The canonical data model: `Example`, `EvalData`, `Finding`, `Status`. |
| `core/ingest.py` | Read a file from disk; route JSON through auto-detection and CSV through the record reader. |
| `core/pairing.py` | Pair two single-model files into one A-vs-B comparison. |
| `adapters/` | Detect a format (Promptfoo, DeepEval, nested JSON, records, CSV) and map it into the canonical model. |
| `stats/` | Pure statistical primitives (resampling, effect size, power, McNemar, agreement). No knowledge of findings or formatting. |
| `audit/` | The checks and the runner. Comparison (`statistical`), single-model (`single`), suites (`suite`), `benchmark_health`, `repeatability`, `judge_reliability`, `judge_calibration`, and the `verdict` rules. |
| `config.py` | `AuditConfig` - every threshold in one place, loadable from `.evaltrust.toml`. |
| `diff.py` | Compare two saved audits to flag regressions. |
| `report/terminal.py` | Render a report to the terminal (`rich`) or as plain ASCII. |
| `api.py` | The public `evaltrust.audit(...)` / `audit_suite(...)` entry points. |
| `cli.py` | The `evaltrust audit` and `evaltrust diff` commands (`--json`, `--plain`, `--explain`). |

## Design principles

**One internal format.** Every adapter maps its input into the same `EvalData`.
The statistics are written once and work for every source. Adding a new format
never touches the audit code.

**`stats/` is pure.** It takes arrays and returns numbers. It doesn't know what a
finding is or how anything is displayed, which makes it easy to test against
reference implementations and easy to reuse.

**Checks degrade gracefully.** A check that needs data the file doesn't have
(repeated runs, multiple judges) returns a `SKIP` finding explaining how to
generate that data, rather than failing.

**The auditor is reproducible.** All resampling takes a seed and is
deterministic. The same input always produces the same report.

**The dependency graph points inward.** `adapters` and `audit` depend on `core`
and `stats`; `core` and `stats` depend on nothing internal. There are no cycles.

## Data model

```python
@dataclass(frozen=True)
class Example:
    id: str
    scores: dict[str, float]                          # model -> final score
    runs: dict[str, list[float]] | None = None        # model -> repeated-run scores
    judges: dict[str, dict[str, float]] | None = None # judge -> {model -> score}

@dataclass(frozen=True)
class EvalData:
    models: list[str]
    examples: list[Example]
    source_format: str
    metadata: dict
```

`scores` is always present. `runs` and `judges` are optional; their presence is
what unlocks the Repeatability and Judge Reliability checks.
