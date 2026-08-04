# Writing a format adapter

Adding support for a new eval tool is the most common contribution, and it never
touches the audit code. An adapter answers two questions: *does this file look
like my format?* and *how do I map it into the canonical model?*

## The shape underneath everything

Whatever a tool's output looks like, it is ultimately reporting: for this example,
this model (or judge) produced this score. That is a `Record`:

```python
@dataclass(frozen=True)
class Record:
    example_id: str
    model: str
    score: float
    judge: str | None = None
```

`records_to_evaldata()` groups a list of records into canonical `EvalData`,
handling repeated runs and multiple judges for you. So most adapters are just:
locate the rows, pull out the fields, build records.

## An adapter

An adapter is any object with a `source_format` string and two methods:

```python
class MyToolAdapter:
    source_format = "mytool"

    def detect(self, raw) -> bool:
        # Is this parsed JSON object my format? Check for a distinctive structure,
        # not a file name.
        return isinstance(raw, dict) and "myToolResults" in raw

    def parse(self, raw) -> EvalData:
        records = []
        for row in raw["myToolResults"]:
            records.append(Record(
                example_id=str(row["caseId"]),
                model=str(row["model"]),
                score=coerce_score(row["passed"]),
            ))
        return records_to_evaldata(records, self.source_format)
```

Use `coerce_score()` to normalize the many ways scores are written (numbers,
booleans, `pass`/`fail`, and so on). If a field can't be interpreted as a score,
let it raise - silently guessing would undermine the point of an auditor.

For formats where the fields aren't in fixed positions, reuse `dicts_to_records()`
from `adapters/generic.py`, which matches column names against the shared alias
tables (`ID_KEYS`, `MODEL_KEYS`, `SCORE_KEYS`, `JUDGE_KEYS`).

## Registering it

Add your adapter to `REGISTRY` in `adapters/registry.py`. Order matters: specific
formats go before the generic fallbacks, so put a tool-specific adapter near the
top and let the generic record/CSV adapters catch everything else.

```python
REGISTRY: list[Adapter] = [
    MyToolAdapter(),
    PromptfooAdapter(),
    NativeNestedAdapter(),
    GenericRecordsAdapter(),
]
```

## Line-format adapters

Line-delimited formats use `LineAdapter` from `adapters/line_registry.py`:

```python
class LineAdapter(Protocol):
    source_format: str
    def detect_lines(self, rows: list[dict]) -> bool: ...
    def parse_lines(
        self, rows: list[dict], *, path: Path | None = None
    ) -> tuple[list[Record], dict]: ...
```

Register specific formats in `LINE_REGISTRY`. Unclaimed rows keep using the
existing generic JSONL record path.

The lm-eval line adapter reads per-sample `samples_<task>_<timestamp>.jsonl`
logs. When a sibling `results_*.json` sits in the same directory, it takes the
top-level `model_name` from that file. A matching timestamp wins; if none match
and exactly one results file is present, that file is used. Mixing samples from
different runs in one directory can mislabel under the sole-file fallback, so
keep each run's samples and results together (lm-eval's default layout). If no
usable sibling is found, the model name is inferred from the samples filename.
Metadata records which path was used via `model_name_inferred` and, on success,
`model_name_source` (filename only).

## Testing it

Add a test with a small fixture that represents the **real** file structure of the
tool, and assert both detection and parsing:

```python
def test_mytool_detects_and_parses():
    raw = {"myToolResults": [
        {"caseId": "q1", "model": "A", "passed": True},
        {"caseId": "q1", "model": "B", "passed": False},
    ]}
    adapter = MyToolAdapter()
    assert adapter.detect(raw)
    data = adapter.parse(raw)
    assert set(data.models) == {"A", "B"}
```

The most valuable fixture is one derived from a genuine export of the tool. If you
have a real sample file, base the test on it - that is how we make sure the
adapter matches reality and not an assumption.

## Lighteval (Hugging Face)

Supports JSON representations of Lighteval per-sample details, with tolerance
for optional fields. Each record has:

- a document (`doc` or `__doc__`): `query`, `choices`, `gold_index`, and usually
  `id` / `task_name`
- a metric dict (`metric` or `__metric__`): sample-level metric names to scores
- a model response (`model_response` or `__model_response__`, optional):
  generations / logprobs

Both the column names observed in recent official reference Parquet exports
(`doc` / `metric` / `model_response`) and the underscore-prefixed names in the
current public Hugging Face documentation (`__doc__` / `__metric__` /
`__model_response__`) are accepted. Native Parquet files are not read directly;
convert them to this JSON shape first.

Normalization rules for this adapter:

- Invalid, null, or unparseable metric values are ignored (not raised).
- Standard-error fields such as `*_stderr` are ignored.
- Valid metrics preserve input order when fanning out into a suite.
- Conflicting aliases (for example unequal `doc` and `__doc__`) are treated as
  malformed and skipped.
- `skipped_rows` counts skipped input rows, not individual metric fields.

Example IDs are preserved as `task_name:id` when `task_name` is present.
Multiple metrics fan out into a suite; the single-audit path prefers common
correctness metrics such as `acc` / `exact_match` / `em`.

**Limitation:** the aggregate `results_{timestamp}.json` file
(`config_general`, `results`, `config_tasks`, …) contains only task-level
summaries. EvalTrust detects that shape and raises a clear error rather than
fabricating per-example rows. Convert details Parquet to the JSON detail shape
above (or export a details list) before auditing.

## Single-model tools

If a tool evaluates one model per run, you don't need anything special: parse it
into a single-model `EvalData`, and users compare two runs with
`evaltrust audit runA.json runB.json`. See [input formats](input-formats.md).
