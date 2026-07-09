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

## Single-model tools

If a tool evaluates one model per run, you don't need anything special: parse it
into a single-model `EvalData`, and users compare two runs with
`evaltrust audit runA.json runB.json`. See [input formats](input-formats.md).
