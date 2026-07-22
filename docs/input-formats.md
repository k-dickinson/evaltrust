# Input formats

You never write an EvalTrust-specific format. Point the tool at whatever your eval
framework produced and it detects the shape by structure - not by file name - and
maps it into one internal representation.

```bash
evaltrust audit results.json
evaltrust audit results.csv
```

## Supported formats

### Promptfoo

Promptfoo compares several providers across the same test cases, which is exactly
the A-vs-B comparison EvalTrust audits. Each provider becomes a model; each test
case becomes an example. Pass the exported results JSON directly.

### DeepEval

DeepEval evaluates one model per run, so its export contains a single model.
EvalTrust reads the evaluation-results export (from `evaluate(...)` or
`deepeval test run`), tolerating both the `test_results`/`metrics_data` and
`testCases`/`metricsData` shapes. Each test case's score is its `success`
(pass/fail), falling back to the mean of its metric scores. To compare two models,
run each and pass both files:

```bash
evaltrust audit deepeval_gpt4.json deepeval_claude.json
```

If DeepEval recorded a model name under `hyperparameters`, it's used as the label;
otherwise the file name supplies it.

### Inspect (UK AISI)

Inspect writes one `EvalLog` per run as a single JSON document. EvalTrust reads
the `.json` log format directly: the model comes from `eval.model`, each entry in
`samples` becomes an example, and the scorer's grade under `sample.scores` becomes
the score. Inspect's grade constants map the way Inspect's own `value_to_float`
maps them - `C`/`I`/`P`/`N` → 1 / 0 / 0.5 / 0 - and numeric scores pass through.
An Inspect log holds a single model, so compare two runs:

```bash
evaltrust audit inspect_run_a.json inspect_run_b.json
```

A log with several scorers is audited on its first scorer (as with OpenEvals);
per-scorer multi-metric support is a possible follow-up.

### OpenAI Evals

OpenAI Evals (`openai/evals`) writes a line-delimited `.jsonl` log per run: a
leading `spec` object, a stream of per-sample events, and a trailing
`final_report`. EvalTrust reads the model from `spec.completion_fns`, and each
`match` event's `data.correct` bool becomes that sample's score (metric
`accuracy`). A log holds a single model, so compare two runs:

```bash
evaltrust audit openai_run_a.jsonl openai_run_b.jsonl
```

Model-graded evals record a config-mapped `choice`/`score` instead of a `correct`
bool; those events are skipped and counted for now, pending a follow-up.

### Langfuse

Langfuse scores can be exported from the public Scores API. For the current v3
API, request the `subject` field group so EvalTrust can associate each score with
its trace:

```bash
curl -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  "https://cloud.langfuse.com/api/public/v3/scores?fields=subject&limit=100" \
  > langfuse_scores.json
evaltrust audit langfuse_scores.json
```

**The command above shows the shape of the request, not a complete export.**
The Scores API paginates, and a single page's `data` array only ever holds
that page's rows - being the *last* page doesn't mean the earlier pages' scores
are included too. EvalTrust refuses to parse a `{"data": [...], "meta": {...}}`
response unless its `meta` proves this was the only page:

- **v2**: only `meta.page == 1` **and** `meta.totalPages == 1` is accepted.
  Any other `page`/`totalPages` combination - including the last page of a
  multi-page result, e.g. `page: 2, totalPages: 2` - is rejected, because that
  response's `data` still only has that one page's rows.
- **v3**: a non-null `meta.cursor` (more results follow) is rejected. A missing
  or null `cursor` means *this* page is the last one, but v3's `meta` carries
  no page count to check against - so it cannot prove an independently saved
  "last page" response actually included every earlier page too. Treat a
  cursor-paginated export the same way: request every page and combine them
  yourself; don't assume a lone saved response is complete just because its
  `cursor` is empty.

Either way, fetch every page and concatenate each page's `data` array into one
combined **bare JSON list** (drop the `meta` wrapper entirely) before running
`evaltrust audit` - a bare list bypasses this check because it can't
accidentally look complete when it isn't.

Score `name` values become metrics. `NUMERIC` values are read directly from
`value`, and `BOOLEAN` values from `value` however Langfuse encodes them for
that API version (a JSON boolean in v3, numeric `1`/`0` in legacy exports).

`CATEGORICAL` scores are read differently depending on the API version,
because v3 and legacy exports disagree on where the label lives:

- **v3** (`subject`-shaped rows): `value` *is* the category string (e.g.
  `"correct"`), read and coerced directly. There is no `stringValue` field in
  v3, and a `configId` (only present when `fields=details` was requested) does
  not change this.
- **Legacy flat exports**: `value` is a number that only means something when
  a `configId` links it to a score config - Langfuse's own schema says it
  "defaults to 0" otherwise - so without a `configId`, the human-readable
  `stringValue` is used instead.

Either way, the resulting string or number must map unambiguously to a number
or a pass/fail-style label (e.g. `correct`/`incorrect`), or the row is skipped
and counted. **Session- and experiment-level scores are unsupported** -
EvalTrust compares models per trace and raises a clear error if a score's
`subject.kind` isn't `trace` or `observation`. **`TEXT` and `CORRECTION`
scores are always skipped** (and counted) - they're free text, not a number.
Legacy exports with flat `traceId` fields are also supported, including when
they arrive `{"data": [...], "meta": {...}}`-wrapped like v3.

If a trace has more than one score with the same metric name - for example a
trace-level score and an observation-level score both named `correctness` -
EvalTrust raises rather than averaging or picking one; combining them needs an
explicit aggregation policy (max, latest, per-observation, ...) decided by a
maintainer, not a silent guess.

A Langfuse export contains scores for one model/run but does not carry a model
identity, so it is labeled `model`. To compare models, export each run separately
and pass both files; their file names become the labels when the generic model
names collide. As with the other [single-model tools](#single-model-tools-two-file-comparison),
`evaltrust audit fileA.json fileB.json` uses only the **first** metric name it
finds in each file - a Langfuse export with several score names needs one audit
per metric, or a follow-up per-metric suite mode:

```bash
evaltrust audit gpt4_langfuse.json claude_langfuse.json
```

### Nested JSON

A structured object with a list of examples, each carrying per-model scores:

```json
{
  "models": ["gpt-4", "claude-3"],
  "examples": [
    { "id": "q1", "scores": { "gpt-4": 1, "claude-3": 0 } },
    { "id": "q2", "scores": { "gpt-4": 0, "claude-3": 1 } }
  ]
}
```

Optional per-example `runs` and `judges` unlock the Repeatability and Judge
Reliability checks:

```json
{
  "id": "q3",
  "scores": { "gpt-4": 1, "claude-3": 1 },
  "runs":   { "gpt-4": [1, 1, 0], "claude-3": [1, 0, 1] },
  "judges": { "gpt": { "gpt-4": 1, "claude-3": 0 },
              "human": { "gpt-4": 1, "claude-3": 1 } }
}
```

An optional `attributes` object tags each example for the per-slice comparison
(`--slice-by <name>`):

```json
{
  "id": "q4",
  "scores": { "gpt-4": 1, "claude-3": 0 },
  "attributes": { "category": "math", "difficulty": "hard" }
}
```

Attributes are currently read only from the nested-JSON adapter. CSV and
generic record lists don't carry slice tags yet — a dedicated slice column for
those formats is a possible follow-up.

An optional `group_id` marks examples that are **not** independent — repeated
judgments of the same item, or items sharing a task/template. When present, the
significance test and confidence intervals resample whole clusters, so they
reflect that correlation instead of assuming every example is independent:

```json
{
  "id": "q5",
  "scores": { "gpt-4": 1, "claude-3": 0 },
  "group_id": "template-A"
}
```

### Pairwise preference

When a judge votes for a winner (A / B / tie) instead of scoring each model, use
`preferences` (judge → winning model id, or `"tie"`) in nested JSON, or a
`winner`/`preference` column (plus an optional `judge` column) in a record list
or CSV. EvalTrust runs an exact sign test on the decisive votes:

```json
{
  "id": "q6",
  "preferences": { "gpt-judge": "gpt-4", "human": "tie" }
}
```

```csv
id,winner,judge
q1,gpt-4,gpt-judge
q1,tie,human
```

### Record lists

A flat list of rows, one per (example, model). Column names are matched flexibly
(`model`/`provider`/`system`, `score`/`pass`/`success`, and so on):

```json
[
  { "id": "q1", "model": "gpt-4", "score": 1 },
  { "id": "q1", "model": "claude-3", "score": 0 }
]
```

### JSONL (line-delimited records)

The same records, one JSON object per line - the streaming-friendly shape many
eval harnesses emit. Point EvalTrust at a `.jsonl` file directly:

```jsonl
{"id": "q1", "model": "gpt-4", "score": 1}
{"id": "q1", "model": "claude-3", "score": 0}
```

Blank lines (and a trailing newline) are ignored. Each line must be a single JSON
object; a malformed line is reported with its line number rather than skipped
silently, and a file whose content is actually a JSON array is read as one JSON
document. A `metric` column fans out into a multi-metric suite exactly as it does
for CSV and record lists.

Known line formats are detected before generic record extraction; unclaimed rows
keep the existing JSONL behavior.

### CSV

Long format - one row per (example, model):

```csv
id,model,score
q1,gpt-4,1
q1,claude-3,0
```

Wide format - one column per model:

```csv
question,gpt-4,claude-3
q1,1,0
q2,0,1
```

Scores can be numbers, booleans, or words like `pass`/`fail`, `true`/`false`,
`yes`/`no`, `correct`/`incorrect`.

## Multiple metrics

If your eval scores several metrics per example (correctness, safety, tone...),
add a `metric` column to the long format. EvalTrust audits each metric separately
and corrects for the number of metrics tested:

```csv
id,model,metric,score
q1,gpt-4,correctness,1
q1,claude-3,correctness,0
q1,gpt-4,safety,1
q1,claude-3,safety,1
```

The same works in JSON record lists (`{"id","model","metric","score"}`). A file
without a `metric` column is treated as a single metric, exactly as before. See
[checks](checks.md#multiple-metrics-suites) for how the metrics are combined.

## Single-model tools (two-file comparison)

Some tools - DeepEval, Langfuse, LangSmith, Ragas, OpenEvals, Inspect - evaluate
one model per run, so a single export contains only one model. Run each model,
then pass both files:

```bash
evaltrust audit gpt4_run.json claude_run.json
```

EvalTrust pairs the two files by example id. Each file must contain exactly one
model; a file that already has several models should be audited on its own. Model
labels default to the models' own names, falling back to the file names if those
collide, and can be overridden with `--model-a` and `--model-b`.

If you only have **one** model and no second file to compare against, don't pass
two files - just audit the single file. EvalTrust switches to auditing whether the
score itself is trustworthy (a confidence interval on it), and `--threshold 0.8`
tests whether the model clears a target. See
[Score Reliability](checks.md#single-model-score-reliability).

## When a format isn't recognized

Detection fails loudly rather than guessing. If EvalTrust can't recognize a file it
tells you what it looked for, so you can reshape the data into one of the formats
above - or, better, [contribute an adapter](adapters.md) for it.
