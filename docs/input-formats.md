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

Some tools - DeepEval, LangSmith, Ragas, OpenEvals, Inspect - evaluate one model per
run, so a single export contains only one model. Run each model, then pass both
files:

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
