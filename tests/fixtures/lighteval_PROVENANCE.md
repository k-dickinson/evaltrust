# Lighteval fixture provenance

## `lighteval_details_agieval_aqua_rat.json`

**Source:** Official Hugging Face Lighteval repository reference details
(`huggingface/lighteval`).

- Upstream Git commit (immutable):
  `31433cc9e19d60635e9f62f271d5f3a8f2ed2696`
- Repository-relative path:
  `tests/reference_details/SmolLM2-1.7B-Instruct-transformers/details_agieval:aqua-rat|0_2025-11-05T14-43-47.148527.parquet`
- Immutable GitHub blob URL:
  https://github.com/huggingface/lighteval/blob/31433cc9e19d60635e9f62f271d5f3a8f2ed2696/tests/reference_details/SmolLM2-1.7B-Instruct-transformers/details_agieval:aqua-rat%7C0_2025-11-05T14-43-47.148527.parquet
- Direct download (Git LFS / media CDN, same commit + path):
  https://media.githubusercontent.com/media/huggingface/lighteval/31433cc9e19d60635e9f62f271d5f3a8f2ed2696/tests/reference_details/SmolLM2-1.7B-Instruct-transformers/details_agieval%3Aaqua-rat%7C0_2025-11-05T14-43-47.148527.parquet
- SHA-256 of the **original Parquet artifact** used for conversion:
  `ef968329bee498b3387ec8df3677ca9bbac72e90599efbe7f78db23f4227b2f6`
- Timestamp in filename: `2025-11-05T14-43-47.148527`
- Model evaluated: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- Task: `agieval:aqua-rat|0`
- Lighteval commit referenced by the matching aggregate results file:
  `config_general.lighteval_sha` = `01bd59882e01ab82971e6ab07c7c39c69ab8664b`
  (evaluation-runtime SHA from the results JSON; distinct from the upstream
  repository commit that stores this reference Parquet)

**Column names in this recent official reference artifact:** `doc`, `metric`,
`model_response` (verified by reading the Parquet schema). This is the shape
preserved in the fixture.

**Current public documentation shape:** the Hugging Face Lighteval docs
describe per-sample detail columns as `__doc__`, `__metric__`, and
`__model_response__`. The adapter accepts those exact aliases as well; they are
covered by focused inline tests rather than this authentic fixture.

**Reduction / sanitization (EvalTrust JSON fixture, not the Parquet digest):**

- Converted from the Parquet above to JSON (EvalTrust JSON adapters; no runtime
  Parquet dependency).
- Kept the first 3 of 10 authentic samples.
- Truncated `model_response.input_tokens` / `output_tokens` for size only;
  `doc`, `metric`, and remaining `model_response` fields are unchanged.
- No credentials, local paths, or tokens were present.
- The SHA-256 above fingerprints the **original Parquet file**. The reduced
  JSON in this repository is a derived fixture and has a different digest.

Redistribution: these are public Lighteval test fixtures (MIT License,
Hugging Face Team).

## `lighteval_results_aggregate.json`

**Source:** Official Lighteval reference scores
`tests/reference_scores/SmolLM2-1.7B-Instruct-results-accelerate.json`.

**Reduction:** Kept the authentic top-level shape (`config_general`, `results`,
`versions`, `config_tasks`, `summary_tasks`, `summary_general`) but retained only
two task entries plus the `all` aggregate. Used as a negative fixture: this
shape is detected as Lighteval but rejected on parse because it has no
per-example scores.
