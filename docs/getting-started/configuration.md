# Configuration

`goldens.yaml` defines what to test: questions, expected chunks, match mode, tags, approval status, and per-question `top_k`. `longprobe.yaml` defines connections, scoring, baselines, and generation settings.

## Working retrieval configuration

```yaml
retriever:
  type: http
  http:
    url: http://localhost:8000/api/retrieve
    method: POST
    body_template: '{"query": "{question}", "top_k": {top_k}}'
    headers:
      Authorization: Bearer ${API_TOKEN}
    response_mapping:
      results_path: data.chunks
      id_field: chunk_id
      text_field: content
      score_field: similarity
    timeout: 30
scoring:
  recall_threshold: 0.8
  fail_on_regression: true
  # Optional; replace with an ID in your full golden set:
  # question_thresholds:
  #   refund-policy: 1.0
baseline:
  db_path: .longprobe/baselines.db
  auto_compare: true
  quality_gate: all_questions
```

Remove the Authorization header when your local endpoint does not require authentication. HTTP settings live under `retriever.http`. Chroma/Pinecone/Qdrant connection fields live directly under `retriever`:

```yaml
retriever:
  type: chroma
  collection: documents
  persist_directory: ./chroma_db
```

Pinecone uses `index_name`, `api_key`, and `namespace`. Qdrant uses `collection`, `host`, `port`, and optional `api_key`. Their normal text-question flow is not connected to query embeddings in this release; use a properly configured application HTTP endpoint or custom Python adapter until that integration is implemented. Framework retrievers require the Python API.

## Strict loading and accepted values

CLI commands that accept a configuration path require that selected file to exist and be valid. Invalid files never silently become default settings. `init`, `edit`, and help need no retriever configuration. Baseline list/delete use their existing `--db-path` option rather than reading `longprobe.yaml`.

Python callers supplying an adapter may omit the conventional `longprobe.yaml`; documented defaults are then used. `./longprobe.yaml` has the same behavior. Missing custom/absolute selected paths raise `FileNotFoundError`. Any existing invalid selected file fails.

- Supplied sections and nested HTTP settings must be mappings.
- Omitted optional fields keep defaults; explicit invalid or null values fail.
- Integer fields accept integers or integer strings; booleans and fractional values are invalid.
- Floats must be finite. Recall threshold must be between 0 and 1.
- Ports must be 1–65535, dimensions nonnegative, and timeouts/batch sizes/generation counts/token limits positive.
- Booleans accept YAML booleans or case-insensitive strings `true/false`, `1/0`, `yes/no`, `on/off`.
- Provider-specific `extra` mappings are preserved; this release does not redesign unknown-key handling.

Validation errors identify fields without printing their values. Correct misspelled or incorrectly nested known settings; extra fields are not automatically used by every CLI adapter.

## Environment variables and active providers

`${VARIABLE}` values are expanded recursively. Unresolved placeholders in fields used by the current operation fail before its service calls. Unused generator/embedder credential placeholders do not block unrelated retrieval checks. Supplied typed numeric/structure settings are still validated even in inactive sections.

Generator configuration example:

```yaml
generator:
  provider: ollama
  model: your-installed-model
  base_url: http://localhost:11434
  num_questions: 20
  temperature: 0.7
  max_tokens: 4096
```

Cloud generation may use `generator.api_key` or the provider environment variable, such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`. Changing a generator model affects generated questions, not checks against existing goldens. Auto-capture validates both generator and retriever settings before generation. Local Chroma, unauthenticated Qdrant, and Ollama do not require unrelated cloud credentials.

## Scoring and baselines

`recall_threshold` controls the global per-question threshold. Invocation overrides supersede it, while explicit `question_thresholds` take precedence for their IDs. Values must be finite numbers in `[0, 1]`; unknown IDs fail against the full loaded suite, even when selection later filters questions.

`baseline.quality_gate` defaults to `all_questions` for both CLI and Python promotion. `aggregate` is an explicit alternative that permits individual failures if the mean meets the global threshold. Regular evaluation still reports every failed question.

`fail_on_regression: true` makes `check` fail for question failures or lost required evidence, including equal-recall evidence swaps. Setting it to false only relaxes completed quality failures; configuration, empty suites, incompatible comparisons, and execution failures remain nonzero. It never bypasses baseline promotion policy.

`baseline.db_path` selects SQLite storage. `auto_compare` uses literal `latest` when present. Explicit `--baseline` labels are required; `--no-compare` skips comparison. CI restores its database explicitly when configured. New report JSON metadata does not require an SQL table migration, but legacy references must be re-established before strict comparison. See [baseline management](../guide/baseline-management.md).

## Embedding and matching limitations

The standalone `QueryEmbedder` utility supports provider/model/dimensions/batch-size settings, but normal `check` and `LongProbe.run()` do not invoke it. Local test mode produces zero vectors, not useful semantic embeddings. Changing only `embedder.model` does not change normal retrieval.

`match_mode: semantic` uses word-frequency cosine similarity. Its threshold belongs to each golden question and does not use `QueryEmbedder`.

## Safe files

Golden saves, initialized configuration files, and generated question exports use a same-directory temporary file, flush, and atomic replacement. Existing restrictions are preserved; read-only or symbolic-link replacement targets are rejected. Recoverable failures preserve original destination bytes. The guarantee is per file for a single writer, not a multi-file transaction or concurrent-write solution.

`init` preserves existing files unless `--force` is specified. Auto-capture always requires a valid existing golden file and merges new drafts. Question-list export requires `--overwrite` to replace an existing `--output` file. Back up important files before upgrading or making deliberate replacements.
