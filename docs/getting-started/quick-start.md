# Quick Start

LongProbe tests whether your retriever still returns expected evidence for a fixed set of questions. It does not automatically inspect every live user query.

## 1. Install and initialize

```bash
pip install longprobe
longprobe init
```

Initialization creates two example files and a baseline directory:

- `goldens.yaml`: questions and required evidence.
- `longprobe.yaml`: retrieval connection, scoring, and baseline settings.
- `.longprobe/`: local baseline storage.

The file content comes from `GOLDENS_TEMPLATE` and `CONFIG_TEMPLATE` strings in CLI source. `init` does not read documents, embeddings, or a database. Existing files are preserved unless you explicitly run `longprobe init --force`.

## 2. Configure a retriever

You need an existing retrieval endpoint or supported store. For a synthetic local example, run this command from a source checkout in another terminal:

```bash
python examples/http_retriever.py
```

This development fixture returns a known refund-policy chunk; it is not a production retrieval service. Configure `longprobe.yaml`:

```yaml
retriever:
  type: http
  http:
    url: http://127.0.0.1:8000/retrieve
    method: POST
    response_mapping:
      results_path: data.chunks
      id_field: chunk_id
      text_field: content
      score_field: similarity

scoring:
  recall_threshold: 0.8
  fail_on_regression: true

baseline:
  db_path: .longprobe/baselines.db
  auto_compare: true
```

For Chroma, install `longprobe[chroma]` and configure `collection` and `persist_directory` directly under `retriever`, not under a nested `chroma` section. The application/store handles its existing embedding model; the `embedder` configuration is not connected to the normal check path in this release.

## 3. Define verified golden questions

Replace the sample cases in `goldens.yaml` with requirements you have verified:

```yaml
name: refund-policy
version: "1.0"
questions:
  - id: refund_policy
    question: What is the refund policy?
    required_chunks:
      - Refunds are allowed within 30 days.
    match_mode: text
    top_k: 5
    status: approved
    tags: [policy, critical]
```

Alternatively, capture a question through your configured retriever:

```bash
longprobe capture --question "What is the refund policy?" --auto
longprobe edit
```

Automatic capture saves drafts. In the editor, review the evidence in YAML as necessary, approve the intended drafts, and save. Interactive capture lets you select evidence and saves approved questions. Do not assume current retrieval output is independently correct ground truth.

For document-based generation, install `longprobe[generate]`, configure your generator provider/model and credentials, then run:

```bash
longprobe generate ./documents --num-questions 20 --auto-capture
longprobe edit
```

PDF/Word extraction depends on installed parsing capabilities. Generating questions without `--auto-capture` produces a question list rather than a complete evaluable golden set. A fixed question count does not guarantee full document coverage; add reviewed cases for important topics and real retrieval failures. LongProbe checks selected golden questions, not arbitrary live user queries. The terminal editor currently edits question text, status, and mode; use YAML to review/change required evidence.

## 4. Check and save a baseline

```bash
longprobe check
longprobe baseline save --label latest
```

Every approved question is sent to the retriever. LongProbe matches the returned evidence, calculates recall, and applies the passing threshold. With `fail_on_regression: true`, a below-threshold question or detected baseline regression makes `check` fail.

`latest` is a literal baseline label. Saving another label does not update it. Both CLI and Python baseline promotion default to every evaluated question passing. Saving is explicit and never edits golden questions. Legacy references remain readable but need a reviewed new evaluation before strict comparison; see [baseline management](../guide/baseline-management.md).

After changing your retrieval pipeline:

```bash
longprobe check
longprobe diff --baseline latest
```

## 5. Recover from input failures

| Failure | Action |
|---|---|
| Missing CLI configuration | Run `init` and configure it, or pass `--config PATH` |
| Invalid YAML or value | Correct the reported field/syntax; LongProbe will not silently use defaults |
| Unresolved active credential | Set the named environment variable or fix the active provider setting |
| No matching/approved questions | Correct tags, add questions, or review and approve drafts |
| Invalid existing goldens during generation | Repair the file or choose a new `--goldens PATH`; trusted data is not reset |
| Existing question-list export | Choose a new output path or explicitly use `--output PATH --overwrite` |

Empty suites may be created/edited, but cannot pass evaluation. `--include-drafts` includes actual drafts, not an empty-suite bypass.

## Other entry points

```bash
longprobe check --output github
longprobe check --tag critical
longprobe watch
```

Watch polls the golden file and runs after a change; it does not initially run or watch configuration/remote index changes. Configuration edits take effect the next time a check is invoked.

See [configuration](configuration.md), [CLI reference](../guide/cli-reference.md), and [v0.2.0 migration notes](../releases/v0.2.0.md) for limits and compatibility changes.
