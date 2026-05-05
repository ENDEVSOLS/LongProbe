# Blueprint: HTTP Adapter, Generate Command & CI Integration

## Overview

Three major features are being added to LongProbe: **(A)** a generic HTTP retriever adapter that lets LongProbe test any RAG API endpoint (not just direct vector DB connections), **(B)** a `longprobe generate` command that uses an LLM to auto-generate golden questions from documents with zero manual effort, and **(C)** a GitHub Actions reusable workflow for CI/CD regression testing on every pull request. Together these transform LongProbe from a niche vector-DB testing tool into a universal RAG regression testing platform.

## Context

### Existing system
- **Adapters** (`src/longprobe/adapters/`): Each adapter (`ChromaAdapter`, `PineconeAdapter`, `QdrantAdapter`, etc.) extends `AbstractRetrieverAdapter` and implements `retrieve(query, top_k) -> list[dict]`.
- **Config** (`src/longprobe/config.py`): `ProbeConfig` is the top-level dataclass with `RetrieverConfig`, `EmbedderConfig`, `ScoringConfig`, `BaselineConfig` sub-sections. Supports `${ENV_VAR}` expansion. Loaded from `longprobe.yaml`.
- **CLI** (`src/longprobe/cli/main.py`): Typer app with `init`, `check`, `capture`, `baseline`, `diff`, `watch` commands. `_create_adapter_from_config()` dispatches by `retriever.type`.
- **Scoring** (`src/longprobe/core/scorer.py`): `RecallScorer` evaluates retrieved chunks against golden set.
- **Golden set** (`src/longprobe/core/golden.py`): `GoldenSet` / `GoldenQuestion` dataclasses with YAML I/O.
- **Existing capture** (`capture` command): Interactively or auto-captures retriever results into a golden set.

### Files this touches
- `src/longprobe/config.py` — add `HttpRetrieverConfig`, `HttpResponseMapping`, `GeneratorConfig`
- `src/longprobe/adapters/http.py` — **new** HTTP adapter
- `src/longprobe/adapters/__init__.py` — register HTTP adapter in factory
- `src/longprobe/core/docparser.py` — **new** document text extraction
- `src/longprobe/core/generator.py` — **new** LLM question generator
- `src/longprobe/cli/main.py` — add `generate` command, update `_create_adapter_from_config`, update `init` template
- `.github/workflows/longprobe.yml` — **new** reusable workflow
- `pyproject.toml` — add optional dependencies

### Patterns to follow
- Lazy imports for optional dependencies (matches `ChromaAdapter`, `PineconeAdapter` pattern)
- `${ENV_VAR}` expansion for secrets in config (matches existing `_expand_env_recursive`)
- Dataclass-based config with `from_dict` construction (matches existing `ProbeConfig.from_dict`)
- Rich console output with progress spinners (matches existing CLI commands)
- Optional dependency groups in `pyproject.toml` (matches existing `[project.optional-dependencies]`)

## Requirements

### A. Generic HTTP Adapter

1. User can set `retriever.type: http` in `longprobe.yaml` and configure `url`, `method`, `body_template`, `headers`, and `response_mapping`.
2. `HttpAdapter.retrieve(query, top_k)` makes an HTTP request using the configured URL, method, headers, and body template with `{question}` and `{top_k}` placeholders substituted.
3. Response is parsed using simple dot-notation path resolution (e.g. `data.chunks` → `response["data"]["chunks"]`) to locate the results array.
4. Each element in the results array is mapped to LongProbe's standard format (`id`, `text`, `score`, `metadata`) using the configured field names.
5. `health_check()` performs a lightweight request (HEAD or GET to the same URL) and returns `True`/`False`.
6. HTTP adapter is registered in `create_adapter()` factory and in the CLI `_create_adapter_from_config()`.
7. `longprobe init` template includes commented-out HTTP adapter section.

### B. `longprobe generate` Command

8. User can run `longprobe generate <path>` where `<path>` is a file or directory containing documents.
9. Supported file formats: PDF, DOCX, PPTX, XLSX, HTML, TXT, Markdown, and any other text-based format.
10. Extracted document text is sent to an LLM to generate realistic questions.
11. Supports multiple LLM providers: OpenAI, Anthropic, Google Gemini, and local models via Ollama (via `litellm`).
12. Generated questions are output to a YAML file or stdout.
13. `--auto-capture` flag automatically sends each generated question through the retriever and saves results as a golden set (chaining generate → retrieve → golden set).
14. `--num-questions`, `--provider`, `--model`, `--output` flags control generation behavior.
15. Generator configuration is also available via `longprobe.yaml` under a new `generator:` section.

### C. GitHub Actions CI/CD

16. A reusable workflow file (`.github/workflows/longprobe.yml`) that runs `longprobe check --output github`.
17. Workflow posts a PR comment summarizing results (pass/fail, recall scores, regressions).
18. Workflow blocks PR merge when regressions exceed a configurable threshold.
19. Workflow supports all adapter types, including HTTP adapter.

## Out of Scope

- **WebSocket or streaming API support** — HTTP adapter is request/response only.
- **OAuth/token refresh flows** — Users handle token lifecycle via `${ENV_VAR}` in headers.
- **Binary file formats beyond common document types** (e.g., audio, video).
- **Fine-tuning or training custom question generation models.**
- **Publishing a separate GitHub Action to the Marketplace** — We ship a reusable workflow YAML in the repo, not a published action.
- **Custom JSONPath with wildcards/filters** — Simple dot-notation only.
- **Rate limiting or request throttling** for the HTTP adapter — users handle this on their API side.
- **Parallel/async retrieval** — sequential requests only (matches existing pattern).

## Technical Design

### Architecture

```
src/longprobe/
├── adapters/
│   ├── http.py              # NEW: Generic HTTP adapter
│   ├── __init__.py           # MODIFIED: register "http"
│   └── base.py               # unchanged
├── core/
│   ├── docparser.py          # NEW: Document text extraction
│   ├── generator.py          # NEW: LLM question generation
│   ├── golden.py             # unchanged (reused by auto-capture)
│   └── scorer.py             # unchanged
├── cli/
│   └── main.py               # MODIFIED: add generate command, update init/adapters
├── config.py                 # MODIFIED: add HttpRetrieverConfig, GeneratorConfig
└── ...

.github/
└── workflows/
    └── longprobe.yml         # NEW: Reusable CI workflow
```

### Data Models / Schemas

#### HttpResponseMapping (new dataclass)

```python
@dataclass
class HttpResponseMapping:
    results_path: str = "results"   # dot-notation path to array in JSON response
    id_field: str = "id"            # field name for chunk ID
    text_field: str = "text"        # field name for chunk text
    score_field: str = "score"      # field name for relevance score
```

#### HttpRetrieverConfig (new dataclass)

```python
@dataclass
class HttpRetrieverConfig:
    url: str = ""
    method: str = "POST"
    body_template: str = '{"query": "{question}", "top_k": {top_k}}'
    headers: Dict[str, str] = field(default_factory=dict)
    response_mapping: HttpResponseMapping = field(default_factory=HttpResponseMapping)
    timeout: int = 30
```

#### GeneratorConfig (new dataclass)

```python
@dataclass
class GeneratorConfig:
    provider: str = "openai"        # openai, anthropic, gemini, ollama
    model: str = "gpt-4o-mini"
    api_key: str = ""               # supports ${ENV_VAR}
    base_url: str = ""              # for Ollama or custom endpoints
    num_questions: int = 50
    temperature: float = 0.7
    max_tokens: int = 4096
```

#### RetrieverConfig changes

```python
@dataclass
class RetrieverConfig:
    # ... existing fields unchanged ...
    http: HttpRetrieverConfig = field(default_factory=HttpRetrieverConfig)  # NEW
```

#### ProbeConfig changes

```python
@dataclass
class ProbeConfig:
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)  # NEW
```

#### YAML config shape (new sections)

```yaml
# --- HTTP adapter (under existing retriever section) ---
retriever:
  type: http
  http:
    url: "http://localhost:8000/api/retrieve"
    method: POST
    body_template: '{"query": "{question}", "top_k": {top_k}}'
    headers:
      Authorization: "Bearer ${API_KEY}"
    response_mapping:
      results_path: "data.chunks"
      id_field: "chunk_id"
      text_field: "content"
      score_field: "similarity"
    timeout: 30

# --- Generator (new top-level section) ---
generator:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  base_url: ""
  num_questions: 50
  temperature: 0.7
  max_tokens: 4096
```

### API / Interface Changes

#### `HttpAdapter.__init__(config: HttpRetrieverConfig)`

New constructor. Takes the HTTP-specific config. Uses `requests` (stdlib-adjacent, already common).

#### `HttpAdapter.retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]`

1. Substitute `{question}` (JSON-escaped string) and `{top_k}` (integer) into `body_template`.
2. Parse the substituted template as JSON.
3. Make HTTP request: `requests.request(method=method, url=url, json=body, headers=headers, timeout=timeout)`.
4. Parse JSON response.
5. Resolve `results_path` by splitting on `.` and traversing the dict.
6. For each element in the resulting array, extract `id`, `text`, `score` using field names from `response_mapping`.
7. Return normalized list of `{"id": ..., "text": ..., "score": ..., "metadata": ...}`.

#### `HttpAdapter.health_check() -> bool`

Make a HEAD or GET request to the URL. Return `True` if status < 400.

#### `DocumentParser` (new class)

```python
class DocumentParser:
    def parse_file(self, path: str) -> str:
        """Extract text from a single file. Returns empty string for unsupported formats."""

    def parse_directory(self, path: str) -> list[tuple[str, str]]:
        """Parse all supported files recursively. Returns [(filename, text)]."""
```

- Uses `markitdown` library for PDF, DOCX, PPTX, XLSX, HTML, images.
- Falls back to direct text reading for `.txt`, `.md`, `.csv`, `.json`, and other text files.
- If `markitdown` is not installed, only text-based files are supported (with a warning for binary files).

#### `QuestionGenerator` (new class)

```python
class QuestionGenerator:
    def __init__(self, config: GeneratorConfig): ...

    def generate(self, documents: list[tuple[str, str]], num_questions: int) -> list[str]:
        """Generate questions from documents. Returns list of question strings."""
```

- Uses `litellm.completion()` for unified multi-provider access.
- Chunks documents to fit within model context window (~4000 tokens per chunk).
- Merges deduplicated questions across chunks.
- Falls back gracefully if `litellm` is not installed (raises `ImportError` with install instructions).

#### CLI: `generate` command

```
longprobe generate <path> [OPTIONS]

Arguments:
  PATH                  File or directory containing documents

Options:
  --num-questions, -n   Number of questions to generate [default: 50]
  --output, -o          Output file path (YAML). Prints to stdout if not set.
  --provider            LLM provider (overrides config)
  --model               LLM model (overrides config)
  --auto-capture        After generating, send questions through retriever and save golden set
  --match-mode, -m      Match mode for auto-capture [default: text]
  --top-k, -k           Top-k for auto-capture [default: 5]
  --goldens, -g         Golden set file path for auto-capture [default: goldens.yaml]
  --config, -c          Config file path [default: longprobe.yaml]
  --tag                 Tags for auto-captured questions (can be specified multiple times)
```

#### CLI: `_create_adapter_from_config()` changes

Add `rtype == "http"` branch that reads `config.retriever.http` and creates `HttpAdapter(http_config)`.

#### CLI: `init` template changes

Add commented-out HTTP adapter section to `CONFIG_TEMPLATE`.

#### `create_adapter()` factory changes

Add `"http": HttpAdapter` to the adapters dict. The factory passes `http_config=kwargs` or a dedicated kwarg.

### Key Implementation Details

#### HTTP Adapter: Body template substitution

The body template is a JSON string with `{question}` and `{top_k}` placeholders. Implementation:

```python
import json

body_str = body_template.replace('{top_k}', str(top_k))
# For {question}, we need JSON-safe escaping
body_str = body_str.replace('{question}', json.dumps(query)[1:-1])  # strip surrounding quotes from json.dumps
body = json.loads(body_str)
```

Why this approach: Using `json.dumps(query)` ensures proper escaping of quotes, newlines, and unicode in the question text. Stripping the surrounding quotes from `json.dumps` lets us insert the escaped string into the template at the right position.

#### HTTP Adapter: Dot-notation path resolution

Simple recursive resolution without a library:

```python
def _resolve_path(data: Any, path: str) -> Any:
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return []  # path not found, return empty
    return current if isinstance(current, list) else []
```

Why: RAG APIs return predictable, flat structures. Full JSONPath is overkill and adds a dependency.

#### Document parsing strategy

1. **Primary**: Use `markitdown.convert(path)` which handles PDF, DOCX, PPTX, XLSX, HTML, images via a single call.
2. **Fallback**: For `.txt`, `.md`, `.csv`, `.json` and other text-based files, read directly with `open(path).read()`.
3. **Binary fallback**: If `markitdown` is not installed, warn for binary files and skip them.

Why `markitdown`: One dependency covers all major document formats. Lighter than `unstructured` (which pulls in NLP models). Microsoft-maintained.

#### LLM question generation strategy

1. Concatenate all document texts.
2. Split into chunks of ~3000 tokens (rough estimate: 1 token ≈ 4 chars).
3. For each chunk, call `litellm.completion()` with a system prompt asking for N questions.
4. Deduplicate across chunks (simple string similarity or exact match).
5. Return exactly `num_questions` (truncate if too many, re-generate if too few).

Why `litellm`: One dependency provides unified access to OpenAI, Anthropic, Google, Ollama, and 100+ other providers. Avoids writing and maintaining per-provider SDK integration code. Matches the "universal tool" positioning.

#### Auto-capture flow

When `--auto-capture` is used:
1. Generate questions via LLM (same as non-auto mode).
2. Load config and create adapter (reuses existing `_load_config` and `_create_adapter_from_config`).
3. For each question, call `adapter.retrieve(question, top_k)`.
4. Build `GoldenQuestion` objects with retrieved chunks as `required_chunks`.
5. Merge into existing golden set or create new one.
6. Write to `goldens.yaml`.

This reuses the existing `capture` command logic but in non-interactive mode.

#### GitHub Actions workflow design

The reusable workflow:
1. Triggers on `pull_request` events.
2. Installs LongProbe with appropriate extras.
3. Runs `longprobe check --output github` which outputs `::error` / `::notice` annotations.
4. Uses `actions/github-script` to post a PR comment with a summary table.
5. Exits with code 1 if regressions detected (blocks merge with required status checks).

### Dependencies

| Dependency | Why | Install |
|---|---|---|
| `requests` | HTTP adapter needs to make HTTP requests. Lightweight, ubiquitous. | Already common; add to base deps |
| `litellm>=1.0.0` | Unified LLM API for multi-provider question generation. | Optional: `pip install longprobe[generate]` |
| `markitdown>=0.1.0` | Document text extraction for PDF, DOCX, PPTX, HTML, etc. | Optional: `pip install longprobe[generate]` |

New optional dependency group in `pyproject.toml`:
```toml
[project.optional-dependencies]
generate = ["litellm>=1.0.0", "markitdown>=0.1.0"]
http = ["requests>=2.28.0"]
```

## Assumptions

1. `requests` is acceptable as a base dependency (it's already a de-facto standard Python library). If not, it can be made optional.
2. `litellm` and `markitdown` are optional dependencies — `generate` command fails gracefully with install instructions if missing.
3. The HTTP adapter uses `requests` (synchronous). No async support needed.
4. `body_template` is always valid JSON after `{question}` and `{top_k}` substitution. If substitution produces invalid JSON, an error is raised with a clear message.
5. Dot-notation paths in `response_mapping.results_path` are simple key traversals (no array indexing, no wildcards).
6. Questions generated by the LLM are reviewed by the user before being used in production tests (unless `--auto-capture` is used, which implies trust).
7. The GitHub Actions workflow is a reusable workflow committed to the project repo, not a published Marketplace action.
8. For `--auto-capture`, the retriever adapter is already configured in `longprobe.yaml` (same config used by `check` and `capture`).
9. Ollama users are expected to have Ollama running locally and set `generator.base_url: "http://localhost:11434"`.
10. File encoding is assumed UTF-8 for text-based files. Binary files are handled by `markitdown`.

## Tasks

---

### Task 1: Generic HTTP Adapter

**Goal:** Add a new `HttpAdapter` that lets LongProbe test any HTTP-based RAG API endpoint. This is the highest-value feature because it makes LongProbe work with any RAG system, not just direct vector DB connections.

**Scope:**
- Create `src/longprobe/adapters/http.py`
- Modify `src/longprobe/config.py` — add `HttpResponseMapping`, `HttpRetrieverConfig`, update `RetrieverConfig`, update `_build_retriever_config`
- Modify `src/longprobe/adapters/__init__.py` — register `"http"` in factory
- Modify `src/longprobe/cli/main.py` — add HTTP branch to `_create_adapter_from_config()`, add HTTP section to `CONFIG_TEMPLATE`
- Modify `pyproject.toml` — add `requests>=2.28.0` to base dependencies
- Create `tests/unit/test_http_adapter.py`

**Acceptance Criteria:**
- [ ] `HttpAdapter.retrieve("What is X?", 5)` makes a POST to the configured URL with substituted body template and returns normalized `list[dict]` with `id`, `text`, `score`, `metadata` keys
- [ ] Dot-notation path resolution correctly navigates nested JSON (e.g. `"data.chunks"` → `response["data"]["chunks"]`) and returns empty list if path is missing
- [ ] `${ENV_VAR}` expansion works in header values and URL (via existing config loader)
- [ ] `health_check()` returns `True` for 2xx responses, `False` for errors/timeouts
- [ ] `longprobe init` produces a config template with a commented-out HTTP section
- [ ] `longprobe check` works end-to-end with `retriever.type: http` in config
- [ ] Unit tests cover: template substitution, path resolution, response mapping, health check, error handling (non-200, timeout, invalid JSON, missing path)

**Error Handling:**
- Non-200 response → raise `RuntimeError` with status code and response body snippet
- Timeout → raise `requests.Timeout` wrapped in a clear error message
- Invalid JSON in body_template after substitution → raise `ValueError` with the malformed template
- Missing `results_path` in response → log warning, return empty list (not an error — the API may have returned no results)
- Missing field in a result item → use defaults (`id=""`, `text=""`, `score=0.0`, `metadata={}`)

**Verify:**
```bash
cd /home/mohsin/Downloads/longprobe
python -m pytest tests/unit/test_http_adapter.py -v
python -c "from longprobe.adapters import create_adapter; a = create_adapter('http', url='http://httpbin.org/post', method='POST'); print('OK')"
```

---

### Task 2: `generate` Command — Document Parsing & LLM Question Generation

**Goal:** Add the core document parsing and LLM-based question generation infrastructure, plus the CLI command that outputs questions to a file or stdout (without auto-capture).

**Scope:**
- Create `src/longprobe/core/docparser.py` — `DocumentParser` class
- Create `src/longprobe/core/generator.py` — `QuestionGenerator` class
- Modify `src/longprobe/config.py` — add `GeneratorConfig`, update `ProbeConfig`, add `_build_generator_config`
- Modify `src/longprobe/cli/main.py` — add `generate` CLI command (without `--auto-capture` flag yet)
- Modify `pyproject.toml` — add `[project.optional-dependencies] generate = ["litellm>=1.0.0", "markitdown>=0.1.0"]`
- Create `tests/unit/test_docparser.py`
- Create `tests/unit/test_generator.py`

**Acceptance Criteria:**
- [ ] `DocumentParser.parse_file()` extracts text from `.txt`, `.md`, `.csv`, `.json` files via direct read, and from PDF/DOCX/PPTX/HTML via `markitdown` (when installed)
- [ ] `DocumentParser.parse_directory()` recursively finds and parses all supported files, returns `[(filename, text)]`
- [ ] `QuestionGenerator.generate()` calls `litellm.completion()` with the configured provider/model and returns a list of question strings
- [ ] Generator config (`generator:` section) is loadable from `longprobe.yaml` with `${ENV_VAR}` expansion for `api_key`
- [ ] CLI flags `--num-questions`, `--provider`, `--model`, `--output` override config values
- [ ] Without `--output`, questions print to stdout (one per line); with `--output questions.yaml`, questions are saved as a simple YAML list
- [ ] Graceful `ImportError` when `litellm` or `markitdown` is not installed, with install instructions

**Error Handling:**
- Empty directory → print warning, exit 0
- No extractable text from any file → print warning, exit 0
- LLM API error → print error with status/message, exit 1
- Unsupported binary file without `markitdown` → print warning for that file, continue with others
- API key missing → print clear error ("Set OPENAI_API_KEY or configure generator.api_key"), exit 1

**Verify:**
```bash
cd /home/mohsin/Downloads/longprobe
python -m pytest tests/unit/test_docparser.py tests/unit/test_generator.py -v
echo "What is Python?" > /tmp/test_doc.txt
python -m longprobe.cli.main generate /tmp/test_doc.txt --num-questions 3 --provider openai 2>&1 | head -5
# (will fail without API key, but should print a clear error, not a traceback)
```

---

### Task 3: `generate --auto-capture` — Auto-Capture Integration

**Goal:** Add the `--auto-capture` flag to the `generate` command that chains question generation → retriever query → golden set creation. This is the "zero-effort" workflow where a developer runs one command and gets a complete golden set.

**Scope:**
- Modify `src/longprobe/cli/main.py` — add `--auto-capture`, `--match-mode`, `--top-k`, `--goldens`, `--tag` flags to the `generate` command; implement the capture loop using existing adapter and `GoldenSet` infrastructure
- Create `tests/unit/test_generate_capture.py` — test the auto-capture flow with a mock adapter

**Acceptance Criteria:**
- [ ] `longprobe generate ./docs --auto-capture` generates questions, queries the retriever for each, and writes results to `goldens.yaml`
- [ ] Works with any adapter type: `http`, `chroma`, `pinecone`, `qdrant` (config-driven, no adapter-specific code)
- [ ] `--match-mode` flag controls whether chunks are saved by `id` or `text` (same as `capture` command)
- [ ] If `goldens.yaml` already exists, new questions are merged (no duplicates, same as existing `capture` behavior)
- [ ] Progress is shown during capture (one spinner line per question)
- [ ] Final summary shows: N questions generated, N captured, output file path

**Error Handling:**
- Retriever not configured → print error with config instructions, exit 1
- Retriever returns empty results for a question → skip that question with warning, continue
- Golden set write error → print error, exit 1
- Partial failure (some questions succeed, some fail) → save successful ones, report failures in summary

**Verify:**
```bash
cd /home/mohsin/Downloads/longprobe
python -m pytest tests/unit/test_generate_capture.py -v
# Manual smoke test (requires running retriever):
# echo "Test content about refunds." > /tmp/test.txt
# python -m longprobe.cli.main generate /tmp/test.txt --auto-capture --num-questions 3
```

---

### Task 4: GitHub Actions Reusable Workflow

**Goal:** Provide a reusable GitHub Actions workflow that teams can reference from their own repos to run LongProbe regression tests on every PR, with PR comment feedback and merge blocking.

**Scope:**
- Create `.github/workflows/longprobe.yml` — reusable workflow
- Create `examples/ci_example/longprobe-reusable.yml` — example caller workflow
- Update `examples/ci_example/` with a README or update existing example

**Acceptance Criteria:**
- [ ] Reusable workflow at `.github/workflows/longprobe.yml` accepts inputs: `python-version`, `config-path`, `goldens-path`, `fail-on-regression`, `longprobe-version`
- [ ] Workflow runs `longprobe check --output github --config <config> --goldens <goldens>` and produces `::error`/`::notice` annotations visible in the PR checks tab
- [ ] Posts a PR comment with a summary table (pass rate, overall recall, regressions) using `actions/github-script`
- [ ] Exits with code 1 when regressions detected (blocks merge when branch protection requires the status check)
- [ ] Example caller workflow demonstrates how to use it from another repo with `uses: ./.github/workflows/longprobe.yml` or `uses: longprobe/longprobe/.github/workflows/longprobe.yml@main`

**Error Handling:**
- Missing config file → workflow fails with clear error in logs
- Missing goldens file → workflow fails with clear error in logs
- LLM API rate limit during `generate` → not handled in CI (generate is a local command; CI only runs `check`)
- PR comment fails (permissions) → workflow still succeeds/fails based on check results; comment failure is non-blocking

**Verify:**
```bash
cd /home/mohsin/Downloads/longprobe
# Validate workflow YAML syntax
python -c "import yaml; yaml.safe_load(open('.github/workflows/longprobe.yml')); print('Valid YAML')"
python -c "import yaml; yaml.safe_load(open('examples/ci_example/longprobe-reusable.yml')); print('Valid YAML')"
# Verify the workflow references valid longprobe commands
grep -q "longprobe check" .github/workflows/longprobe.yml && echo "Command found"
```

---

## Full Verification

After all tasks are complete:

```bash
cd /home/mohsin/Downloads/longprobe

# 1. Run full test suite
python -m pytest tests/ -v

# 2. Verify HTTP adapter works (unit tests mock requests)
python -m pytest tests/unit/test_http_adapter.py -v

# 3. Verify generate command loads without errors
python -c "from longprobe.cli.main import app; print('CLI OK')"

# 4. Verify config parsing with new sections
python -c "
from longprobe.config import ProbeConfig
c = ProbeConfig.from_dict({
    'retriever': {'type': 'http', 'http': {'url': 'http://localhost:8000/api'}},
    'generator': {'provider': 'openai', 'model': 'gpt-4o-mini'}
})
print(f'Retriever: {c.retriever.type}, HTTP URL: {c.retriever.http.url}')
print(f'Generator: {c.generator.provider}, Model: {c.generator.model}')
"

# 5. Verify init produces new template
rm -f /tmp/lp_test_longprobe.yaml
cd /tmp && python -m longprobe.cli.main init --force
grep -q "type: http" /tmp/longprobe.yaml && echo "HTTP template present"
grep -q "generator:" /tmp/longprobe.yaml && echo "Generator template present"

# 6. Validate GitHub Actions workflow
python -c "import yaml; yaml.safe_load(open('/home/mohsin/Downloads/longprobe/.github/workflows/longprobe.yml')); print('Workflow valid')"
```
