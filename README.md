# LongProbe 🔬

[![PyPI version](https://badge.fury.io/py/longprobe.svg)](https://badge.fury.io/py/longprobe)
[![Python Versions](https://img.shields.io/pypi/pyversions/longprobe.svg)](https://pypi.org/project/longprobe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/ENDEVSOLS/LongProbe/workflows/LongProbe%20CI/badge.svg)](https://github.com/ENDEVSOLS/LongProbe/actions)

> "Did my last commit break retrieval?" — now you know in seconds.

LongProbe is a sub-second RAG regression harness. Define your Golden Questions once.
Run `longprobe check` on every commit. Get an exact diff of which document chunks
were lost in your latest change — before your users notice.

## Part of the Long Suite

LongProbe is part of the [EnDevSols Long Suite](https://endevsols.com/open-source) of RAG tools:

- **[LongParser](https://github.com/ENDEVSOLS/LongParser)** - Document ingestion and chunking
- **[LongTrainer](https://github.com/ENDEVSOLS/Long-Trainer)** - RAG chatbot framework
- **[LongTracer](https://github.com/ENDEVSOLS/LongTracer)** - Hallucination detection
- **[LongProbe](https://github.com/ENDEVSOLS/LongProbe)** - Retrieval regression testing ← You are here

Together they cover the full RAG pipeline from ingestion to production monitoring.

## Why LongProbe?

Every RAG developer faces the same silent killer: you refactor chunking strategy,
upgrade LangChain, or add a new document — and your retrieval silently degrades.
DeepEval and RAGChecker are heavyweight evaluation frameworks meant for batch analysis,
not fast regression checks in a dev loop.

**LongProbe gives you `pytest --watch` for your RAG pipeline.**

## Features

- ⚡ **Sub-second checks** on small golden sets
- 📋 **Golden Questions + Required Chunks** defined in simple YAML
- 🔍 **Three match modes**: exact ID, text substring, semantic similarity
- 📊 **Recall Score** with per-question breakdown
- 🔄 **Regression diff**: exactly which chunks were lost/gained
- 💾 **SQLite baseline store**: compare against any previous run
- 🧪 **pytest plugin**: integrate into existing test suites
- 🔌 **Pluggable adapters**: LangChain, LlamaIndex, Chroma, Pinecone, Qdrant
- 🖥️ **Beautiful CLI** with Rich tables, JSON, and GitHub Actions output
- 👀 **Watch mode**: auto re-run on file changes
- 🏗️ **CI/CD ready**: fails pipeline on regression

## Quick Start

### Installation

```bash
# Install with UV
uv pip install longprobe

# Install with pip
pip install longprobe

# Install with optional dependencies
uv pip install longprobe[chroma]      # ChromaDB support
uv pip install longprobe[openai]      # OpenAI embeddings
uv pip install longprobe[all]         # Everything
uv pip install longprobe[chroma,openai]  # Specific extras

# Install for development
git clone https://github.com/ENDEVSOLS/LongProbe.git
cd LongProbe
uv sync --dev
```

### Initialize

```bash
longprobe init
```

This creates:
- `.longprobe/` — directory for baseline storage
- `goldens.yaml` — example golden questions
- `longprobe.yaml` — configuration file

### Define Golden Questions

Edit `goldens.yaml` with your test cases:

```yaml
name: "my-rag-golden-set"
version: "1.0"

questions:
  - id: "q1"
    question: "What is the termination clause?"
    match_mode: "id"            # exact chunk ID match
    required_chunks:
      - "contracts_chunk_42"
      - "contracts_chunk_43"
    top_k: 5
    tags: ["contracts", "critical"]

  - id: "q2"
    question: "What are the payment terms?"
    match_mode: "text"          # substring match
    required_chunks:
      - "net 30 days from invoice"
    top_k: 5

  - id: "q3"
    question: "Who can sign contracts?"
    match_mode: "semantic"      # embedding similarity
    semantic_threshold: 0.80
    required_chunks:
      - "The following officers are authorized to sign on behalf of the company"
    top_k: 10
```

### Configure Your Retriever

Edit `longprobe.yaml`:

```yaml
retriever:
  type: "chroma"                 # Or "http" to test a RAG API
  chroma:
    persist_directory: "./chroma_db"
    collection: "my_documents"
  # http:
  #   url: "http://localhost:8000/api/retrieve"
  #   method: "POST"
  #   body_template: '{"query": "{question}"}'
  #   response_mapping:
  #     results_path: "data.chunks"
  #     text_field: "content"

embedder:
  provider: "local"              # openai | huggingface | local
  model: "text-embedding-3-small"

scoring:
  recall_threshold: 0.8
  fail_on_regression: true

baseline:
  db_path: ".longprobe/baselines.db"
  auto_compare: true
```

### Run Checks

```bash
# Run against live vector store
longprobe check --goldens goldens.yaml

# Override settings
longprobe check --threshold 0.9 --top-k 10

# JSON output for automation
longprobe check --output json

# GitHub Actions annotations
longprobe check --output github
```

## CLI Reference

### `longprobe init`
Create starter configuration files.
```bash
longprobe init          # Create goldens.yaml and longprobe.yaml
longprobe init --force  # Overwrite existing files
```

### `longprobe generate`
Automatically generate Golden Questions by analyzing your documents with an LLM.
```bash
longprobe generate ./docs                    # Read markdown/PDFs and save to questions.txt
longprobe generate ./docs --capture --auto   # Generate AND automatically save the chunks
```

### `longprobe capture`
Build your `goldens.yaml` file by automatically querying your retriever.
```bash
longprobe capture -q "What is the refund policy?"        # Interactive mode
longprobe capture --auto --questions-file questions.txt  # Auto-save whatever is retrieved
longprobe capture --auto -q "What is X?" --tag doc:legal # Scope the test to a tag
```

### `longprobe check`
Run probes against the golden set.
```bash
longprobe check                                    # Use defaults
longprobe check -g goldens.yaml -c longprobe.yaml  # Specify files
longprobe check -o json                            # JSON output
longprobe check -o github                          # GitHub Actions format
longprobe check -o table                           # Rich table (default)
longprobe check -k 10                              # Override top_k
longprobe check -t 0.9                             # Override threshold
```

### `longprobe baseline save`
Save current results as a named baseline.
```bash
longprobe baseline save                  # Save as "latest"
longprobe baseline save --label v1.2     # Save with custom label
```

### `longprobe baseline list`
List all saved baselines.
```bash
longprobe baseline list
```

### `longprobe baseline delete`
Delete a saved baseline.
```bash
longprobe baseline delete --label v1.2
```

### `longprobe diff`
Compare current results against a saved baseline.
```bash
longprobe diff                          # Compare against "latest"
longprobe diff --baseline v1.2          # Compare against specific label
longprobe diff --output json            # JSON diff output
```

### `longprobe watch`
Watch golden file and re-run on changes.
```bash
longprobe watch                         # 2s interval (default)
longprobe watch --interval 5            # 5s interval
```

## Match Modes

### ID Match (`match_mode: "id"`)
Exact string match on chunk/document IDs. Best when you control the IDs in your vector store.
```yaml
- id: "q1"
  question: "What is X?"
  match_mode: "id"
  required_chunks:
    - "doc_a_chunk_3"
    - "doc_b_chunk_7"
```

### Text Match (`match_mode: "text"`)
Case-insensitive substring matching. Checks if the required text appears anywhere in the retrieved documents.
```yaml
- id: "q2"
  question: "What are the payment terms?"
  match_mode: "text"
  required_chunks:
    - "net 30 days from invoice"
```

### Semantic Match (`match_mode: "semantic"`)
Word-frequency cosine similarity. Useful when exact text may vary but meaning should be preserved.
```yaml
- id: "q3"
  question: "Who can authorize payments?"
  match_mode: "semantic"
  semantic_threshold: 0.80
  required_chunks:
    - "Only the CFO and CEO may authorize payments exceeding $10,000"
```

## Python API

### Basic Usage

```python
from longprobe import LongProbe
from longprobe.adapters import ChromaAdapter

# Create adapter for your vector store
adapter = ChromaAdapter(
    collection_name="my_documents",
    persist_directory="./chroma_db"
)

# Create and run probe
probe = LongProbe(
    adapter=adapter,
    goldens_path="goldens.yaml",
    config_path="longprobe.yaml"
)
report = probe.run()

print(f"Overall Recall: {report.overall_recall:.2f}")
print(f"Pass Rate: {report.pass_rate:.2f}")

# Check missing chunks
missing = probe.get_missing_chunks()
for q_id, chunks in missing.items():
    print(f"  {q_id}: {chunks}")
```

### With LangChain

```python
from longprobe import LongProbe
from longprobe.adapters import LangChainRetrieverAdapter

# Wrap your existing LangChain retriever
adapter = LangChainRetrieverAdapter(your_langchain_retriever)

probe = LongProbe(adapter=adapter, goldens_path="goldens.yaml")
report = probe.run()

assert report.overall_recall >= 0.85, f"Recall too low: {report.overall_recall}"
```

### With LlamaIndex

```python
from longprobe import LongProbe
from longprobe.adapters import LlamaIndexRetrieverAdapter

adapter = LlamaIndexRetrieverAdapter(your_llamaindex_retriever)
probe = LongProbe(adapter=adapter, goldens_path="goldens.yaml")
report = probe.run()
```

### Baseline Management

```python
from longprobe import LongProbe, ChromaAdapter

probe = LongProbe(
    adapter=ChromaAdapter(collection_name="docs", persist_directory="./db"),
    goldens_path="goldens.yaml"
)

# Run and save baseline
report = probe.run()
probe.save_baseline(label="v1.0")

# After making changes...
report2 = probe.run()

# Compare against baseline
diff = probe.diff(baseline_label="v1.0")
print(f"Regressions: {len(diff['regressions'])}")
print(f"Improvements: {len(diff['improvements'])}")
```

## Pytest Integration

### Configuration

Install the pytest plugin (it auto-registers via entry points):

```python
# conftest.py
import pytest
from longprobe import LongProbe
from longprobe.adapters import ChromaAdapter

@pytest.fixture
def probe():
    adapter = ChromaAdapter(
        collection_name="test_docs",
        persist_directory="./test_db"
    )
    return LongProbe(
        adapter=adapter,
        goldens_path="tests/goldens.yaml",
        recall_threshold=0.85
    )
```

### Writing Tests

```python
def test_retrieval_recall(probe):
    report = probe.run()
    assert report.overall_recall >= 0.85, (
        f"Recall dropped to {report.overall_recall:.2f}. "
        f"Lost chunks: {probe.get_missing_chunks()}"
    )

def test_critical_questions_found(probe):
    report = probe.run()
    missing = probe.get_missing_chunks()
    critical_missing = {
        q_id: chunks for q_id, chunks in missing.items()
        if any("critical" in tag for tag in
               next(r.tags for r in report.results if r.question_id == q_id))
    }
    assert not critical_missing, f"Critical chunks missing: {critical_missing}"

def test_no_regression_vs_baseline(probe):
    report = probe.run()
    assert not report.regression_detected, (
        f"Regression detected! Delta: {report.recall_delta}"
    )
```

### Pytest CLI Options

```bash
pytest --longprobe-goldens goldens.yaml --longprobe-config longprobe.yaml
pytest --longprobe-fail-threshold 0.85
```

## Retriever Adapters

### ChromaDB (Direct)
```yaml
retriever:
  type: chroma
  collection: my_collection
  persist_directory: ./chroma_db
```

### Pinecone (Direct)
```yaml
retriever:
  type: pinecone
  index_name: my-index
  api_key: ${PINECONE_API_KEY}
  namespace: ""
```

### Qdrant (Direct)
```yaml
retriever:
  type: qdrant
  collection: my_collection
  host: localhost
  port: 6333
  api_key: ${QDRANT_API_KEY}
```

### LangChain (Programmatic)
```python
from longprobe.adapters import LangChainRetrieverAdapter
adapter = LangChainRetrieverAdapter(your_retriever)
```

### LlamaIndex (Programmatic)
```python
from longprobe.adapters import LlamaIndexRetrieverAdapter
adapter = LlamaIndexRetrieverAdapter(your_retriever)
```

## GitHub Actions

```yaml
name: RAG Regression Check

on: [push, pull_request]

jobs:
  rag-probe:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv pip install longprobe[chroma]
      - name: Run RAG regression check
        run: longprobe check --goldens goldens.yaml --output github
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Configuration Reference

| Section | Field | Type | Default | Description |
|---------|-------|------|---------|-------------|
| `retriever` | `type` | string | `"chroma"` | Adapter type |
| `retriever` | `collection` | string | `""` | Collection name |
| `retriever` | `persist_directory` | string | `""` | Local DB path |
| `retriever` | `index_name` | string | `""` | Pinecone index |
| `retriever` | `host` | string | `""` | Qdrant host |
| `retriever` | `port` | int | `6333` | Qdrant port |
| `retriever` | `api_key` | string | `""` | API key (supports `${ENV_VAR}`) |
| `embedder` | `provider` | string | `"openai"` | Embedding provider |
| `embedder` | `model` | string | `"text-embedding-3-small"` | Model name |
| `embedder` | `dimensions` | int | `0` | Embedding dimensions |
| `scoring` | `recall_threshold` | float | `0.8` | Min recall to pass |
| `scoring` | `fail_on_regression` | bool | `true` | Exit 1 on regression |
| `baseline` | `db_path` | string | `.longprobe/baselines.db` | SQLite path |
| `baseline` | `auto_compare` | bool | `true` | Auto-compare vs baseline |

## Development

```bash
# Install for development
uv sync --dev

# Run unit tests
uv run pytest tests/unit/ -v

# Run all tests including integration
uv run pytest tests/ -v --run-integration

# Lint
uv run ruff check src/

# Format
uv run ruff format src/
```

## How It Works

```
goldens.yaml → GoldenLoader → QueryEmbedder → RetrieverAdapter → RecallScorer
                                                                      ↓
                                                                BaselineStore → DiffReporter
```

1. **Define** your Golden Questions + Required Fact Chunks in YAML
2. **Embed** each question using your configured embedding model
3. **Retrieve** from your live vector store using the pluggable adapter
4. **Score** each question by checking if required chunks appear in Top-K results
5. **Compare** against saved baselines to detect regressions
6. **Report** a Recall Score, diff of lost chunks, and optionally fail CI/CD

## License

MIT License — see [LICENSE](LICENSE) for details.
