<div align="center">

<p align="center"><img src="https://raw.githubusercontent.com/ENDEVSOLS/LongProbe/main/assets/longProbe-with-bg.png" alt="LongProbe Logo" width="320"/></p>

**Sub-second RAG regression testing for production pipelines**

[![PyPI version](https://badge.fury.io/py/longprobe.svg)](https://badge.fury.io/py/longprobe)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/longprobe?period=total&units=international_system&left_color=black&right_color=green&left_text=downloads)](https://pepy.tech/projects/longprobe)
[![Python Versions](https://img.shields.io/pypi/pyversions/longprobe.svg)](https://pypi.org/project/longprobe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/ENDEVSOLS/LongProbe/workflows/LongProbe%20CI/badge.svg)](https://github.com/ENDEVSOLS/LongProbe/actions)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://endevsols.github.io/LongProbe)

[Quick Start](#quick-start) • [Documentation](https://endevsols.github.io/LongProbe) • [Python API](#python-api) • [CI/CD](#github-actions)

</div>

---

## Overview

> "Did my last commit break retrieval?" — now you know in seconds.

LongProbe is a **sub-second RAG regression harness**. Define your Golden Questions once, run `longprobe check` on every commit, and get an exact diff of which document chunks were lost in your latest change — before your users notice.

**Think `pytest --watch` for your RAG pipeline.**

## 🎬 Demos

### Complete Workflow
Full RAG regression testing workflow: auto-capture golden questions, run tests, save baseline, detect regressions.

![Complete Workflow](https://raw.githubusercontent.com/ENDEVSOLS/LongProbe/main/assets/01-complete-workflow.gif)

### Monitor RAG Quality
Detailed quality monitoring with Python API and comprehensive results.

![Monitor RAG Quality](https://raw.githubusercontent.com/ENDEVSOLS/LongProbe/main/assets/02-python-api.gif)

### Detect Regressions
Baseline comparison and regression detection with deployment verdict.

![Detect Regressions](https://raw.githubusercontent.com/ENDEVSOLS/LongProbe/main/assets/03-baseline-tracking.gif)

## Why LongProbe?

Every RAG developer faces the same silent killer: you refactor chunking strategy, upgrade LangChain, or add a new document — and your retrieval silently degrades. DeepEval and RAGChecker are heavyweight evaluation frameworks meant for batch analysis, not fast regression checks in a dev loop.

**LongProbe gives you instant feedback:**
- ⚡ **Sub-second checks** on small golden sets
- 🔍 **Exact diffs** showing which chunks were lost/gained
- 📊 **Recall scores** with per-question breakdown
- 💾 **Baseline tracking** to catch regressions over time
- 🧪 **pytest integration** for existing test suites
- 🔌 **Pluggable adapters** for any vector store

## Part of the Long Suite

LongProbe is part of the [EnDevSols Long Suite](https://endevsols.com/open-source) of RAG tools:

- **[LongParser](https://github.com/ENDEVSOLS/LongParser)** - Document ingestion and chunking
- **[LongTrainer](https://github.com/ENDEVSOLS/Long-Trainer)** - RAG chatbot framework
- **[LongTracer](https://github.com/ENDEVSOLS/LongTracer)** - Hallucination detection
- **[LongProbe](https://github.com/ENDEVSOLS/LongProbe)** - Retrieval regression testing ← You are here

Together they cover the full RAG pipeline from ingestion to production monitoring.

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
# Install with UV (recommended)
uv pip install longprobe

# Install with pip
pip install longprobe

# Install with optional dependencies
uv pip install longprobe[chroma]      # ChromaDB support
uv pip install longprobe[openai]      # OpenAI embeddings
uv pip install longprobe[all]         # Everything
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
      - "The following officers are authorized to sign"
    top_k: 10
```

### Configure Your Retriever

Edit `longprobe.yaml`:

```yaml
retriever:
  type: "chroma"
  chroma:
    persist_directory: "./chroma_db"
    collection: "my_documents"

embedder:
  provider: "local"
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

### Core Commands

| Command | Description |
|---------|-------------|
| `longprobe init` | Create starter configuration files |
| `longprobe check` | Run probes against the golden set |
| `longprobe diff` | Compare current results against baseline |
| `longprobe baseline save` | Save current results as baseline |
| `longprobe baseline list` | List all saved baselines |
| `longprobe watch` | Watch golden file and re-run on changes |
| `longprobe generate` | Auto-generate Golden Questions from documents |
| `longprobe capture` | Build goldens.yaml by querying your retriever |

### Examples

```bash
# Initialize project
longprobe init

# Run checks with custom config
longprobe check -g goldens.yaml -c longprobe.yaml

# Save baseline for comparison
longprobe baseline save --label v1.0

# Compare against baseline
longprobe diff --baseline v1.0

# Watch mode for development
longprobe watch --interval 2

# Generate questions from documents
longprobe generate ./docs --capture --auto
```

## Python API

### Basic Usage

```python
from longprobe import LongProbe
from longprobe.adapters import create_adapter

# Create adapter for your vector store
adapter = create_adapter(
    "chroma",
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

print(f"Overall Recall: {report.overall_recall:.2%}")
print(f"Pass Rate: {report.pass_rate:.2%}")
```

### Baseline Management

```python
from longprobe import LongProbe
from longprobe.adapters import create_adapter

adapter = create_adapter("chroma", collection_name="docs", persist_directory="./db")
probe = LongProbe(adapter=adapter, goldens_path="goldens.yaml")

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

## Pytest Integration

### Configuration

```python
# conftest.py
import pytest
from longprobe import LongProbe
from longprobe.adapters import create_adapter

@pytest.fixture
def probe():
    adapter = create_adapter(
        "chroma",
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
        f"Recall dropped to {report.overall_recall:.2f}"
    )

def test_no_regression_vs_baseline(probe):
    report = probe.run()
    assert not report.regression_detected, (
        f"Regression detected! Delta: {report.recall_delta}"
    )
```

## Retriever Adapters

LongProbe supports multiple vector stores and retrieval frameworks:

| Adapter | Type | Configuration |
|---------|------|---------------|
| **ChromaDB** | Direct | `type: chroma` |
| **Pinecone** | Direct | `type: pinecone` |
| **Qdrant** | Direct | `type: qdrant` |
| **HTTP API** | Direct | `type: http` |
| **LangChain** | Programmatic | `LangChainRetrieverAdapter` |
| **LlamaIndex** | Programmatic | `LlamaIndexRetrieverAdapter` |

### ChromaDB Example

```yaml
retriever:
  type: chroma
  collection: my_collection
  persist_directory: ./chroma_db
```

### HTTP API Example

```yaml
retriever:
  type: http
  url: "http://localhost:8000/api/retrieve"
  method: "POST"
  body_template: '{"query": "{question}"}'
  response_mapping:
    results_path: "data.chunks"
    text_field: "content"
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

## Match Modes

### ID Match (`match_mode: "id"`)
Exact string match on chunk/document IDs. Best when you control the IDs in your vector store.

### Text Match (`match_mode: "text"`)
Case-insensitive substring matching. Checks if the required text appears anywhere in the retrieved documents.

### Semantic Match (`match_mode: "semantic"`)
Word-frequency cosine similarity. Useful when exact text may vary but meaning should be preserved.

## Development

```bash
# Install for development
git clone https://github.com/ENDEVSOLS/LongProbe.git
cd LongProbe
uv sync --dev

# Run tests
uv run pytest tests/unit/ -v
uv run pytest tests/ -v --run-integration

# Lint and format
uv run ruff check src/
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

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

For security issues, please see [SECURITY.md](SECURITY.md).

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

[Website](https://endevsols.com) • [GitHub](https://github.com/ENDEVSOLS) • [Documentation](https://github.com/ENDEVSOLS/LongProbe)

</div>
