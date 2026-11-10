# Installation

## Requirements and Verified Support Matrix

LongProbe runs on Python 3.11 and 3.12 across standard operating systems. Support tiers below reflect environments verified in continuous integration:

### Verified Platform Matrix

| Environment Component | Verified Tier | Status & Verification |
|-----------------------|---------------|-----------------------|
| **Python 3.11** | Tier 1 (CI Verified) | Fully tested and supported across all unit and contract tests |
| **Python 3.12** | Tier 1 (CI Verified) | Primary target runtime for testing and release packaging |
| **Python 3.13** | Community | Not yet gated in mandatory CI; experimental |
| **Ubuntu Linux (x86_64)** | Tier 1 (CI Verified) | Verified on Ubuntu 22.04 LTS and 24.04 LTS GitHub runners |
| **macOS (Darwin x86_64 / arm64)** | Tier 2 (Compatible) | Compatible via standard POSIX runtime; unverified in automated CI gates |
| **Windows (x86_64)** | Tier 2 (Compatible) | Path separators normalized; unverified in automated CI gates |

### Adapter Verification Matrix

| Adapter / Backend | Package Extra | Offline CI Status | Requirements / Notes |
|-------------------|---------------|-------------------|----------------------|
| **HTTP (Generic REST API)** | *(built-in)* | :white_check_mark: Verified | No extras needed; built into base package |
| **ChromaDB** | `longprobe[chroma]` | :white_check_mark: Verified | Verified with deterministic offline embedding fixtures |
| **LangChain** | `longprobe[langchain]` | :white_check_mark: Verified | Contract and adapter tests verified |
| **LlamaIndex** | `longprobe[llamaindex]` | :white_check_mark: Verified | Contract and adapter tests verified |
| **Pinecone** | `longprobe[pinecone]` | :warning: Unverified in offline CI | Requires live cloud API key and network connectivity |
| **Qdrant** | `longprobe[qdrant]` | :warning: Unverified in offline CI | Requires live container or cluster endpoint |

*Note: Live cloud services and unverified environments require customer integration testing in staging prior to production gating.*

## Install the v0.2.0 candidate

Version 0.2.0 is implemented locally but has not been published by this work. The package-index commands below install an available published version; they do not guarantee these new features.

From the repository root containing `pyproject.toml`, choose one installation method:

```bash
pip install -e .
# Alternatively, after building the candidate locally:
pip install dist/longprobe-0.2.0-py3-none-any.whl
```

See [development installation](#development-installation) and the [build verification record](../releases/v0.2.0-verification.md). Confirm the installed version under [verify installation](#verify-installation).

## Install with pip

```bash
pip install longprobe
```

## Install with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer:

```bash
uv pip install longprobe
```

## Optional Dependencies

LongProbe supports various vector stores and embedding providers through optional dependencies:

### ChromaDB Support

```bash
pip install longprobe[chroma]
# or
uv pip install longprobe[chroma]
```

### OpenAI Embeddings

```bash
pip install longprobe[openai]
```

### Pinecone Support

```bash
pip install longprobe[pinecone]
```

### Qdrant Support

```bash
pip install longprobe[qdrant]
```

### All Optional Dependencies

```bash
pip install longprobe[all]
```

### Specific Combinations

```bash
pip install longprobe[chroma,openai]
```

## Development Installation

For contributing to LongProbe:

```bash
# Clone the repository
git clone https://github.com/ENDEVSOLS/LongProbe.git
cd LongProbe

# Install with development dependencies
uv sync --dev

# Or with pip
pip install -e ".[dev]"
```

## Verify Installation

```bash
python -c 'import longprobe; print(longprobe.__version__)'
longprobe --help
```

Python 3.10 and older are not supported. Package installers enforce the Python 3.11 minimum.

## Next Steps

- [Quick Start Guide](quick-start.md) - Get started in 5 minutes
- [Configuration](configuration.md) - Configure your vector store and embeddings
