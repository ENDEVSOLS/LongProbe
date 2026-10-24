# Installation

## Requirements

- Python 3.11 or higher
- pip or uv package manager

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
