# Installation

## Requirements

- Python 3.9 or higher
- pip or uv package manager

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
longprobe --version
```

You should see output like:

```
LongProbe version 0.1.0
```

## Next Steps

- [Quick Start Guide](quick-start.md) - Get started in 5 minutes
- [Configuration](configuration.md) - Configure your vector store and embeddings
