# Contributing to LongProbe

Thank you for your interest in contributing to LongProbe! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/LongProbe.git
   cd LongProbe
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ENDEVSOLS/LongProbe.git
   ```

## Development Setup

LongProbe uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Set up development environment

```bash
# Install dependencies
uv sync --dev

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### Install optional dependencies

```bash
# For specific adapters
uv sync --extra chroma
uv sync --extra openai
uv sync --extra all  # Install everything
```

## Making Changes

1. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following our [code style guidelines](#code-style)

3. **Write or update tests** for your changes

4. **Update documentation** if needed (README, docstrings, etc.)

## Testing

### Run all tests

```bash
# Unit tests only
uv run pytest tests/unit/ -v

# All tests including integration
uv run pytest tests/ -v --run-integration
```

### Run specific test files

```bash
uv run pytest tests/unit/test_scorer.py -v
```

### Run with coverage

```bash
uv run pytest tests/ --cov=src/longprobe --cov-report=html
```

### Run linting

```bash
# Check code style
uv run ruff check src/

# Auto-fix issues
uv run ruff check src/ --fix

# Format code
uv run ruff format src/
```

## Submitting Changes

1. **Commit your changes** with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "feat: add support for new vector store"
   ```

   Use conventional commit prefixes:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Test changes
   - `refactor:` - Code refactoring
   - `chore:` - Maintenance tasks

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub:
   - Go to the [LongProbe repository](https://github.com/ENDEVSOLS/LongProbe)
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill in the PR template with details about your changes
   - Link any related issues

### Pull Request Guidelines

- **Keep PRs focused** - One feature or fix per PR
- **Write clear descriptions** - Explain what and why, not just how
- **Update tests** - Ensure all tests pass
- **Update documentation** - Keep docs in sync with code changes
- **Follow code style** - Run linting before submitting
- **Be responsive** - Address review feedback promptly

## Code Style

LongProbe follows Python best practices and uses Ruff for linting and formatting.

### Key Guidelines

- **Python 3.10+** syntax and features
- **Type hints** for all function signatures
- **Docstrings** for all public modules, classes, and functions (Google style)
- **Line length** - 100 characters maximum
- **Imports** - Organized and sorted (Ruff handles this)
- **Naming conventions**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Example

```python
from typing import List, Dict, Optional

def score_retrieval(
    question: str,
    retrieved_docs: List[Dict[str, str]],
    threshold: float = 0.8,
) -> Optional[float]:
    """Score retrieval quality against expected results.
    
    Args:
        question: The query string to evaluate.
        retrieved_docs: List of retrieved document dictionaries.
        threshold: Minimum score to consider successful.
        
    Returns:
        Recall score between 0.0 and 1.0, or None if evaluation fails.
        
    Raises:
        ValueError: If threshold is not between 0.0 and 1.0.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be 0.0-1.0, got {threshold}")
    
    # Implementation here
    return 0.95
```

## Reporting Bugs

Found a bug? Please [open an issue](https://github.com/ENDEVSOLS/LongProbe/issues/new) with:

- **Clear title** - Summarize the issue
- **Description** - What happened vs. what you expected
- **Steps to reproduce** - Minimal example to reproduce the bug
- **Environment** - Python version, OS, LongProbe version
- **Error messages** - Full stack traces if applicable
- **Screenshots** - If relevant

### Bug Report Template

```markdown
**Description**
A clear description of the bug.

**To Reproduce**
1. Step 1
2. Step 2
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., macOS 14.0]
- Python: [e.g., 3.12.0]
- LongProbe: [e.g., 0.1.0]

**Additional Context**
Any other relevant information.
```

## Suggesting Features

Have an idea? [Open a feature request](https://github.com/ENDEVSOLS/LongProbe/issues/new) with:

- **Use case** - What problem does this solve?
- **Proposed solution** - How should it work?
- **Alternatives** - Other approaches you considered
- **Examples** - Code examples or mockups if applicable

## Development Tips

### Testing with different vector stores

```bash
# ChromaDB
uv sync --extra chroma
uv run pytest tests/integration/test_chroma_adapter.py

# All integrations
uv sync --all-extras
uv run pytest tests/ --run-integration
```

### Testing CLI commands

```bash
# Install in development mode
uv pip install -e .

# Test CLI
longprobe --help
longprobe check --goldens examples/goldens.yaml
```

### Debugging

```python
# Add breakpoints
import pdb; pdb.set_trace()

# Or use pytest with pdb
uv run pytest tests/unit/test_scorer.py -v --pdb
```

## Questions?

- **Documentation**: Check the [README](README.md)
- **Issues**: Search [existing issues](https://github.com/ENDEVSOLS/LongProbe/issues)
- **Contact**: opensource@endevsols.com

## License

By contributing to LongProbe, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to LongProbe! 🚀
