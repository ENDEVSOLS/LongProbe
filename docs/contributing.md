# Contributing to LongProbe

Thank you for your interest in contributing to LongProbe! This guide will help you get started.

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/LongProbe.git
cd LongProbe
```

### 2. Install Dependencies

```bash
# Install with development dependencies
uv sync --dev

# Or with pip
pip install -e ".[dev]"
```

### 3. Run Tests

```bash
# Run unit tests
uv run pytest tests/unit/ -v

# Run all tests including integration
uv run pytest tests/ -v --run-integration

# Run with coverage
uv run pytest --cov=longprobe --cov-report=html
```

### 4. Lint and Format

```bash
# Check code style
uv run ruff check src/

# Auto-fix issues
uv run ruff check --fix src/

# Format code
uv run ruff format src/
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clear, concise code
- Add tests for new features
- Update documentation as needed
- Follow existing code style

### 3. Test Your Changes

```bash
# Run tests
uv run pytest tests/

# Check linting
uv run ruff check src/

# Format code
uv run ruff format src/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature"
# or
git commit -m "fix: resolve bug in X"
```

Use conventional commit messages:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

We use Ruff for linting and formatting:

```bash
# Check style
uv run ruff check src/

# Format code
uv run ruff format src/
```

## Testing Guidelines

### Writing Tests

```python
# tests/unit/test_feature.py
import pytest
from longprobe import LongProbe

def test_feature():
    """Test description."""
    # Arrange
    probe = LongProbe(...)
    
    # Act
    result = probe.some_method()
    
    # Assert
    assert result == expected
```

### Running Specific Tests

```bash
# Run specific test file
uv run pytest tests/unit/test_scorer.py -v

# Run specific test
uv run pytest tests/unit/test_scorer.py::test_recall_calculation -v

# Run with markers
uv run pytest -m "not slow" -v
```

## Documentation

### Building Docs Locally

```bash
# Install docs dependencies
pip install mkdocs-material mkdocstrings[python]

# Serve docs locally
mkdocs serve

# Build docs
mkdocs build
```

### Writing Documentation

- Use clear, simple language
- Include code examples
- Add links to related pages
- Test all code examples

## Pull Request Process

1. **Update Documentation** - If you add features, update docs
2. **Add Tests** - New features need tests
3. **Pass CI** - All tests must pass
4. **Code Review** - Address reviewer feedback
5. **Squash Commits** - Clean up commit history if needed

## Release Process

(For maintainers)

1. Update `CHANGELOG.md`
2. Bump version in `pyproject.toml`
3. Create git tag: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. GitHub Actions will build and create release

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/ENDEVSOLS/LongProbe/issues)
- **Discussions**: [Ask questions](https://github.com/ENDEVSOLS/LongProbe/discussions)

## Code of Conduct

Be respectful and inclusive. We're all here to build great tools together.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
