# LongProbe Documentation

This directory contains the source files for LongProbe's documentation, built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## Building Documentation Locally

### Install Dependencies

```bash
# Install docs dependencies
pip install mkdocs mkdocs-material mkdocstrings[python]

# Or with uv
uv pip install mkdocs mkdocs-material "mkdocstrings[python]"
```

### Serve Locally

```bash
# Start development server
mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

### Build Static Site

```bash
# Build documentation
mkdocs build

# Output will be in site/ directory
```

## Documentation Structure

```
docs/
├── index.md                    # Home page
├── getting-started/
│   ├── installation.md         # Installation guide
│   ├── quick-start.md          # Quick start tutorial
│   └── configuration.md        # Configuration reference
├── guide/
│   ├── golden-questions.md     # Writing golden questions
│   ├── match-modes.md          # Match mode explanations
│   ├── cli-reference.md        # CLI documentation
│   ├── python-api.md           # Python API guide
│   └── baseline-management.md  # Baseline tracking
├── integrations/
│   ├── vector-stores.md        # Vector store adapters
│   ├── pytest.md               # Pytest integration
│   ├── cicd.md                 # CI/CD setup
│   ├── langchain.md            # LangChain integration
│   └── llamaindex.md           # LlamaIndex integration
├── demos/
│   ├── overview.md             # Demos overview
│   ├── test-retrieval.md       # Demo 1
│   ├── monitor-quality.md      # Demo 2
│   └── detect-regressions.md   # Demo 3
├── api/
│   ├── core.md                 # Core API reference
│   ├── adapters.md             # Adapter API
│   └── scoring.md              # Scoring API
├── contributing.md             # Contribution guide
└── changelog.md                # Version history
```

## Writing Documentation

### Style Guide

- Use clear, simple language
- Include code examples
- Add links to related pages
- Use admonitions for important notes
- Test all code examples

### Code Examples

````markdown
```python
from longprobe import LongProbe

probe = LongProbe(...)
report = probe.run()
```
````

### Admonitions

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a tip.
```

### Links

```markdown
[Link text](../path/to/page.md)
[External link](https://example.com)
```

## Deployment

Documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch.

See `.github/workflows/docs.yml` for the deployment workflow.

## Contributing

When adding new features:

1. Update relevant documentation pages
2. Add code examples
3. Test locally with `mkdocs serve`
4. Submit PR with documentation changes

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [MkDocstrings](https://mkdocstrings.github.io/)
