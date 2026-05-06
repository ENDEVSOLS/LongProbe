# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-05-06

### Added
- Professional MkDocs documentation site with Material theme
- Three live demo GIFs showcasing key features:
  - Test RAG Retrieval (quick validation)
  - Monitor RAG Quality (detailed analysis)
  - Detect Regressions (baseline comparison)
- Comprehensive documentation structure:
  - Getting Started guides (Installation, Quick Start, Configuration)
  - User Guide (Golden Questions, Match Modes, CLI Reference, Python API, Baseline Management)
  - Integration guides (Vector Stores, Pytest, CI/CD, LangChain, LlamaIndex)
  - Demo pages with detailed explanations
  - API Reference
  - Contributing guide
- GitHub Actions workflow for automatic documentation deployment
- Custom CSS for improved documentation UI
- Professional README with logo, badges, and demo GIFs

### Changed
- Updated README with centered logo and GitHub raw URLs for images
- Improved documentation navigation with collapsible sidebar sections
- Enhanced logo visibility in documentation header

### Fixed
- Logo display issues in documentation
- Asset paths for GitHub Pages compatibility

## [0.1.0] - 2026-05-05

### Added
- Initial release of LongProbe
- Golden question management with YAML configuration
- Three match modes: ID, text, and semantic similarity
- Baseline storage and regression detection using SQLite
- CLI with multiple output formats (table, JSON, GitHub Actions)
- Pytest plugin integration for test suite integration
- Retriever adapters:
  - ChromaDB direct adapter
  - Pinecone direct adapter
  - Qdrant direct adapter
  - LangChain retriever wrapper
  - LlamaIndex retriever wrapper
  - HTTP/REST API adapter with configurable mapping
- Question generation with LLM support (OpenAI, Anthropic, Gemini, Ollama)
- Auto-capture functionality to build golden sets from live retrievers
- Watch mode for continuous testing during development
- GitHub Actions reusable workflow for CI/CD integration
- Comprehensive test suite with 172 unit tests
- Rich CLI output with color-coded tables and progress indicators
- Diff reporting showing exactly which chunks were lost/gained
- Per-question recall scoring with configurable thresholds
- Support for question tags and metadata
- Environment variable expansion in configuration files
- Document parsing for multiple formats (TXT, MD, CSV, JSON, PDF, DOCX, PPTX, XLSX)

### Documentation
- Comprehensive README with quick start guide
- CLI reference documentation
- Python API examples
- Configuration reference
- Match mode explanations
- GitHub Actions integration examples
- Example configuration files and golden sets

[Unreleased]: https://github.com/ENDEVSOLS/LongProbe/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/ENDEVSOLS/LongProbe/releases/tag/v0.1.1
[0.1.0]: https://github.com/ENDEVSOLS/LongProbe/releases/tag/v0.1.0
