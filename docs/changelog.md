# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added
- Versioned evaluation contracts and shared JSON/SQLite report validation; legacy rows remain readable without automatic rewriting.
- Explicit per-question thresholds, baseline quality policy, reference identity, and coverage/requirement/evidence differences.
- `check --baseline LABEL`, `check --no-compare`, JSON baseline-save output, and read-only `report PATH` rendering.
- Supported session-scoped `longprobe_report` pytest fixture and automatic opt-in quality enforcement.
- Trusted baseline artifact restoration and one-evaluation CI feedback with retained diagnostics.

### Changed
- CLI and Python share evaluation and baseline promotion; the default requires every evaluated approved question to pass.
- Equal-recall loss of required evidence is a regression. Changed contracts/coverage and metadata-free references cannot certify strict comparison.
- Parsed JSON commands emit one stdout document; handled failures use safe error envelopes and documented exit categories.
- Reusable CI honors package version selection and only relaxes completed quality failures when explicitly configured.

### Migration
- Back up old baseline databases and explicitly establish reviewed new-format references; no SQL table migration is required.
- Global threshold overrides preserve explicit per-question thresholds. Aggregate promotion requires an explicit policy selection.
- Adopt `longprobe_report` instead of private pytest config assignments; old named-decorator/rank-delta changelog claims are not implemented interfaces.
- See [v0.2.0 release notes](releases/v0.2.0.md) for format, exit-code, fixture, and baseline changes. Hosted CI/publication remain pending.

## [0.1.3] - Local reliability milestone (unpublished)

### Fixed
- Runtime version now derives from installed package metadata; base startup keeps optional SDKs lazy and missing SDK calls give installation guidance.
- Invalid CLI configuration fails explicitly. Strict shapes, scalar types/ranges, and active environment credentials are validated before service calls; deliberate Python default configuration remains supported.
- Empty evaluations fail across the scorer, Python facade, and CLI; failed reruns invalidate older reports for baseline saving.
- Golden writes validate content and use same-directory atomic replacement, preserving existing data on ordinary preparation/write failures.
- Auto-capture validates existing goldens before generation and never resets an unreadable or invalid set. Initialization preserves files unless forced.

### Added
- Explicit `generate --output PATH --overwrite` for question-list replacement; incompatible output/auto-capture flags fail preflight.
- Controlled HTTP workflow and failure-injection regression tests across supported Python versions.
- Local HTTP quick-start fixture, accurate configuration examples, and recovery/known-limitations documentation.

### Compatibility
- Minimum Python version is now 3.11; Python 3.10 support and its CI job have been removed.
- Empty selections and invalid settings that previously passed or fell back to defaults now fail.
- Use an installed editable checkout for development; the runtime version requires distribution metadata.
- No baseline database/schema migration. Embedding integration, baseline-policy unification, default pytest fixture repair, and general JSON-output cleanup remain deferred.

## [0.1.2] - 2026-08-11

> Historical development entry. Evaluation, baseline, pytest, and CI consistency was completed locally in the v0.2.0 candidate; hosted acceptance remains pending.

### Added
- **Draft vs. Approved Status Schema**: Added `status` field (`draft` / `approved`) to `GoldenQuestion`. Auto-captured questions start in `draft` state so unvetted questions do not pollute regression suites.
- **Interactive TUI Editor (`longprobe edit`)**: Terminal editor using Rich to browse, filter, edit, and approve draft golden questions into trusted ground truth.
- **Pytest Plugin (`pytest-longprobe`)**: Early pytest integration. For v0.2.0 use the `longprobe_report` fixture; there is no supported `@longprobe_check` decorator.
- **Strict Baseline Quality Gating (`longprobe baseline save`)**: Prevents saving low-recall or degraded probe reports as trusted baselines (`--threshold`), evaluating strictly on approved questions.
- **Two-Baseline Comparison (`longprobe diff <a` `<b>`)**: Compare any two historical snapshot baselines side-by-side with `--threshold` quality verification.
- **Diagnostic Summaries (`longprobe explain`)**: Human-readable missing-evidence analysis. Rank deltas and distance drops are not implemented.
- **CLI `--include-drafts` Option**: Flag on `check` and live `diff` to evaluate draft questions on demand.

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
