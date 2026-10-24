.PHONY: help install test lint format demos clean setup-demos check-demos

help: ## Show this help message
	@echo "LongProbe - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install development dependencies
	uv sync --all-extras

test: ## Run tests
	pytest tests/ -v

lint: ## Run linting
	ruff check src/ tests/

format: ## Format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

setup-demos: ## Install this checkout for self-contained demo recordings
	@bash demos/setup_demo_env.sh

check-demos: ## Execute demo scenarios without recording pauses
	demos/.recording_venv/bin/python demo_run/complete_workflow.py --no-delay
	demos/.recording_venv/bin/python demo_run/monitor_rag_quality.py --no-delay
	demos/.recording_venv/bin/python demo_run/detect_regressions.py --no-delay
	demos/.recording_venv/bin/python demo_run/test_rag_retrieval.py --no-delay

demos: ## Record and synchronize GIFs (requires VHS, ttyd, Chromium, and ffmpeg)
	@command -v vhs >/dev/null 2>&1 || { echo "VHS is required; see demos/README.md"; exit 1; }
	@test -x demos/.recording_venv/bin/python || { echo "Run make setup-demos first"; exit 1; }
	vhs demos/01-complete-workflow.tape
	vhs demos/01-quick-check.tape
	vhs demos/02-python-api.tape
	vhs demos/03-baseline-tracking.tape
	cp demos/01-complete-workflow.gif demos/01-quick-check.gif demos/02-python-api.gif demos/03-baseline-tracking.gif assets/
	cp demos/01-complete-workflow.gif demos/01-quick-check.gif demos/02-python-api.gif demos/03-baseline-tracking.gif docs/assets/
	@echo "Demo GIFs synchronized to README and documentation assets."

clean: ## Clean build artifacts
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build package
	uv build

publish-test: build ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	twine upload dist/*
