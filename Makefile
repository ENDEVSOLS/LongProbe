.PHONY: help install test lint format demos clean setup-demos

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

setup-demos: ## Setup demo environment (run this first!)
	@echo "Setting up demo environment..."
	@bash demos/setup_demo_env.sh
	@echo "Warming up (downloading models)..."
	@source demos/.demo_venv/bin/activate && cd demo_run && longprobe check --config longprobe.yaml > /dev/null 2>&1 || true
	@echo "✓ Demo environment ready!"

demos: ## Generate all demo GIFs (requires vhs and setup-demos)
	@echo "Generating demos..."
	@command -v vhs >/dev/null 2>&1 || { echo "Error: vhs not installed. Run: brew install vhs"; exit 1; }
	@test -d demos/.demo_venv || { echo "Error: Demo environment not set up. Run: make setup-demos"; exit 1; }
	vhs demos/01-quick-start.tape
	vhs demos/02-diff-output.tape
	vhs demos/03-json-output.tape
	@echo "✓ Demos generated in demos/ directory"

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
