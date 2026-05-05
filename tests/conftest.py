"""
Shared pytest configuration for LongProbe test suite.

Provides custom markers and command-line options for controlling test
execution (e.g. integration tests that require external services).
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers with the pytest configuration."""
    config.addinivalue_line(
        "markers", "integration: integration tests (may require external services)"
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options for the test suite.

    Options:
        --run-integration: Also run tests marked with ``@pytest.mark.integration``.
            By default these are *skipped* to keep CI fast.
    """
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that may require external services",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip integration tests unless ``--run-integration`` is provided."""
    if not config.getoption("--run-integration", default=False):
        skip_integration = pytest.mark.skip(
            reason="need --run-integration option to run"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
