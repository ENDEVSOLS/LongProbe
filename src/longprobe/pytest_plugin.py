"""
LongProbe pytest plugin.

Registers pytest options and provides fixtures for RAG regression testing.

Usage in conftest.py::

    from longprobe import LongProbe, ChromaAdapter

    @pytest.fixture
    def probe():
        adapter = ChromaAdapter(collection_name="my_docs", persist_directory="./db")
        return LongProbe(adapter=adapter, goldens_path="goldens.yaml")

Usage in tests::

    def test_retrieval_recall(probe):
        report = probe.run()
        assert report.overall_recall >= 0.85, (
            f"Recall dropped to {report.overall_recall:.2f}. "
            f"Lost chunks: {probe.get_missing_chunks()}"
        )

Command line::

    pytest --longprobe-goldens goldens.yaml --longprobe-config longprobe.yaml
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser
    from _pytest.fixtures import FixtureRequest
    from _pytest.nodes import Item


# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------


def pytest_addoption(parser: Parser) -> None:
    """Register longprobe-specific pytest command line options."""
    group = parser.getgroup("longprobe")
    group.addoption(
        "--longprobe-goldens",
        action="store",
        dest="longprobe_goldens",
        default="goldens.yaml",
        help="Path to LongProbe golden questions YAML file (default: goldens.yaml)",
    )
    group.addoption(
        "--longprobe-config",
        action="store",
        dest="longprobe_config",
        default="longprobe.yaml",
        help="Path to LongProbe configuration YAML file (default: longprobe.yaml)",
    )
    group.addoption(
        "--longprobe-fail-threshold",
        action="store",
        dest="longprobe_fail_threshold",
        type=float,
        default=None,
        help=(
            "Minimum overall recall to pass. "
            "Fail tests if recall drops below this value."
        ),
    )


# ---------------------------------------------------------------------------
# Configuration hook
# ---------------------------------------------------------------------------


@pytest.hookimpl(trylast=True)
def pytest_configure(config: Config) -> None:
    """Store longprobe options on the config object for later access."""
    config._longprobe_goldens = config.getoption("longprobe_goldens", "goldens.yaml")  # type: ignore[attr-defined]
    config._longprobe_config = config.getoption("longprobe_config", "longprobe.yaml")  # type: ignore[attr-defined]
    config._longprobe_fail_threshold = config.getoption("longprobe_fail_threshold", None)  # type: ignore[attr-defined]
    config._longprobe_report = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def longprobe_goldens_path(request: FixtureRequest) -> str:
    """Fixture that returns the path to the golden questions file.

    The path is determined by the ``--longprobe-goldens`` CLI option (defaults
    to ``goldens.yaml``).
    """
    return request.config._longprobe_goldens  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def longprobe_config_path(request: FixtureRequest) -> str:
    """Fixture that returns the path to the longprobe config file.

    The path is determined by the ``--longprobe-config`` CLI option (defaults
    to ``longprobe.yaml``).
    """
    return request.config._longprobe_config  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def longprobe_fail_threshold(request: FixtureRequest) -> float | None:
    """Fixture that returns the configured fail threshold (or ``None``).

    Set via ``--longprobe-fail-threshold``.  When *None*, no automatic
    threshold check is applied.
    """
    return request.config._longprobe_fail_threshold  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Optional convenience fixture – auto-build a LongProbe instance
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def longprobe_adapter(
    longprobe_config_path: str,
) -> Any:
    """Lazily build an adapter from the longprobe configuration file.

    Override this fixture in your ``conftest.py`` if you need a custom adapter
    (e.g. ``ChromaAdapter``, ``PineconeAdapter``, etc.) or different
    initialisation logic.
    """
    try:
        from longprobe.config import load_config
    except ImportError:
        pytest.skip("longprobe is not installed; skipping adapter fixture")
        return None  # pragma: no cover – unreachable, satisfies type checker

    try:
        cfg = load_config(longprobe_config_path)
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"Cannot load longprobe config: {exc}")
        return None  # pragma: no cover

    adapter_cls = cfg.get("adapter_class")
    if adapter_cls is None:
        pytest.skip(
            "No adapter_class specified in longprobe config; "
            "provide a custom longprobe_adapter fixture instead"
        )
        return None  # pragma: no cover

    # Resolve dotted-path strings, e.g. "longprobe.ChromaAdapter"
    parts = adapter_cls.rsplit(".", 1)
    if len(parts) == 2:
        import importlib

        mod_path, attr_name = parts
        try:
            mod = importlib.import_module(mod_path)
            adapter_cls = getattr(mod, attr_name)
        except (ImportError, AttributeError) as exc:
            pytest.skip(f"Cannot resolve adapter class '{adapter_cls}': {exc}")
            return None  # pragma: no cover

    # Forward remaining config keys as kwargs to the adapter constructor.
    adapter_kwargs = {
        k: v for k, v in cfg.items() if k not in ("adapter_class",)
    }
    return adapter_cls(**adapter_kwargs)


@pytest.fixture(scope="session")
def longprobe_probe(
    longprobe_adapter: Any,
    longprobe_goldens_path: str,
) -> Any:
    """Return a fully-initialised :class:`~longprobe.LongProbe` instance.

    This fixture depends on :fixture:`longprobe_adapter` which, by default,
    reads adapter settings from the config file.  Override either fixture in
    ``conftest.py`` for full control.
    """
    try:
        from longprobe import LongProbe
    except ImportError:
        pytest.skip("longprobe is not installed; skipping probe fixture")
        return None  # pragma: no cover

    if longprobe_adapter is None:
        pytest.skip("No adapter available; skipping probe fixture")
        return None  # pragma: no cover

    return LongProbe(adapter=longprobe_adapter, goldens_path=longprobe_goldens_path)


# ---------------------------------------------------------------------------
# Automatic threshold enforcement
# ---------------------------------------------------------------------------


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    session: Any,
    config: Config,
    items: list[Item],
) -> None:
    """Optionally inject a final summarising test that enforces the threshold.

    A synthetic test item is appended to the collected list when
    ``--longprobe-fail-threshold`` is provided.  The test reads
    ``config._longprobe_report`` (which a user's test is expected to store)
    and fails if the overall recall is below the threshold.
    """
    threshold = config._longprobe_fail_threshold  # type: ignore[attr-defined]
    if threshold is None:
        return

    from _pytest.python import Function

    # Define the test function inline.
    def _longprobe_threshold_check() -> None:
        report = config._longprobe_report  # type: ignore[attr-defined]
        if report is None:
            pytest.fail(
                "No LongProbe report was generated.  Make sure at least one "
                "test calls ``probe.run()`` and stores the result in "
                "``request.config._longprobe_report``."
            )
            return  # pragma: no cover
        if report.overall_recall < threshold:
            pytest.fail(
                f"LongProbe overall recall ({report.overall_recall:.2f}) is below "
                f"the configured threshold ({threshold:.2f})."
            )

    # Build a lightweight synthetic Function item.
    synthetic = Function.from_parent(
        parent=session,  # type: ignore[arg-type]
        name="longprobe_threshold_check",
        callobj=_longprobe_threshold_check,
    )
    items.append(synthetic)


# ---------------------------------------------------------------------------
# Session-finish summary
# ---------------------------------------------------------------------------


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Print a LongProbe summary table if a report was generated.

    The report is expected to be stored on the config object by user tests::

        request.config._longprobe_report = probe.run()
    """
    report = session.config._longprobe_report  # type: ignore[attr-defined]
    if report is None:
        return

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        # Fall back to plain-text output when rich is unavailable.
        _print_plain_summary(report)
        return

    console = Console()

    table = Table(title="LongProbe Results", show_header=True, header_style="bold cyan")
    table.add_column("Question ID", style="dim")
    table.add_column("Recall", justify="right")
    table.add_column("Missing Chunks")
    table.add_column("Status")

    for result in report.results:
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        chunks_str = ", ".join(result.missing_chunks) if result.missing_chunks else "-"
        table.add_row(
            result.question_id,
            f"{result.recall_score:.2f}",
            chunks_str,
            status,
        )

    console.print()
    console.print(table)
    console.print(
        Panel(
            f"Overall Recall: {report.overall_recall:.2f} | "
            f"Pass Rate: {report.pass_rate:.2f}",
            style="bold",
        )
    )


# ---------------------------------------------------------------------------
# Plain-text fallback summary (when rich is not installed)
# ---------------------------------------------------------------------------


def _print_plain_summary(report: Any) -> None:
    """Print a minimal plain-text summary of a LongProbe report.

    This fallback is used when the ``rich`` library is not installed.
    It relies only on the Python standard library.
    """
    divider = "=" * 72
    print()
    print(divider)
    print("  LongProbe Results")
    print(divider)
    header = f"  {'Question ID':<30} {'Recall':>8}  {'Missing Chunks':<24} {'Status'}"
    print(header)
    print("-" * 72)
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        chunks_str = ", ".join(result.missing_chunks) if result.missing_chunks else "-"
        line = f"  {result.question_id:<30} {result.recall_score:>8.2f}  {chunks_str:<24} {status}"
        print(line)
    print(divider)
    print(
        f"  Overall Recall: {report.overall_recall:.2f}  |  "
        f"Pass Rate: {report.pass_rate:.2f}"
    )
    print(divider)
    print()
