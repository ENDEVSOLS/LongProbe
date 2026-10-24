"""Opt-in pytest fixtures for shared LongProbe evaluation.

Use ``longprobe_report`` to evaluate once per pytest session. The enforcement
option activates that fixture automatically; unrelated pytest runs stay idle.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from longprobe.core.scorer import ProbeReport

_REPORT_KEY: pytest.StashKey[ProbeReport] = pytest.StashKey()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("longprobe")
    group.addoption(
        "--longprobe-goldens", default="goldens.yaml", help="Golden questions YAML path."
    )
    group.addoption(
        "--longprobe-config", default="longprobe.yaml", help="LongProbe configuration YAML path."
    )
    group.addoption(
        "--longprobe-fail-threshold",
        type=float,
        default=None,
        help="Run LongProbe and enforce per-question passing with this global threshold; explicit question thresholds take precedence.",
    )
    group.addoption(
        "--longprobe-include-drafts",
        action="store_true",
        default=False,
        help="Include drafts in pytest analysis (not baseline promotion).",
    )


def pytest_configure(config: pytest.Config) -> None:
    threshold = config.getoption("longprobe_fail_threshold")
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise pytest.UsageError("--longprobe-fail-threshold must be a finite number in [0, 1].")


@pytest.fixture(scope="session")
def longprobe_goldens_path(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("longprobe_goldens")


@pytest.fixture(scope="session")
def longprobe_config_path(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("longprobe_config")


@pytest.fixture(scope="session")
def longprobe_fail_threshold(request: pytest.FixtureRequest) -> float | None:
    return request.config.getoption("longprobe_fail_threshold")


@pytest.fixture(scope="session")
def longprobe_include_drafts(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("longprobe_include_drafts")


@pytest.fixture(scope="session")
def longprobe_adapter(longprobe_config_path: str):
    """Build the configured adapter; override this fixture for an application retriever."""
    from longprobe.adapters import create_adapter_from_config
    from longprobe.config import ProbeConfig

    return create_adapter_from_config(ProbeConfig.from_yaml(longprobe_config_path))


@pytest.fixture(scope="session")
def longprobe_probe(
    longprobe_adapter,
    longprobe_goldens_path: str,
    longprobe_config_path: str,
    longprobe_include_drafts: bool,
    longprobe_fail_threshold: float | None,
):
    from longprobe import LongProbe
    from longprobe.config import ProbeConfig

    # A selected pytest path is required even for a custom application adapter.
    ProbeConfig.from_yaml(longprobe_config_path)
    return LongProbe(
        adapter=longprobe_adapter,
        goldens_path=longprobe_goldens_path,
        config_path=longprobe_config_path,
        include_drafts=longprobe_include_drafts,
        recall_threshold=longprobe_fail_threshold,
    )


@pytest.fixture(scope="session")
def longprobe_report(longprobe_probe, request: pytest.FixtureRequest):
    """Return one completed evaluation; no private config assignment is required."""
    report = longprobe_probe.run()
    request.config.stash[_REPORT_KEY] = deepcopy(report)
    return report


@pytest.fixture(scope="session", autouse=True)
def _longprobe_enforce(request: pytest.FixtureRequest):
    if request.config.getoption("longprobe_fail_threshold") is None:
        return
    report = request.getfixturevalue("longprobe_report")
    if not report.quality["passed"]:
        pytest.fail(
            "LongProbe quality gate failed: " + "; ".join(report.quality["reasons"]), pytrace=False
        )


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    report = config.stash.get(_REPORT_KEY, None)
    if report is None:
        return
    terminalreporter.write_sep("=", "LongProbe Results")
    terminalreporter.write_line(
        f"Overall recall: {report.overall_recall:.2%}; pass rate: {report.pass_rate:.2%}"
    )
    identity = report.baseline_identity
    terminalreporter.write_line(
        f"Baseline: {identity.get('resolved_label') or identity.get('requested_label') or 'none'}; comparison: {identity.get('status', 'not_requested')}"
    )
    for result in report.results:
        terminalreporter.write_line(
            f"{result.question_id}: {'PASS' if result.passed else 'FAIL'}; missing: {', '.join(result.missing_chunks) or 'none'}"
        )
