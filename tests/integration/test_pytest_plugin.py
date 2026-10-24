"""Run installed pytest entry-point fixtures in isolated user projects."""

import os
import subprocess
import sys

import pytest
import yaml

from longprobe.core.golden import GoldenQuestion, GoldenSet

pytestmark = pytest.mark.integration


@pytest.fixture
def project(tmp_path):
    GoldenSet("suite", "1", [GoldenQuestion("q1", "Q1", ["a", "b"])]).to_yaml(
        str(tmp_path / "cases.yaml")
    )
    config = {
        "retriever": {"type": "http", "http": {"url": "http://configured.example/retrieve"}},
        "scoring": {"recall_threshold": 0.4},
        "baseline": {"auto_compare": False, "db_path": str(tmp_path / "b.db")},
    }
    (tmp_path / "selected.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "conftest.py").write_text("""
from pathlib import Path
import requests

class Response:
    status_code = 200
    def json(self):
        return {"results": [{"id": "a", "text": "Evidence"}]}

def fake_request(**kwargs):
    assert kwargs["url"] == "http://configured.example/retrieve"
    with Path("calls.txt").open("a") as stream:
        stream.write("call\\n")
    return Response()

def pytest_sessionstart(session):
    requests.request = fake_request
""")
    return tmp_path


def run_project(project, source, *args, configured=True):
    (project / "test_user.py").write_text(source)
    env = dict(os.environ)
    for key in ("PYTEST_ADDOPTS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        env.pop(key, None)
    command = [sys.executable, "-m", "pytest", "-q", "test_user.py"]
    if configured:
        command += ["--longprobe-config", "selected.yaml", "--longprobe-goldens", "cases.yaml"]
    return subprocess.run(
        [*command, *args], cwd=project, env=env, text=True, capture_output=True, timeout=30
    )


def test_default_adapter_and_report_fixture_run_once_with_selected_config(project):
    result = run_project(
        project,
        """
def test_one(longprobe_report):
    assert longprobe_report.results[0].passed
    assert longprobe_report.results[0].contract["recall_threshold"] == .4

def test_two(longprobe_report):
    assert longprobe_report.overall_recall == .5
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (project / "calls.txt").read_text() == "call\n"
    assert "LongProbe Results" in result.stdout


def test_enforcement_runs_without_user_requesting_report(project):
    result = run_project(
        project, "def test_regular(): assert True", "--longprobe-fail-threshold", ".4"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (project / "calls.txt").read_text() == "call\n"


def test_enforcement_does_not_skip_failed_evaluation(project):
    result = run_project(
        project, "def test_regular(): assert True", "--longprobe-fail-threshold", ".8"
    )
    assert result.returncode != 0
    assert "LongProbe quality gate failed" in result.stdout
    assert "skipped" not in result.stdout


def test_unused_plugin_does_not_load_config_or_retrieve(project):
    (project / "selected.yaml").unlink()
    result = run_project(project, "def test_regular(): assert True", configured=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (project / "calls.txt").exists()
    assert not (project / "b.db").exists()


def test_invalid_requested_config_fails_instead_of_skipping(project):
    (project / "selected.yaml").write_text("scoring: []")
    result = run_project(project, "def test_report(longprobe_report): assert longprobe_report")
    assert result.returncode != 0
    assert "skipped" not in result.stdout
    assert not (project / "calls.txt").exists()


def test_custom_adapter_needs_no_unrelated_provider_credentials(project):
    config = yaml.safe_load((project / "selected.yaml").read_text())
    config["retriever"] = {"type": "pinecone", "api_key": "${UNUSED_PINECONE_KEY}"}
    (project / "selected.yaml").write_text(yaml.safe_dump(config))
    with (project / "conftest.py").open("a") as stream:
        stream.write("""
import pytest
@pytest.fixture(scope="session")
def longprobe_adapter():
    class Adapter:
        def retrieve(self, query, top_k):
            return [{"id": "a"}, {"id": "b"}]
    return Adapter()
""")
    result = run_project(
        project, "def test_report(longprobe_report): assert longprobe_report.pass_rate == 1"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (project / "calls.txt").exists()


def test_missing_selected_config_with_custom_adapter_does_not_use_python_defaults(project):
    (project / "selected.yaml").unlink()
    with (project / "conftest.py").open("a") as stream:
        stream.write("""
import pytest
@pytest.fixture(scope="session")
def longprobe_adapter():
    class Adapter:
        def retrieve(self, query, top_k):
            raise AssertionError("must not retrieve")
    return Adapter()
""")
    result = run_project(project, "def test_report(longprobe_report): assert longprobe_report")
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stdout
    assert not (project / "calls.txt").exists()


@pytest.mark.parametrize("threshold", ["nan", "1.2", "-0.1"])
def test_invalid_threshold_fails_before_retrieval(project, threshold):
    result = run_project(
        project, "def test_regular(): assert True", "--longprobe-fail-threshold", threshold
    )
    assert result.returncode == 4  # pytest usage error, not a CLI exit category.
    assert not (project / "calls.txt").exists()


def test_empty_suite_with_explicit_enforcement_is_not_success(project):
    GoldenSet("suite", "1", []).to_yaml(str(project / "cases.yaml"))
    result = run_project(
        project, "def test_regular(): assert True", "--longprobe-fail-threshold", ".8"
    )
    assert result.returncode != 0
    assert "No questions" in result.stdout
    assert not (project / "calls.txt").exists()
