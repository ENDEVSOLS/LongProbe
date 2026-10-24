"""JSON stdout, explicit references, and execution/quality exit categories."""

import json
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

from longprobe.cli.main import app
from longprobe.core.baseline import BaselineStore
from longprobe.core.golden import GoldenQuestion, GoldenSet


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    GoldenSet(
        "policies",
        "1",
        [
            GoldenQuestion("q1", "Q1", ["a", "b"]),
            GoldenQuestion("draft", "Draft", ["a"], status="draft"),
        ],
    ).to_yaml("goldens.yaml")
    config = {
        "retriever": {"type": "http", "http": {"url": "http://localhost/retrieve"}},
        "baseline": {"db_path": str(tmp_path / "b.db")},
    }
    path = tmp_path / "longprobe.yaml"
    path.write_text(yaml.safe_dump(config))
    adapter = Mock()
    adapter.retrieve.return_value = [{"id": "a"}, {"id": "b"}]
    factory = Mock(return_value=adapter)
    monkeypatch.setattr("longprobe.cli.main._create_adapter_from_config", factory)
    return CliRunner(), adapter, factory, config, path


def test_clean_json_keeps_draft_notices_on_stderr(project):
    runner, adapter, _, _, _ = project
    result = runner.invoke(app, ["check", "--output", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["baseline_identity"]["status"] == "not_found"
    assert "draft question(s) excluded" in result.stderr
    assert adapter.retrieve.call_count == 1


@pytest.mark.parametrize(
    "args",
    [
        ["check", "--tag", "unknown"],
        ["check", "--threshold", "2"],
        ["check", "--top-k", "0"],
        ["check", "--config", "missing.yaml"],
        ["check", "--baseline", "missing"],
        ["diff", "--baseline", "missing"],
    ],
)
def test_setup_failures_are_one_json_object_and_no_retrieval(project, args):
    runner, adapter, _, _, _ = project
    result = runner.invoke(app, [*args, "--output", "json"])
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["status"] == "error"
    adapter.retrieve.assert_not_called()


def test_quality_relaxation_does_not_relabel_failed_result(project):
    runner, adapter, _, config, path = project
    config["scoring"] = {"fail_on_regression": False}
    path.write_text(yaml.safe_dump(config))
    adapter.retrieve.return_value = [{"id": "a"}]
    result = runner.invoke(app, ["check", "--output", "json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "failed"
    missing = runner.invoke(app, ["check", "--baseline", "missing", "--output", "json"])
    assert missing.exit_code == 2


def test_retrieval_exception_is_execution_error_without_secret(project):
    runner, adapter, _, _, _ = project
    adapter.retrieve.side_effect = ValueError("secret-token-never-print")
    result = runner.invoke(app, ["check", "--output", "json"])
    assert result.exit_code == 3
    assert json.loads(result.stdout)["error"]["code"] == "execution_error"
    assert "secret-token-never-print" not in result.output


def test_rejected_baseline_save_is_parseable_and_preserves_reference(project):
    runner, adapter, _, config, _ = project
    assert runner.invoke(app, ["baseline", "save", "--output", "json"]).exit_code == 0
    store = BaselineStore(config["baseline"]["db_path"])
    reference = store.load()
    adapter.retrieve.return_value = [{"id": "a"}]
    result = runner.invoke(app, ["baseline", "save", "--output", "json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["quality"]["failed_questions"] == ["q1"]
    assert store.load() == reference


def test_explicit_reference_and_stored_diff_use_same_contract(project):
    runner, adapter, _, _, _ = project
    assert runner.invoke(app, ["baseline", "save", "--label", "release-a"]).exit_code == 0
    assert runner.invoke(app, ["baseline", "save", "--label", "release-b"]).exit_code == 0
    explicit = runner.invoke(app, ["check", "--baseline", "release-a", "--output", "json"])
    assert explicit.exit_code == 0
    assert json.loads(explicit.stdout)["baseline_identity"]["resolved_label"] == "release-a"
    calls = adapter.retrieve.call_count
    stored = runner.invoke(app, ["diff", "release-a", "release-b", "--output", "json"])
    assert stored.exit_code == 0, stored.output
    assert json.loads(stored.stdout)["comparison"]["compatible"]
    assert adapter.retrieve.call_count == calls


def test_incompatible_comparison_stays_nonzero_with_relaxed_quality(project):
    runner, _adapter, _, config, path = project
    runner.invoke(app, ["baseline", "save"])
    suite = GoldenSet.from_yaml("goldens.yaml")
    suite.questions[0].required_chunks = ["a"]
    suite.to_yaml("goldens.yaml")
    config["scoring"] = {"fail_on_regression": False}
    path.write_text(yaml.safe_dump(config))
    result = runner.invoke(app, ["check", "--output", "json"])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["comparison"]["changed_questions"]
    assert payload["recall_delta"] is None


def test_saved_report_renders_without_config_or_retrieval(project, tmp_path):
    runner, adapter, factory, _, path = project
    result = runner.invoke(app, ["check", "--output", "json"])
    saved = tmp_path / "result.json"
    saved.write_text(result.stdout)
    path.unlink()
    factory.reset_mock()
    calls = adapter.retrieve.call_count
    for output in ("json", "github", "table"):
        rendered = runner.invoke(app, ["report", str(saved), "--output", output])
        assert rendered.exit_code == 0, rendered.output
    assert adapter.retrieve.call_count == calls
    factory.assert_not_called()


def test_report_and_mode_errors_are_handled_before_retrieval(project, tmp_path):
    runner, adapter, factory, _, _ = project
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version":99}')
    result = runner.invoke(app, ["report", str(bad), "--output", "json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "error"
    mode = runner.invoke(app, ["check", "--output", "invalid"])
    assert mode.exit_code == 2
    adapter.retrieve.assert_not_called()
    factory.assert_not_called()


def test_parser_usage_failure_uses_stderr_and_exit_two(project):
    runner, adapter, _, _, _ = project
    result = runner.invoke(app, ["check", "--top-k", "not-an-integer", "--output", "json"])
    assert result.exit_code == 2
    assert result.stdout == ""
    adapter.retrieve.assert_not_called()
