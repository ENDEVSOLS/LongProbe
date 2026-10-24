"""CI restoration, installation inputs, and verdicts without GitHub access."""

import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock

import pytest
import yaml

from longprobe.ci import (
    determine_exit,
    main,
    markdown_report,
    restore_baseline,
    run_evaluation,
    validate_baseline_inputs,
)
from longprobe.config import ConfigurationError, ProbeConfig
from longprobe.core.baseline import BaselineStore
from longprobe.core.evaluation import evaluate
from longprobe.core.golden import GoldenQuestion, GoldenSet
from longprobe.core.reporting import report_to_dict


@pytest.fixture
def project(tmp_path):
    suite = GoldenSet("policies", "1", [GoldenQuestion("q", "Question", ["a", "b"])])
    goldens = tmp_path / "goldens.yaml"
    suite.to_yaml(str(goldens))
    destination = tmp_path / "restored.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"baseline": {"auto_compare": False, "db_path": str(destination)}})
    )
    config = ProbeConfig.from_yaml(str(config_path))
    report = evaluate(suite, lambda q, k: [{"id": "a"}, {"id": "b"}], config, compare=False)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    BaselineStore(str(artifact / "baselines.db")).save(report, "release")
    return suite, config_path, goldens, config, report, artifact, destination


def test_restores_explicit_label_without_rewriting_source(project):
    _, config, goldens, _, report, artifact, destination = project
    before = (artifact / "baselines.db").read_bytes()
    restore_baseline(artifact, destination, config, goldens, "release")
    assert BaselineStore(str(destination)).load("release") == report
    assert (artifact / "baselines.db").read_bytes() == before


@pytest.mark.parametrize(
    "change",
    ["missing-label", "mismatched-path", "incompatible", "multiple-files", "malformed", "legacy"],
)
def test_invalid_artifact_preserves_existing_destination(project, change):
    suite, config, goldens, _, report, artifact, destination = project
    destination.write_bytes(b"existing-reference-preserved")
    label = "release"
    if change == "missing-label":
        label = "missing"
    elif change == "mismatched-path":
        destination = destination.parent / "wrong.db"
        destination.write_bytes(b"existing-reference-preserved")
    elif change == "incompatible":
        suite.questions[0].required_chunks = ["new"]
        suite.to_yaml(str(goldens))
    elif change == "multiple-files":
        (artifact / "another.db").write_bytes(b"wrong")
    elif change == "malformed":
        (artifact / "baselines.db").write_bytes(b"not-sqlite")
    elif change == "legacy":
        report.schema_version = None
        report.results[0].contract = None
        BaselineStore(str(artifact / "baselines.db")).save(report, "release")
    with pytest.raises(ConfigurationError):
        restore_baseline(artifact, destination, config, goldens, label)
    assert destination.read_bytes() == b"existing-reference-preserved"


@pytest.mark.parametrize(
    "artifact,run_id", [("name", ""), ("", "123"), ("name", "-1"), ("name", "abc")]
)
def test_baseline_inputs_must_be_an_explicit_valid_pair(artifact, run_id):
    with pytest.raises(ConfigurationError):
        validate_baseline_inputs(artifact, run_id, "latest")


def test_missing_ci_configuration_is_a_setup_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "longprobe.ci",
            "restore",
            "--artifact-dir",
            str(tmp_path),
            "--destination",
            str(tmp_path / "b.db"),
            "--config",
            str(tmp_path / "absent.yaml"),
            "--goldens",
            str(tmp_path / "goldens.yaml"),
        ],
    )
    assert main() == 2


def test_relaxed_gate_does_not_hide_invalid_execution(project, tmp_path):
    suite, _, _, config, report, _, _ = project
    path = tmp_path / "result.json"
    path.write_text(json.dumps(report_to_dict(report)))
    assert determine_exit(path, 0, True) == 0
    assert determine_exit(path, 1, False) == 3  # Contradictory passing report.
    failed = evaluate(suite, lambda q, k: [], config, compare=False)
    path.write_text(json.dumps(report_to_dict(failed)))
    assert determine_exit(path, 0, True) == 1  # CLI may have nonblocking quality exits.
    assert determine_exit(path, 1, False) == 0
    assert determine_exit(path, 2, False) == 2
    assert determine_exit(path, 3, False) == 3
    path.write_text("not json")
    assert determine_exit(path, 1, False) == 3


def test_feedback_is_derived_from_one_evaluation(project, tmp_path, monkeypatch, capsys):
    _, config, goldens, _, report, _, _ = project
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write(json.dumps(report_to_dict(report)))
        kwargs["stderr"].write("Retained diagnostics")
        return CompletedProcess(command, 0)

    monkeypatch.setattr("longprobe.ci.subprocess.run", execute)
    outputs = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
    directory = tmp_path / "results"
    assert run_evaluation(config, goldens, label="release", directory=directory) == 0
    assert len(calls) == 1
    assert "check" in calls[0]
    assert "--baseline" in calls[0]
    assert "has-result=true" in outputs.read_text()
    assert "exit-code=0" in outputs.read_text()
    assert "Retained diagnostics" in (directory / "longprobe-diagnostics.txt").read_text()
    assert "PASS" in (directory / "longprobe-comment.md").read_text()
    assert "LongProbe Retrieval" in capsys.readouterr().out


def test_comment_escapes_document_text(project):
    *_, report, _artifact, _destination = project
    report.results[0].missing_chunks = ["<script>alert(1)</script>\n| fake |"]
    rendered = markdown_report(report)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "\\| fake \\|" in rendered


@pytest.mark.parametrize(
    "version,extras,expected",
    [
        ("0.2.0", "chroma,generate", "longprobe[chroma,generate]==0.2.0"),
        ("0.2.0rc1", "", "longprobe==0.2.0rc1"),
        ("", "", "longprobe"),
    ],
)
def test_workflow_install_step_honors_version_as_data(monkeypatch, version, extras, expected):
    path = Path(__file__).parents[2] / ".github/workflows/longprobe.yml"
    workflow = yaml.safe_load(path.read_text())
    install = next(
        step["run"]
        for step in workflow["jobs"]["longprobe-check"]["steps"]
        if step.get("name") == "Install requested LongProbe version"
    )
    code = "\n".join(install.splitlines()[1:-1])
    monkeypatch.setenv("LONGPROBE_VERSION", version)
    monkeypatch.setenv("LONGPROBE_EXTRAS", extras)
    install_call = Mock()
    monkeypatch.setattr("subprocess.run", install_call)
    monkeypatch.setattr("importlib.metadata.version", lambda name: version or "0.2.0")
    exec(compile(code, "workflow-install", "exec"), {})
    assert install_call.call_args.args[0][-1] == expected


@pytest.mark.parametrize(
    "version,extras",
    [("0.2.0; echo injected", ""), ("0.2.0", "http"), ("", "chroma];echo injected")],
)
def test_workflow_rejects_invalid_install_arguments(monkeypatch, version, extras):
    path = Path(__file__).parents[2] / ".github/workflows/longprobe.yml"
    steps = yaml.safe_load(path.read_text())["jobs"]["longprobe-check"]["steps"]
    install = next(
        step["run"] for step in steps if step.get("name") == "Install requested LongProbe version"
    )
    code = "\n".join(install.splitlines()[1:-1])
    monkeypatch.setenv("LONGPROBE_VERSION", version)
    monkeypatch.setenv("LONGPROBE_EXTRAS", extras)
    call = Mock()
    monkeypatch.setattr("subprocess.run", call)
    with pytest.raises(SystemExit):
        exec(compile(code, "workflow-install", "exec"), {})
    call.assert_not_called()
