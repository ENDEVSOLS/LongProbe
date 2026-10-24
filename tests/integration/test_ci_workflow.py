"""Real CLI subprocess against a controlled server and restored SQLite artifact."""

import json
import shutil

import pytest
import yaml
from typer.testing import CliRunner

from longprobe.ci import restore_baseline, run_evaluation
from longprobe.cli.main import app
from longprobe.core.golden import GoldenQuestion, GoldenSet

from .test_reliability_workflow import retrieval_service as retrieval_service

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("blocking", [True, False])
def test_restored_baseline_and_single_run_feedback(
    tmp_path, monkeypatch, retrieval_service, blocking
):
    url, state = retrieval_service
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    suite = GoldenSet("fixture", "1", [GoldenQuestion("refund", "Refund?", ["policy-1"])])
    suite.to_yaml("goldens.yaml")
    config = {
        "retriever": {
            "type": "http",
            "http": {
                "url": url,
                "response_mapping": {
                    "results_path": "data.chunks",
                    "id_field": "chunk_id",
                    "text_field": "content",
                    "score_field": "similarity",
                },
            },
        },
        "baseline": {"db_path": "baseline.db"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    runner = CliRunner()
    initial = runner.invoke(
        app, ["baseline", "save", "--config", str(config_path), "--label", "release"]
    )
    assert initial.exit_code == 0, initial.output
    artifact = tmp_path / "downloaded"
    artifact.mkdir()
    shutil.move("baseline.db", artifact / "baselines.db")
    restore_baseline(
        artifact, tmp_path / "baseline.db", config_path, tmp_path / "goldens.yaml", "release"
    )

    state["calls"] = 0
    results = tmp_path / "results"
    assert (
        run_evaluation(config_path, tmp_path / "goldens.yaml", label="release", directory=results)
        == 0
    )
    assert state["calls"] == 1
    report = json.loads((results / "longprobe-result.json").read_text())
    assert report["baseline_identity"]["resolved_label"] == "release"
    assert report["comparison"]["compatible"]
    assert "PASS" in (results / "longprobe-comment.md").read_text()

    state["calls"] = 0
    state["results"] = []
    code = run_evaluation(
        config_path,
        tmp_path / "goldens.yaml",
        label="release",
        blocking=blocking,
        directory=results,
    )
    assert code == (1 if blocking else 0)
    assert state["calls"] == 1
    report = json.loads((results / "longprobe-result.json").read_text())
    assert report["status"] == "failed"
    assert report["comparison"]["regressions"][0]["newly_lost_chunks"] == ["policy-1"]
    assert "FAIL" in (results / "longprobe-comment.md").read_text()
