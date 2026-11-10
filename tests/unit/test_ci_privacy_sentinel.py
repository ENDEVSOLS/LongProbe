"""Synthetic sentinel tests for LP-03: CI privacy and redaction by default."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest
import yaml

from longprobe.adapters.http import HttpAdapter, HttpAdapterError
from longprobe.ci import run_evaluation
from longprobe.config import HttpResponseMapping, HttpRetrieverConfig, ProbeConfig
from longprobe.core.baseline import BaselineStore
from longprobe.core.evaluation import evaluate
from longprobe.core.golden import GoldenQuestion, GoldenSet
from longprobe.core.reporting import report_to_dict

SENTINEL_QUESTION = "SECRET_PROMPT_QUESTION_CANARY_98765"
SENTINEL_CHUNK = "SECRET_EVIDENCE_CHUNK_CANARY_43210"
SENTINEL_RESPONSE_SNIPPET = "SECRET_RAW_RETRIEVAL_PAYLOAD_CANARY_11223"
SENTINEL_HTTP_ERROR = "SECRET_DATABASE_PASSWORD_EXPOSED_CANARY_99887"


@pytest.fixture
def sentinel_eval_data(tmp_path: Path):
    suite = GoldenSet(
        "sentinel-suite",
        "1.0",
        [GoldenQuestion("q_sentinel", SENTINEL_QUESTION, [SENTINEL_CHUNK])],
    )
    db_path = tmp_path / "baselines.db"
    cfg = ProbeConfig()
    cfg.baseline.db_path = str(db_path)

    # 1. Baseline report where chunk was found
    base_report = evaluate(
        suite,
        lambda q, k: [{"id": SENTINEL_CHUNK, "text": SENTINEL_CHUNK}],
        cfg,
        compare=False,
    )
    BaselineStore(str(db_path)).save(base_report, "base")

    # 2. Current report where chunk was lost and snippet returned; diffed against baseline
    store = BaselineStore(str(db_path))
    curr_report = evaluate(
        suite,
        lambda q, k: [{"id": "other_id", "text": SENTINEL_RESPONSE_SNIPPET}],
        cfg,
        store=store,
        baseline_label="base",
        compare=True,
    )

    return suite, cfg, base_report, curr_report, tmp_path


def test_sentinel_privacy_on_all_shared_surfaces(sentinel_eval_data, tmp_path, monkeypatch, capsys):
    """Confirm sensitive question text, evidence, and response fragments are absent from all shared surfaces."""
    suite, cfg, _base_report, curr_report, _ = sentinel_eval_data
    results_dir = tmp_path / "ci_results"

    config_path = tmp_path / "longprobe.yaml"
    config_path.write_text(yaml.safe_dump({"baseline": {"db_path": cfg.baseline.db_path}}))
    goldens_path = tmp_path / "goldens.yaml"
    suite.to_yaml(str(goldens_path))

    # Mock subprocess.run in ci.py to return the evaluated curr_report
    def execute(command, **kwargs):
        kwargs["stdout"].write(json.dumps(report_to_dict(curr_report)))
        kwargs["stderr"].write("Diagnostics without secrets\n")
        return CompletedProcess(command, 1)  # Process exited with quality failure

    monkeypatch.setattr("longprobe.ci.subprocess.run", execute)

    exit_code = run_evaluation(
        config=config_path,
        goldens=goldens_path,
        label="base",
        blocking=False,
        directory=results_dir,
        redact=True,
    )

    # Completed quality failure relaxed in advisory mode
    assert exit_code == 0
    captured = capsys.readouterr()

    # Shared Surface 1: stdout annotations
    stdout_text = captured.out
    assert SENTINEL_QUESTION not in stdout_text
    assert SENTINEL_CHUNK not in stdout_text
    assert SENTINEL_RESPONSE_SNIPPET not in stdout_text
    assert "q_sentinel" in stdout_text
    assert "missing 1 of 1 chunk(s)" in stdout_text
    assert "Required evidence lost: 1 chunk(s)" in stdout_text

    # Shared Surface 2: PR markdown comment (longprobe-comment.md)
    comment_file = results_dir / "longprobe-comment.md"
    assert comment_file.exists()
    comment_text = comment_file.read_text(encoding="utf-8")
    assert SENTINEL_QUESTION not in comment_text
    assert SENTINEL_CHUNK not in comment_text
    assert SENTINEL_RESPONSE_SNIPPET not in comment_text
    assert "q\\_sentinel" in comment_text
    assert "missing: 1 of 1 chunk(s)" in comment_text
    assert "Regression q\\_sentinel: 1 evidence chunk(s) lost" in comment_text

    # Shared Surface 3: Default uploaded JSON artifact (longprobe-result.json)
    result_json_file = results_dir / "longprobe-result.json"
    assert result_json_file.exists()
    json_text = result_json_file.read_text(encoding="utf-8")
    assert SENTINEL_QUESTION not in json_text
    assert SENTINEL_CHUNK not in json_text
    assert SENTINEL_RESPONSE_SNIPPET not in json_text

    # Verify JSON retains verdicts, metrics, and safe diff counts intact
    payload = json.loads(json_text)
    assert payload["redacted"] is True
    assert payload["status"] == "failed"
    assert payload["overall_recall"] == 0.0
    assert payload["regression_detected"] is True
    assert payload["total_questions"] == 1
    assert payload["failed_questions"] == 1
    assert len(payload["results"]) == 1
    res = payload["results"][0]
    assert res["question_id"] == "q_sentinel"
    assert res["passed"] is False
    assert res["required_count"] == 1
    assert res["found_count"] == 0
    assert res["missing_count"] == 1
    assert "question_text" not in res
    assert "required_chunks" not in res

    regressions = payload["comparison"]["regressions"]
    assert len(regressions) == 1
    assert regressions[0]["question_id"] == "q_sentinel"
    assert regressions[0]["lost_count"] == 1
    assert "newly_lost_chunks" not in regressions[0]

    # Shared Surface 4: Diagnostics file (longprobe-diagnostics.txt)
    diag_file = results_dir / "longprobe-diagnostics.txt"
    assert diag_file.exists()
    diag_text = diag_file.read_text(encoding="utf-8")
    assert SENTINEL_QUESTION not in diag_text
    assert SENTINEL_CHUNK not in diag_text
    assert SENTINEL_RESPONSE_SNIPPET not in diag_text


def test_sentinel_privacy_on_http_error(sentinel_eval_data, tmp_path, monkeypatch, capsys):
    """Confirm server error snippets and credentials are absent from diagnostics and shared outputs on error."""
    suite, cfg, _, _, _ = sentinel_eval_data
    results_dir = tmp_path / "ci_results_err"

    config_path = tmp_path / "longprobe.yaml"
    config_path.write_text(yaml.safe_dump({"baseline": {"db_path": cfg.baseline.db_path}}))
    goldens_path = tmp_path / "goldens.yaml"
    suite.to_yaml(str(goldens_path))

    # Mock subprocess.run simulating CLI error exit with sanitized diagnostic
    def execute(command, **kwargs):
        kwargs["stdout"].write(json.dumps({
            "schema_version": 2,
            "status": "error",
            "error": {"code": "http_error", "message": "HTTP adapter received status 500."},
        }))
        kwargs["stderr"].write("HTTP adapter received status 500.\n")
        return CompletedProcess(command, 3)

    monkeypatch.setattr("longprobe.ci.subprocess.run", execute)

    exit_code = run_evaluation(
        config=config_path,
        goldens=goldens_path,
        label="base",
        blocking=False,
        directory=results_dir,
        redact=True,
    )

    # Execution failure: exit 3 even in advisory mode
    assert exit_code == 3
    captured = capsys.readouterr()

    # Check stdout, stderr diagnostics, and result file
    assert SENTINEL_HTTP_ERROR not in captured.out
    assert SENTINEL_HTTP_ERROR not in captured.err

    diag_file = results_dir / "longprobe-diagnostics.txt"
    assert diag_file.exists()
    diag_text = diag_file.read_text(encoding="utf-8")
    assert SENTINEL_HTTP_ERROR not in diag_text


def test_private_evidence_retained_in_sqlite_and_explicit_opt_in(sentinel_eval_data, tmp_path, monkeypatch):
    """Confirm SQLite baselines and explicit non-redacted runs retain full private evidence."""
    suite, cfg, _, curr_report, _ = sentinel_eval_data

    # 1. Verify SQLite baseline retains the unredacted evidence
    store = BaselineStore(cfg.baseline.db_path)
    loaded = store.load("base")
    assert loaded is not None
    assert loaded.results[0].question == SENTINEL_QUESTION
    assert loaded.results[0].required_chunks == [SENTINEL_CHUNK]

    # 2. Explicit non-redacted CI run with redact=False
    results_dir = tmp_path / "ci_results_private"
    config_path = tmp_path / "longprobe.yaml"
    config_path.write_text(yaml.safe_dump({"baseline": {"db_path": cfg.baseline.db_path}}))
    goldens_path = tmp_path / "goldens.yaml"
    suite.to_yaml(str(goldens_path))

    def execute(command, **kwargs):
        kwargs["stdout"].write(json.dumps(report_to_dict(curr_report)))
        kwargs["stderr"].write("Diagnostics\n")
        return CompletedProcess(command, 1)

    monkeypatch.setattr("longprobe.ci.subprocess.run", execute)

    run_evaluation(
        config=config_path,
        goldens=goldens_path,
        label="base",
        blocking=False,
        directory=results_dir,
        redact=False,
    )

    comment = (results_dir / "longprobe-comment.md").read_text(encoding="utf-8")
    assert SENTINEL_CHUNK.replace("_", r"\_") in comment

    result_json = (results_dir / "longprobe-result.json").read_text(encoding="utf-8")
    assert SENTINEL_CHUNK in result_json


def test_adapter_level_sentinel_sanitization():
    """Verify HTTP adapter does not leak error response body or transport secrets into exception message."""
    cfg = HttpRetrieverConfig(
        url="http://internal.api/search",
        response_mapping=HttpResponseMapping(
            results_path="data.chunks",
            id_field="id",
            text_field="text",
        ),
    )
    adapter = HttpAdapter(cfg)

    # 500 error containing sensitive database details
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = f"Traceback: Database connection string postgres://admin:{SENTINEL_HTTP_ERROR}@db"

    with patch("longprobe.adapters.http.requests.request", return_value=mock_resp):
        with pytest.raises(HttpAdapterError) as exc_info:
            adapter.retrieve("test question")
        assert exc_info.value.category == "http_error"
        assert SENTINEL_HTTP_ERROR not in str(exc_info.value)
        assert "500" in str(exc_info.value)
