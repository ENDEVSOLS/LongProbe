"""Unit tests for LP-02: HTTP adapter response validation, empty vs malformed handling, and error sanitization."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import yaml
from typer.testing import CliRunner

from longprobe.adapters.http import HttpAdapter, HttpAdapterError
from longprobe.cli.main import app
from longprobe.config import HttpResponseMapping, HttpRetrieverConfig, ProbeConfig
from longprobe.core.evaluation import evaluate
from longprobe.core.golden import GoldenQuestion, GoldenSet


@pytest.fixture
def http_config() -> HttpRetrieverConfig:
    return HttpRetrieverConfig(
        url="http://api.local/search",
        response_mapping=HttpResponseMapping(
            results_path="data.chunks",
            id_field="chunk_id",
            text_field="content",
            score_field="relevance",
        ),
    )


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    suite = GoldenSet("test", "1.0", [GoldenQuestion("q1", "What is LongProbe?", ["doc-1"])])
    suite.to_yaml("goldens.yaml")

    cfg = {
        "retriever": {
            "type": "http",
            "http": {
                "url": "http://api.local/search",
                "response_mapping": {
                    "results_path": "data.chunks",
                    "id_field": "chunk_id",
                    "text_field": "content",
                },
            },
        },
        "scoring": {
            "recall_threshold": 1.0,
        },
        "baseline": {
            "db_path": str(tmp_path / "baselines.db"),
        },
    }
    Path("longprobe.yaml").write_text(yaml.safe_dump(cfg))
    return CliRunner(), suite, tmp_path


class TestHttpResponseValidation:
    """Tests distinguishing valid empty retrieval results from malformed responses."""

    @patch("longprobe.adapters.http.requests.request")
    def test_valid_empty_list_returns_empty_and_evaluates(
        self, mock_request: MagicMock, http_config: HttpRetrieverConfig
    ) -> None:
        """A valid empty list payload completes evaluation as quality failure, not execution error."""
        adapter = HttpAdapter(http_config)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"chunks": []}}
        mock_request.return_value = mock_resp

        results = adapter.retrieve("test query", top_k=5)
        assert results == []

        # Run evaluate with this adapter; should produce recall = 0.0 without execution error
        suite = GoldenSet("s", "1", [GoldenQuestion("q1", "Q1", ["doc-1"])])
        config = ProbeConfig()
        report = evaluate(suite, adapter.retrieve, config, compare=False)
        assert len(report.results) == 1
        assert report.overall_recall == 0.0
        assert report.results[0].passed is False
        assert report.results[0].question_id == "q1"

    @patch("longprobe.adapters.http.requests.request")
    def test_missing_results_path_raises_invalid_response(
        self, mock_request: MagicMock, http_config: HttpRetrieverConfig
    ) -> None:
        adapter = HttpAdapter(http_config)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"wrong_key": []}
        mock_request.return_value = mock_resp

        with pytest.raises(HttpAdapterError) as exc_info:
            adapter.retrieve("test query")
        assert exc_info.value.category == "invalid_response"
        assert "not found in response" in str(exc_info.value)

    @patch("longprobe.adapters.http.requests.request")
    def test_non_list_results_raises_invalid_response(
        self, mock_request: MagicMock, http_config: HttpRetrieverConfig
    ) -> None:
        adapter = HttpAdapter(http_config)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"chunks": "not-a-list"}}
        mock_request.return_value = mock_resp

        with pytest.raises(HttpAdapterError) as exc_info:
            adapter.retrieve("test query")
        assert exc_info.value.category == "invalid_response"
        assert "did not resolve to a list" in str(exc_info.value)

    @patch("longprobe.adapters.http.requests.request")
    def test_non_dict_item_in_results_raises_invalid_response(
        self, mock_request: MagicMock, http_config: HttpRetrieverConfig
    ) -> None:
        adapter = HttpAdapter(http_config)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"chunks": ["not-a-dict"]}}
        mock_request.return_value = mock_resp

        with pytest.raises(HttpAdapterError) as exc_info:
            adapter.retrieve("test query")
        assert exc_info.value.category == "invalid_response"
        assert "is not an object" in str(exc_info.value)

    @patch("longprobe.adapters.http.requests.request")
    def test_missing_mapped_fields_raises_invalid_response(
        self, mock_request: MagicMock, http_config: HttpRetrieverConfig
    ) -> None:
        adapter = HttpAdapter(http_config)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"chunks": [{"other_field": 123}]}}
        mock_request.return_value = mock_resp

        with pytest.raises(HttpAdapterError) as exc_info:
            adapter.retrieve("test query")
        assert exc_info.value.category == "invalid_response"
        assert "missing required mapped fields" in str(exc_info.value)


class TestHttpErrorSanitization:
    """Tests ensuring error diagnostics strip raw response payloads."""

    @patch("longprobe.adapters.http.requests.request")
    def test_http_500_sanitizes_raw_server_body(
        self, mock_request: MagicMock, http_config: HttpRetrieverConfig
    ) -> None:
        adapter = HttpAdapter(http_config)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "SECRET_SQL_TRACEBACK_WITH_CREDENTIALS"
        mock_request.return_value = mock_resp

        with pytest.raises(HttpAdapterError) as exc_info:
            adapter.retrieve("test query")
        assert exc_info.value.category == "http_error"
        assert "SECRET_SQL_TRACEBACK_WITH_CREDENTIALS" not in str(exc_info.value)
        assert "500" in str(exc_info.value)

    @patch("longprobe.adapters.http.requests.request")
    def test_transport_error_sanitized(
        self, mock_request: MagicMock, http_config: HttpRetrieverConfig
    ) -> None:
        adapter = HttpAdapter(http_config)
        mock_request.side_effect = requests.ConnectionError(
            "Connection refused by secret internal IP"
        )

        with pytest.raises(HttpAdapterError) as exc_info:
            adapter.retrieve("test query")
        assert exc_info.value.category == "transport_error"
        assert "secret internal IP" not in str(exc_info.value)


class TestCliHttpExitCodes:
    """Tests exit codes in CLI commands and CI for empty vs invalid HTTP responses."""

    @patch("longprobe.adapters.http.requests.request")
    def test_cli_valid_empty_is_quality_failure(self, mock_request: MagicMock, cli_env) -> None:
        from longprobe.ci import determine_exit

        runner, _, tmp_path = cli_env
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"chunks": []}}
        mock_request.return_value = mock_resp

        # Blocking check -> exit code 1 (quality failure)
        res = runner.invoke(app, ["check", "--output", "json"])
        assert res.exit_code == 1
        data = json.loads(res.stdout)
        assert data["status"] == "failed"

        # Advisory / non-blocking check via config (fail_on_regression: False) -> exit code 0
        cfg_path = tmp_path / "longprobe.yaml"
        cfg_dict = yaml.safe_load(cfg_path.read_text())
        cfg_dict["scoring"]["fail_on_regression"] = False
        cfg_path.write_text(yaml.safe_dump(cfg_dict))

        res_advisory = runner.invoke(app, ["check", "--output", "json"])
        assert res_advisory.exit_code == 0
        data_advisory = json.loads(res_advisory.stdout)
        assert data_advisory["status"] == "failed"

        # CI determine_exit with quality failure and blocking=False -> exits 0
        report_file = tmp_path / "result.json"
        report_file.write_text(res.stdout)
        assert determine_exit(report_file, process_exit=1, blocking=False) == 0

    @patch("longprobe.adapters.http.requests.request")
    def test_cli_malformed_response_is_exit_3_even_in_nonblocking(
        self, mock_request: MagicMock, cli_env
    ) -> None:
        from longprobe.ci import determine_exit

        runner, _, tmp_path = cli_env
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"wrong_key": []}
        mock_request.return_value = mock_resp

        # Blocking check -> exit code 3 (execution error)
        res = runner.invoke(app, ["check", "--output", "json"])
        assert res.exit_code == 3
        data = json.loads(res.stdout)
        assert data["status"] == "error"
        assert data["error"]["code"] == "invalid_response"

        # Non-blocking config mode (fail_on_regression: False) MUST STILL BE EXIT CODE 3
        cfg_path = tmp_path / "longprobe.yaml"
        cfg_dict = yaml.safe_load(cfg_path.read_text())
        cfg_dict["scoring"]["fail_on_regression"] = False
        cfg_path.write_text(yaml.safe_dump(cfg_dict))

        res_nb = runner.invoke(app, ["check", "--output", "json"])
        assert res_nb.exit_code == 3
        data_nb = json.loads(res_nb.stdout)
        assert data_nb["status"] == "error"
        assert data_nb["error"]["code"] == "invalid_response"

        # CI determine_exit with execution error 3 and blocking=False MUST STILL BE 3
        report_file = tmp_path / "result.json"
        report_file.write_text(res.stdout)
        assert determine_exit(report_file, process_exit=3, blocking=False) == 3
