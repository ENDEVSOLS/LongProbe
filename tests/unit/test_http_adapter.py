"""Tests for ``longprobe.adapters.http`` — HttpAdapter.

Covers body template substitution, dot-notation path resolution, response
field mapping, health check, and all error handling paths (non-200, timeout,
invalid JSON, missing path, missing fields).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from longprobe.adapters.http import HttpAdapter
from longprobe.config import (
    HttpRetrieverConfig,
    HttpResponseMapping,
    ProbeConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_http_config() -> HttpRetrieverConfig:
    """Return a basic HTTP config pointing at a fake URL."""
    return HttpRetrieverConfig(
        url="http://localhost:8000/api/retrieve",
        method="POST",
        body_template='{"query": "{question}", "top_k": {top_k}}',
        headers={"Authorization": "Bearer test-key"},
        response_mapping=HttpResponseMapping(
            results_path="results",
            id_field="id",
            text_field="text",
            score_field="score",
        ),
        timeout=30,
    )


@pytest.fixture
def adapter(basic_http_config: HttpRetrieverConfig) -> HttpAdapter:
    """Return an HttpAdapter with basic config."""
    return HttpAdapter(config=basic_http_config)


@pytest.fixture
def nested_config() -> HttpRetrieverConfig:
    """Return an HTTP config with nested response path."""
    return HttpRetrieverConfig(
        url="http://localhost:8000/api/search",
        method="POST",
        body_template='{"q": "{question}", "count": {top_k}}',
        headers={},
        response_mapping=HttpResponseMapping(
            results_path="data.chunks",
            id_field="chunk_id",
            text_field="content",
            score_field="similarity",
        ),
        timeout=10,
    )


@pytest.fixture
def nested_adapter(nested_config: HttpRetrieverConfig) -> HttpAdapter:
    """Return an HttpAdapter with nested path config."""
    return HttpAdapter(config=nested_config)


# ---------------------------------------------------------------------------
# _build_body tests
# ---------------------------------------------------------------------------

class TestBuildBody:
    """Tests for HttpAdapter._build_body."""

    def test_basic_substitution(self) -> None:
        """Standard {question} and {top_k} replacement."""
        result = HttpAdapter._build_body(
            '{"query": "{question}", "top_k": {top_k}}',
            "What is Python?",
            5,
        )
        assert result == {"query": "What is Python?", "top_k": 5}

    def test_question_with_quotes(self) -> None:
        """Quotes in the question are JSON-escaped."""
        result = HttpAdapter._build_body(
            '{"query": "{question}", "top_k": {top_k}}',
            'He said "hello"',
            3,
        )
        assert result == {"query": 'He said "hello"', "top_k": 3}

    def test_question_with_newlines(self) -> None:
        """Newlines in the question are JSON-escaped."""
        result = HttpAdapter._build_body(
            '{"query": "{question}", "top_k": {top_k}}',
            "line1\nline2",
            5,
        )
        assert result == {"query": "line1\nline2", "top_k": 5}

    def test_custom_field_names(self) -> None:
        """Template with custom field names."""
        result = HttpAdapter._build_body(
            '{"q": "{question}", "count": {top_k}}',
            "test query",
            10,
        )
        assert result == {"q": "test query", "count": 10}

    def test_invalid_json_raises_value_error(self) -> None:
        """Invalid JSON after substitution raises ValueError."""
        with pytest.raises(ValueError, match="invalid JSON"):
            HttpAdapter._build_body(
                '{"broken": {question}}',  # missing quotes around {question}
                "test",
                5,
            )

    def test_template_without_placeholders(self) -> None:
        """Template with no placeholders passes through as-is."""
        result = HttpAdapter._build_body(
            '{"static": "value"}',
            "ignored",
            5,
        )
        assert result == {"static": "value"}


# ---------------------------------------------------------------------------
# _resolve_path tests
# ---------------------------------------------------------------------------

class TestResolvePath:
    """Tests for HttpAdapter._resolve_path."""

    def test_simple_path(self) -> None:
        """Single-level path resolves correctly."""
        data = {"results": [1, 2, 3]}
        assert HttpAdapter._resolve_path(data, "results") == [1, 2, 3]

    def test_nested_path(self) -> None:
        """Multi-level dot-notation path resolves correctly."""
        data = {"data": {"chunks": [{"id": "a"}]}}
        assert HttpAdapter._resolve_path(data, "data.chunks") == [{"id": "a"}]

    def test_deeply_nested_path(self) -> None:
        """Three-level path resolves correctly."""
        data = {"response": {"data": {"items": ["x"]}}}
        assert HttpAdapter._resolve_path(data, "response.data.items") == ["x"]

    def test_missing_key_returns_empty_list(self) -> None:
        """Missing key in path returns empty list."""
        data = {"other": []}
        assert HttpAdapter._resolve_path(data, "results") == []

    def test_partial_path_missing_returns_empty_list(self) -> None:
        """Missing intermediate key returns empty list."""
        data = {"data": {}}
        assert HttpAdapter._resolve_path(data, "data.chunks") == []

    def test_non_dict_intermediate_returns_empty_list(self) -> None:
        """Non-dict value in path returns empty list."""
        data = {"data": "not a dict"}
        assert HttpAdapter._resolve_path(data, "data.chunks") == []

    def test_empty_path(self) -> None:
        """Empty path returns the root data."""
        data = {"key": "value"}
        assert HttpAdapter._resolve_path(data, "") == data


# ---------------------------------------------------------------------------
# retrieve() tests
# ---------------------------------------------------------------------------

class TestRetrieve:
    """Tests for HttpAdapter.retrieve."""

    @patch("longprobe.adapters.http.requests.request")
    def test_basic_retrieval(self, mock_request: MagicMock, adapter: HttpAdapter) -> None:
        """Standard retrieval returns normalised results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"id": "chunk-1", "text": "Hello world", "score": 0.95},
                {"id": "chunk-2", "text": "Goodbye world", "score": 0.80},
            ],
        }
        mock_request.return_value = mock_response

        results = adapter.retrieve("What is X?", top_k=5)

        assert len(results) == 2
        assert results[0]["id"] == "chunk-1"
        assert results[0]["text"] == "Hello world"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"] == {}
        assert results[1]["id"] == "chunk-2"

        # Verify the request was made correctly.
        mock_request.assert_called_once_with(
            method="POST",
            url="http://localhost:8000/api/retrieve",
            json={"query": "What is X?", "top_k": 5},
            headers={"Authorization": "Bearer test-key"},
            timeout=30,
        )

    @patch("longprobe.adapters.http.requests.request")
    def test_nested_path_retrieval(
        self, mock_request: MagicMock, nested_adapter: HttpAdapter
    ) -> None:
        """Nested response path resolves correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "chunks": [
                    {"chunk_id": "c1", "content": "Text A", "similarity": 0.9},
                ],
            },
        }
        mock_request.return_value = mock_response

        results = nested_adapter.retrieve("test query", top_k=3)

        assert len(results) == 1
        assert results[0]["id"] == "c1"
        assert results[0]["text"] == "Text A"
        assert results[0]["score"] == 0.9

        # Verify custom body template fields.
        call_kwargs = mock_request.call_args
        assert call_kwargs.kwargs["json"] == {"q": "test query", "count": 3}

    @patch("longprobe.adapters.http.requests.request")
    def test_missing_fields_use_defaults(
        self, mock_request: MagicMock, adapter: HttpAdapter
    ) -> None:
        """Missing fields in result items use safe defaults."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"id": "chunk-1"},  # missing text and score
            ],
        }
        mock_request.return_value = mock_response

        results = adapter.retrieve("test", top_k=5)

        assert len(results) == 1
        assert results[0]["id"] == "chunk-1"
        assert results[0]["text"] == ""
        assert results[0]["score"] == 0.0

    @patch("longprobe.adapters.http.requests.request")
    def test_extra_fields_go_to_metadata(
        self, mock_request: MagicMock, adapter: HttpAdapter
    ) -> None:
        """Fields not mapped to id/text/score go to metadata."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "c1",
                    "text": "content",
                    "score": 0.5,
                    "source": "doc.pdf",
                    "page": 3,
                },
            ],
        }
        mock_request.return_value = mock_response

        results = adapter.retrieve("test", top_k=5)

        assert results[0]["metadata"] == {"source": "doc.pdf", "page": 3}

    @patch("longprobe.adapters.http.requests.request")
    def test_missing_path_returns_empty_list(
        self, mock_request: MagicMock, adapter: HttpAdapter
    ) -> None:
        """Missing results_path in response returns empty list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"other_key": []}
        mock_request.return_value = mock_response

        results = adapter.retrieve("test", top_k=5)

        assert results == []

    @patch("longprobe.adapters.http.requests.request")
    def test_non_list_results_returns_empty_list(
        self, mock_request: MagicMock, adapter: HttpAdapter
    ) -> None:
        """When resolved path is not a list, return empty."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": "not a list"}
        mock_request.return_value = mock_response

        results = adapter.retrieve("test", top_k=5)

        assert results == []

    @patch("longprobe.adapters.http.requests.request")
    def test_non_dict_items_skipped(
        self, mock_request: MagicMock, adapter: HttpAdapter
    ) -> None:
        """Non-dict items in the results array are skipped."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"id": "c1", "text": "valid", "score": 0.5},
                "invalid string item",
                42,
            ],
        }
        mock_request.return_value = mock_response

        results = adapter.retrieve("test", top_k=5)

        assert len(results) == 1
        assert results[0]["id"] == "c1"

    @patch("longprobe.adapters.http.requests.request")
    def test_non_200_raises_runtime_error(
        self, mock_request: MagicMock, adapter: HttpAdapter
    ) -> None:
        """Non-2xx response raises RuntimeError with status and body."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_request.return_value = mock_response

        with pytest.raises(RuntimeError, match="status 500"):
            adapter.retrieve("test", top_k=5)

    @patch("longprobe.adapters.http.requests.request")
    def test_404_raises_runtime_error(
        self, mock_request: MagicMock, adapter: HttpAdapter
    ) -> None:
        """404 response raises RuntimeError."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_request.return_value = mock_response

        with pytest.raises(RuntimeError, match="status 404"):
            adapter.retrieve("test", top_k=5)

    @patch("longprobe.adapters.http.requests.request")
    def test_timeout_raises_requests_timeout(
        self, mock_request: MagicMock, adapter: HttpAdapter
    ) -> None:
        """Request timeout raises requests.Timeout with clear message."""
        mock_request.side_effect = requests.Timeout("timed out")

        with pytest.raises(requests.Timeout, match="timed out after 30s"):
            adapter.retrieve("test", top_k=5)

    @patch("longprobe.adapters.http.requests.request")
    def test_non_json_response_raises_runtime_error(
        self, mock_request: MagicMock, adapter: HttpAdapter
    ) -> None:
        """Non-JSON response body raises RuntimeError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("err", "doc", 0)
        mock_response.text = "<html>Error</html>"
        mock_request.return_value = mock_response

        with pytest.raises(RuntimeError, match="non-JSON response"):
            adapter.retrieve("test", top_k=5)

    @patch("longprobe.adapters.http.requests.request")
    def test_empty_results_array(
        self, mock_request: MagicMock, adapter: HttpAdapter
    ) -> None:
        """Empty results array returns empty list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_request.return_value = mock_response

        results = adapter.retrieve("test", top_k=5)

        assert results == []


# ---------------------------------------------------------------------------
# health_check() tests
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Tests for HttpAdapter.health_check."""

    @patch("longprobe.adapters.http.requests.head")
    def test_healthy_returns_true(self, mock_head: MagicMock, adapter: HttpAdapter) -> None:
        """2xx response from HEAD returns True."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        assert adapter.health_check() is True

    @patch("longprobe.adapters.http.requests.get")
    @patch("longprobe.adapters.http.requests.head")
    def test_head_not_allowed_falls_back_to_get(
        self, mock_head: MagicMock, mock_get: MagicMock, adapter: HttpAdapter
    ) -> None:
        """HEAD returning 405 falls back to GET."""
        mock_head_response = MagicMock()
        mock_head_response.status_code = 405
        mock_head.return_value = mock_head_response

        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get.return_value = mock_get_response

        assert adapter.health_check() is True
        mock_get.assert_called_once()

    @patch("longprobe.adapters.http.requests.head")
    def test_500_returns_false(
        self, mock_head: MagicMock, adapter: HttpAdapter
    ) -> None:
        """5xx response returns False."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_head.return_value = mock_response

        assert adapter.health_check() is False

    @patch("longprobe.adapters.http.requests.head")
    def test_connection_error_returns_false(
        self, mock_head: MagicMock, adapter: HttpAdapter
    ) -> None:
        """Connection error returns False."""
        mock_head.side_effect = requests.ConnectionError("refused")

        assert adapter.health_check() is False

    @patch("longprobe.adapters.http.requests.head")
    def test_timeout_returns_false(
        self, mock_head: MagicMock, adapter: HttpAdapter
    ) -> None:
        """Timeout during health check returns False."""
        mock_head.side_effect = requests.Timeout("timeout")

        assert adapter.health_check() is False


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Tests for loading HTTP config through ProbeConfig."""

    def test_load_http_config_from_dict(self) -> None:
        """HTTP config loads correctly from a YAML-like dict."""
        config = ProbeConfig.from_dict({
            "retriever": {
                "type": "http",
                "http": {
                    "url": "http://example.com/api/search",
                    "method": "POST",
                    "body_template": '{"q": "{question}", "n": {top_k}}',
                    "headers": {
                        "Authorization": "Bearer ${MY_API_KEY}",
                    },
                    "response_mapping": {
                        "results_path": "data.items",
                        "id_field": "doc_id",
                        "text_field": "body",
                        "score_field": "relevance",
                    },
                    "timeout": 60,
                },
            },
        })

        assert config.retriever.type == "http"
        assert config.retriever.http.url == "http://example.com/api/search"
        assert config.retriever.http.method == "POST"
        assert config.retriever.http.timeout == 60
        assert config.retriever.http.response_mapping.results_path == "data.items"
        assert config.retriever.http.response_mapping.id_field == "doc_id"
        assert config.retriever.http.response_mapping.text_field == "body"
        assert config.retriever.http.response_mapping.score_field == "relevance"

    def test_env_var_expansion_in_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """${ENV_VAR} in header values are expanded."""
        monkeypatch.setenv("TEST_API_KEY", "secret-123")

        config = ProbeConfig.from_dict({
            "retriever": {
                "type": "http",
                "http": {
                    "url": "http://example.com/api",
                    "headers": {
                        "Authorization": "Bearer ${TEST_API_KEY}",
                    },
                },
            },
        })

        assert config.retriever.http.headers["Authorization"] == "Bearer secret-123"

    def test_env_var_expansion_in_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """${ENV_VAR} in URL is expanded."""
        monkeypatch.setenv("RAG_HOST", "my-rag-server.internal")

        config = ProbeConfig.from_dict({
            "retriever": {
                "type": "http",
                "http": {
                    "url": "http://${RAG_HOST}:8000/api/retrieve",
                },
            },
        })

        assert config.retriever.http.url == "http://my-rag-server.internal:8000/api/retrieve"

    def test_default_http_config(self) -> None:
        """Default ProbeConfig has sensible HTTP defaults."""
        config = ProbeConfig.defaults()
        assert config.retriever.http.url == ""
        assert config.retriever.http.method == "POST"
        assert config.retriever.http.timeout == 30
        assert config.retriever.http.response_mapping.results_path == "results"

    def test_http_config_without_http_section(self) -> None:
        """Config with type: http but no http section uses defaults."""
        config = ProbeConfig.from_dict({"retriever": {"type": "http"}})
        assert config.retriever.http.url == ""
        assert config.retriever.http.method == "POST"


# ---------------------------------------------------------------------------
# Factory integration tests
# ---------------------------------------------------------------------------

class TestFactoryIntegration:
    """Tests for create_adapter with type 'http'."""

    def test_create_http_adapter(self) -> None:
        """Factory creates HttpAdapter with correct type."""
        from longprobe.adapters import create_adapter

        cfg = HttpRetrieverConfig(url="http://localhost:8000/api")
        adapter = create_adapter("http", config=cfg)
        assert isinstance(adapter, HttpAdapter)

    def test_create_http_adapter_with_response_mapping(self) -> None:
        """Factory creates adapter with custom response mapping."""
        from longprobe.adapters import create_adapter

        cfg = HttpRetrieverConfig(
            url="http://localhost:8000/api",
            response_mapping=HttpResponseMapping(
                results_path="data.chunks",
                id_field="chunk_id",
                text_field="content",
                score_field="similarity",
            ),
        )
        adapter = create_adapter("http", config=cfg)
        assert isinstance(adapter, HttpAdapter)

    def test_unknown_adapter_type_still_raises(self) -> None:
        """Factory still raises ValueError for unknown types."""
        from longprobe.adapters import create_adapter

        with pytest.raises(ValueError, match="Unknown adapter type"):
            create_adapter("nonexistent")
