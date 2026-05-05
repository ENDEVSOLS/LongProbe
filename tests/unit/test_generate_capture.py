"""Tests for the `generate --auto-capture` flow.

Tests the auto-capture integration by mocking the LLM generation step
and the retriever adapter, verifying that questions are correctly
captured into a golden set file.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from longprobe.adapters.base import AbstractRetrieverAdapter
from longprobe.config import (
    GeneratorConfig,
    HttpRetrieverConfig,
    ProbeConfig,
    RetrieverConfig,
)
from longprobe.core.golden import GoldenSet


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------

class MockAdapter(AbstractRetrieverAdapter):
    """A mock adapter that returns predictable results."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self._results = results or [
            {"id": "chunk-1", "text": "Some text about refunds", "score": 0.9},
            {"id": "chunk-2", "text": "Shipping takes 3-5 days", "score": 0.8},
        ]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return self._results[:top_k]


class EmptyAdapter(AbstractRetrieverAdapter):
    """An adapter that always returns empty results."""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return []


class FailingAdapter(AbstractRetrieverAdapter):
    """An adapter that raises on retrieve."""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        raise RuntimeError("Connection refused")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config() -> ProbeConfig:
    """Return a config with HTTP retriever and generator settings."""
    return ProbeConfig(
        retriever=RetrieverConfig(
            type="http",
            http=HttpRetrieverConfig(url="http://localhost:8000/api/retrieve"),
        ),
        generator=GeneratorConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key="test-key",
            num_questions=3,
        ),
    )


@pytest.fixture
def tmp_goldens_dir() -> Path:
    """Create a temp directory for golden set files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


# ---------------------------------------------------------------------------
# Auto-capture logic tests
# ---------------------------------------------------------------------------

class TestAutoCapture:
    """Tests for the auto-capture flow within the generate command."""

    @patch("longprobe.cli.main._create_adapter_from_config")
    @patch("longprobe.cli.main._load_config")
    @patch("longprobe.cli.main.QuestionGenerator")
    @patch("longprobe.cli.main.DocumentParser")
    def test_auto_capture_writes_golden_set(
        self,
        mock_parser_cls: MagicMock,
        mock_gen_cls: MagicMock,
        mock_load_config: MagicMock,
        mock_create_adapter: MagicMock,
        tmp_goldens_dir: Path,
        mock_config: ProbeConfig,
    ) -> None:
        """Auto-capture generates questions, queries retriever, writes goldens.yaml."""
        goldens_path = tmp_goldens_dir / "goldens.yaml"
        doc_path = tmp_goldens_dir / "doc.txt"
        doc_path.write_text("Test content", encoding="utf-8")

        # Setup mocks.
        mock_load_config.return_value = mock_config

        mock_parser = MagicMock()
        mock_parser.parse_path.return_value = [(str(doc_path), "Test content")]
        mock_parser_cls.return_value = mock_parser

        mock_generator = MagicMock()
        mock_generator.generate.return_value = [
            "What is the refund policy?",
            "How long does shipping take?",
        ]
        mock_gen_cls.return_value = mock_generator

        mock_create_adapter.return_value = MockAdapter()

        # Run via CLI invoke.
        from typer.testing import CliRunner
        from longprobe.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate", str(doc_path),
                "--auto-capture",
                "--goldens", str(goldens_path),
                "--num-questions", "2",
                "--match-mode", "text",
            ],
            catch_exceptions=False,
        )

        # Check the golden set was written.
        assert goldens_path.exists(), f"Golden set not written. Output: {result.output}"

        with open(goldens_path) as f:
            data = yaml.safe_load(f)

        assert data["name"] == "auto-generated"
        assert len(data["questions"]) == 2
        assert any("refund" in q["question"].lower() for q in data["questions"])
        # Each question should have required_chunks from the mock adapter.
        for q in data["questions"]:
            assert len(q["required_chunks"]) > 0

    @patch("longprobe.cli.main._create_adapter_from_config")
    @patch("longprobe.cli.main._load_config")
    @patch("longprobe.cli.main.QuestionGenerator")
    @patch("longprobe.cli.main.DocumentParser")
    def test_auto_capture_merges_existing_goldens(
        self,
        mock_parser_cls: MagicMock,
        mock_gen_cls: MagicMock,
        mock_load_config: MagicMock,
        mock_create_adapter: MagicMock,
        tmp_goldens_dir: Path,
        mock_config: ProbeConfig,
    ) -> None:
        """Auto-capture merges into an existing goldens.yaml without duplicates."""
        goldens_path = tmp_goldens_dir / "goldens.yaml"
        doc_path = tmp_goldens_dir / "doc.txt"
        doc_path.write_text("Test content", encoding="utf-8")

        # Write an existing golden set.
        existing = GoldenSet(name="my-set", version="1.0", questions=[])
        existing.to_yaml(str(goldens_path))

        mock_load_config.return_value = mock_config
        mock_parser = MagicMock()
        mock_parser.parse_path.return_value = [(str(doc_path), "Test content")]
        mock_parser_cls.return_value = mock_parser

        mock_generator = MagicMock()
        mock_generator.generate.return_value = ["What is Python?"]
        mock_gen_cls.return_value = mock_generator

        mock_create_adapter.return_value = MockAdapter()

        from typer.testing import CliRunner
        from longprobe.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate", str(doc_path),
                "--auto-capture",
                "--goldens", str(goldens_path),
                "--num-questions", "1",
            ],
            catch_exceptions=False,
        )

        with open(goldens_path) as f:
            data = yaml.safe_load(f)

        # Name should be preserved from existing set.
        assert data["name"] == "my-set"
        assert len(data["questions"]) == 1

    @patch("longprobe.cli.main._create_adapter_from_config")
    @patch("longprobe.cli.main._load_config")
    @patch("longprobe.cli.main.QuestionGenerator")
    @patch("longprobe.cli.main.DocumentParser")
    def test_auto_capture_with_id_match_mode(
        self,
        mock_parser_cls: MagicMock,
        mock_gen_cls: MagicMock,
        mock_load_config: MagicMock,
        mock_create_adapter: MagicMock,
        tmp_goldens_dir: Path,
        mock_config: ProbeConfig,
    ) -> None:
        """Auto-capture with --match-mode id saves chunk IDs."""
        goldens_path = tmp_goldens_dir / "goldens.yaml"
        doc_path = tmp_goldens_dir / "doc.txt"
        doc_path.write_text("Test content", encoding="utf-8")

        mock_load_config.return_value = mock_config
        mock_parser = MagicMock()
        mock_parser.parse_path.return_value = [(str(doc_path), "Test content")]
        mock_parser_cls.return_value = mock_parser

        mock_generator = MagicMock()
        mock_generator.generate.return_value = ["What is X?"]
        mock_gen_cls.return_value = mock_generator

        mock_create_adapter.return_value = MockAdapter()

        from typer.testing import CliRunner
        from longprobe.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate", str(doc_path),
                "--auto-capture",
                "--goldens", str(goldens_path),
                "--match-mode", "id",
                "--num-questions", "1",
            ],
            catch_exceptions=False,
        )

        with open(goldens_path) as f:
            data = yaml.safe_load(f)

        assert data["questions"][0]["match_mode"] == "id"
        assert "chunk-1" in data["questions"][0]["required_chunks"]

    @patch("longprobe.cli.main._create_adapter_from_config")
    @patch("longprobe.cli.main._load_config")
    @patch("longprobe.cli.main.QuestionGenerator")
    @patch("longprobe.cli.main.DocumentParser")
    def test_auto_capture_skips_empty_results(
        self,
        mock_parser_cls: MagicMock,
        mock_gen_cls: MagicMock,
        mock_load_config: MagicMock,
        mock_create_adapter: MagicMock,
        tmp_goldens_dir: Path,
        mock_config: ProbeConfig,
    ) -> None:
        """Questions with no retriever results are skipped with warning."""
        goldens_path = tmp_goldens_dir / "goldens.yaml"
        doc_path = tmp_goldens_dir / "doc.txt"
        doc_path.write_text("Test content", encoding="utf-8")

        mock_load_config.return_value = mock_config
        mock_parser = MagicMock()
        mock_parser.parse_path.return_value = [(str(doc_path), "Test content")]
        mock_parser_cls.return_value = mock_parser

        mock_generator = MagicMock()
        mock_generator.generate.return_value = ["Q1?", "Q2?"]
        mock_gen_cls.return_value = mock_generator

        mock_create_adapter.return_value = EmptyAdapter()

        from typer.testing import CliRunner
        from longprobe.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate", str(doc_path),
                "--auto-capture",
                "--goldens", str(goldens_path),
                "--num-questions", "2",
            ],
            catch_exceptions=False,
        )

        # Golden set should NOT be created (all questions skipped).
        assert not goldens_path.exists()
        assert "Skipped" in result.output

    @patch("longprobe.cli.main._create_adapter_from_config")
    @patch("longprobe.cli.main._load_config")
    @patch("longprobe.cli.main.QuestionGenerator")
    @patch("longprobe.cli.main.DocumentParser")
    def test_auto_capture_handles_partial_failures(
        self,
        mock_parser_cls: MagicMock,
        mock_gen_cls: MagicMock,
        mock_load_config: MagicMock,
        mock_create_adapter: MagicMock,
        tmp_goldens_dir: Path,
        mock_config: ProbeConfig,
    ) -> None:
        """Some questions failing still saves the successful ones."""
        goldens_path = tmp_goldens_dir / "goldens.yaml"
        doc_path = tmp_goldens_dir / "doc.txt"
        doc_path.write_text("Test content", encoding="utf-8")

        mock_load_config.return_value = mock_config
        mock_parser = MagicMock()
        mock_parser.parse_path.return_value = [(str(doc_path), "Test content")]
        mock_parser_cls.return_value = mock_parser

        mock_generator = MagicMock()
        mock_generator.generate.return_value = ["Good question?", "Bad question?"]
        mock_gen_cls.return_value = mock_generator

        # Adapter fails on first call, succeeds on second.
        call_count = 0

        def mock_retrieve(query: str, top_k: int = 5) -> list[dict]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Connection refused")
            return [{"id": "chunk-1", "text": "Result", "score": 0.9}]

        adapter = MagicMock()
        adapter.retrieve = mock_retrieve
        mock_create_adapter.return_value = adapter

        from typer.testing import CliRunner
        from longprobe.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate", str(doc_path),
                "--auto-capture",
                "--goldens", str(goldens_path),
                "--num-questions", "2",
            ],
            catch_exceptions=False,
        )

        # One question should have been captured.
        assert goldens_path.exists()
        with open(goldens_path) as f:
            data = yaml.safe_load(f)
        assert len(data["questions"]) == 1

    @patch("longprobe.cli.main._load_config")
    @patch("longprobe.cli.main.QuestionGenerator")
    @patch("longprobe.cli.main.DocumentParser")
    def test_auto_capture_requires_retriever_config(
        self,
        mock_parser_cls: MagicMock,
        mock_gen_cls: MagicMock,
        mock_load_config: MagicMock,
        tmp_goldens_dir: Path,
    ) -> None:
        """Auto-capture with no retriever configured prints error and exits."""
        doc_path = tmp_goldens_dir / "doc.txt"
        doc_path.write_text("Test content", encoding="utf-8")

        # Config with no retriever type set.
        empty_config = ProbeConfig(
            generator=GeneratorConfig(
                provider="openai",
                model="gpt-4o-mini",
                api_key="test-key",
            ),
        )
        mock_load_config.return_value = empty_config

        mock_parser = MagicMock()
        mock_parser.parse_path.return_value = [(str(doc_path), "Test content")]
        mock_parser_cls.return_value = mock_parser

        mock_generator = MagicMock()
        mock_generator.generate.return_value = ["What is X?"]
        mock_gen_cls.return_value = mock_generator

        from typer.testing import CliRunner
        from longprobe.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate", str(doc_path),
                "--auto-capture",
                "--goldens", str(tmp_goldens_dir / "goldens.yaml"),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 1
        assert "retriever" in result.output.lower() or "Error" in result.output

    @patch("longprobe.cli.main._create_adapter_from_config")
    @patch("longprobe.cli.main._load_config")
    @patch("longprobe.cli.main.QuestionGenerator")
    @patch("longprobe.cli.main.DocumentParser")
    def test_auto_capture_with_tags(
        self,
        mock_parser_cls: MagicMock,
        mock_gen_cls: MagicMock,
        mock_load_config: MagicMock,
        mock_create_adapter: MagicMock,
        tmp_goldens_dir: Path,
        mock_config: ProbeConfig,
    ) -> None:
        """Auto-capture with --tag applies tags to captured questions."""
        goldens_path = tmp_goldens_dir / "goldens.yaml"
        doc_path = tmp_goldens_dir / "doc.txt"
        doc_path.write_text("Test content", encoding="utf-8")

        mock_load_config.return_value = mock_config
        mock_parser = MagicMock()
        mock_parser.parse_path.return_value = [(str(doc_path), "Test content")]
        mock_parser_cls.return_value = mock_parser

        mock_generator = MagicMock()
        mock_generator.generate.return_value = ["What is X?"]
        mock_gen_cls.return_value = mock_generator

        mock_create_adapter.return_value = MockAdapter()

        from typer.testing import CliRunner
        from longprobe.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate", str(doc_path),
                "--auto-capture",
                "--goldens", str(goldens_path),
                "--tag", "faq",
                "--tag", "generated",
                "--num-questions", "1",
            ],
            catch_exceptions=False,
        )

        with open(goldens_path) as f:
            data = yaml.safe_load(f)

        assert data["questions"][0]["tags"] == ["faq", "generated"]
