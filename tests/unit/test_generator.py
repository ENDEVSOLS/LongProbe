"""Tests for ``longprobe.core.generator`` — QuestionGenerator.

Covers question generation with mocked litellm, document chunking,
output parsing, deduplication, and error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from longprobe.config import GeneratorConfig
from longprobe.core.generator import QuestionGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gen_config() -> GeneratorConfig:
    """Return a basic generator config."""
    return GeneratorConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key-123",
        num_questions=5,
        temperature=0.7,
    )


@pytest.fixture
def generator(gen_config: GeneratorConfig) -> QuestionGenerator:
    """Return a QuestionGenerator with test config."""
    return QuestionGenerator(gen_config)


def _mock_response(text: str) -> MagicMock:
    """Create a mock litellm response with the given content."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# _parse_questions tests
# ---------------------------------------------------------------------------

class TestParseQuestions:
    """Tests for QuestionGenerator._parse_questions."""

    def test_plain_lines(self) -> None:
        """Plain one-per-line format."""
        raw = "What is Python?\nWhat is Java?\nHow does GC work?"
        result = QuestionGenerator._parse_questions(raw)
        assert result == [
            "What is Python?",
            "What is Java?",
            "How does GC work?",
        ]

    def test_numbered_list(self) -> None:
        """Numbered list with '1. ' prefix."""
        raw = "1. What is Python?\n2. What is Java?\n3. How does GC work?"
        result = QuestionGenerator._parse_questions(raw)
        assert result == [
            "What is Python?",
            "What is Java?",
            "How does GC work?",
        ]

    def test_paren_numbered_list(self) -> None:
        """Numbered list with '1) ' prefix."""
        raw = "1) What is Python?\n2) What is Java?"
        result = QuestionGenerator._parse_questions(raw)
        assert result == ["What is Python?", "What is Java?"]

    def test_bullet_list(self) -> None:
        """Bullet list with '- ' prefix."""
        raw = "- What is Python?\n- What is Java?"
        result = QuestionGenerator._parse_questions(raw)
        assert result == ["What is Python?", "What is Java?"]

    def test_asterisk_list(self) -> None:
        """Bullet list with '* ' prefix."""
        raw = "* What is Python?\n* What is Java?"
        result = QuestionGenerator._parse_questions(raw)
        assert result == ["What is Python?", "What is Java?"]

    def test_empty_lines_ignored(self) -> None:
        """Empty lines are skipped."""
        raw = "What is Python?\n\n\nWhat is Java?"
        result = QuestionGenerator._parse_questions(raw)
        assert len(result) == 2

    def test_trailing_whitespace(self) -> None:
        """Trailing whitespace is stripped."""
        raw = "What is Python?   \n  What is Java?  "
        result = QuestionGenerator._parse_questions(raw)
        assert result == ["What is Python?", "What is Java?"]


# ---------------------------------------------------------------------------
# _deduplicate tests
# ---------------------------------------------------------------------------

class TestDeduplicate:
    """Tests for QuestionGenerator._deduplicate."""

    def test_no_duplicates(self) -> None:
        """List without duplicates is unchanged."""
        qs = ["What is Python?", "What is Java?"]
        result = QuestionGenerator._deduplicate(qs)
        assert result == qs

    def test_exact_duplicates(self) -> None:
        """Exact duplicates are removed."""
        qs = ["What is Python?", "What is Python?", "What is Java?"]
        result = QuestionGenerator._deduplicate(qs)
        assert len(result) == 2

    def test_case_insensitive_duplicates(self) -> None:
        """Case-insensitive duplicates are removed."""
        qs = ["What is Python?", "what is python?", "WHAT IS PYTHON?"]
        result = QuestionGenerator._deduplicate(qs)
        assert len(result) == 1

    def test_preserves_first_occurrence(self) -> None:
        """First occurrence is preserved."""
        qs = ["What is Python?", "what is python?"]
        result = QuestionGenerator._deduplicate(qs)
        assert result[0] == "What is Python?"


# ---------------------------------------------------------------------------
# _chunk_text tests
# ---------------------------------------------------------------------------

class TestChunkText:
    """Tests for QuestionGenerator._chunk_text."""

    def test_short_text_single_chunk(self) -> None:
        """Text shorter than max_chars stays as one chunk."""
        text = "Short text"
        chunks = QuestionGenerator._chunk_text(text, 1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_split(self) -> None:
        """Text longer than max_chars is split."""
        text = "Word " * 5000  # ~25000 chars
        chunks = QuestionGenerator._chunk_text(text, 1000)
        assert len(chunks) > 1

    def test_paragraph_boundaries(self) -> None:
        """Splitting prefers paragraph boundaries."""
        para1 = "A" * 500
        para2 = "B" * 500
        text = f"{para1}\n\n{para2}"
        chunks = QuestionGenerator._chunk_text(text, 800)
        # Should split at paragraph boundary, not mid-paragraph.
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# _concatenate_documents tests
# ---------------------------------------------------------------------------

class TestConcatenateDocuments:
    """Tests for QuestionGenerator._concatenate_documents."""

    def test_single_document(self) -> None:
        """Single document gets a source header."""
        result = QuestionGenerator._concatenate_documents(
            [("doc1.txt", "Hello world")]
        )
        assert "[Source: doc1.txt]" in result
        assert "Hello world" in result

    def test_multiple_documents(self) -> None:
        """Multiple documents are separated."""
        result = QuestionGenerator._concatenate_documents(
            [("doc1.txt", "Content A"), ("doc2.txt", "Content B")]
        )
        assert "[Source: doc1.txt]" in result
        assert "[Source: doc2.txt]" in result

    def test_empty_documents_skipped(self) -> None:
        """Empty documents are not included."""
        result = QuestionGenerator._concatenate_documents(
            [("empty.txt", ""), ("doc.txt", "Content")]
        )
        assert "[Source: empty.txt]" not in result
        assert "[Source: doc.txt]" in result


# ---------------------------------------------------------------------------
# generate() tests (with mocked litellm)
# ---------------------------------------------------------------------------

class TestGenerate:
    """Tests for QuestionGenerator.generate with mocked LLM calls."""

    def test_basic_generation(
        self, generator: QuestionGenerator
    ) -> None:
        """Basic generation returns questions from LLM."""
        with patch.object(generator, "_generate_for_chunk", return_value=[
            "What is the refund policy?",
            "How long does shipping take?",
            "What are the pricing tiers?",
        ]):
            generator._litellm_checked = True
            questions = generator.generate(
                [("doc.txt", "Some document content about shipping and returns.")],
                num_questions=3,
            )

        assert len(questions) == 3
        assert "What is the refund policy?" in questions

    def test_deduplication_across_chunks(
        self, generator: QuestionGenerator
    ) -> None:
        """Duplicate questions across chunks are removed."""
        with patch.object(generator, "_generate_for_chunk", return_value=[
            "What is Python?",
            "What is Java?",
        ]):
            generator._litellm_checked = True
            # Create a long document that will be chunked.
            long_doc = ("Word " * 5000) + " test content"
            questions = generator.generate(
                [("doc.txt", long_doc)],
                num_questions=5,
            )

        # All returned questions should be unique.
        assert len(questions) == len(set(q.lower() for q in questions))

    def test_trims_to_requested_count(
        self, generator: QuestionGenerator
    ) -> None:
        """Output is trimmed to exactly num_questions."""
        with patch.object(generator, "_generate_for_chunk", return_value=[
            "Q1?", "Q2?", "Q3?", "Q4?", "Q5?", "Q6?", "Q7?", "Q8?", "Q9?", "Q10?",
        ]):
            generator._litellm_checked = True
            questions = generator.generate(
                [("doc.txt", "content")],
                num_questions=3,
            )
        assert len(questions) == 3

    def test_empty_documents_returns_empty(self, generator: QuestionGenerator) -> None:
        """Empty document list returns empty question list."""
        generator._litellm_checked = True
        questions = generator.generate([], num_questions=5)
        assert questions == []

    def test_litellm_not_installed_raises(self, gen_config: GeneratorConfig) -> None:
        """ImportError when litellm is not installed."""
        with patch.dict("sys.modules", {"litellm": None}):
            gen = QuestionGenerator(gen_config)
            with pytest.raises(ImportError, match="litellm"):
                gen._ensure_litellm()

    def test_llm_api_error_raises(
        self, generator: QuestionGenerator
    ) -> None:
        """RuntimeError when LLM API call fails."""
        with patch.object(generator, "_generate_for_chunk", side_effect=RuntimeError("LLM API call failed: rate limit")):
            generator._litellm_checked = True
            with pytest.raises(RuntimeError, match="LLM API call failed"):
                generator.generate(
                    [("doc.txt", "content")],
                    num_questions=3,
                )


# ---------------------------------------------------------------------------
# Provider routing tests
# ---------------------------------------------------------------------------

class TestProviderRouting:
    """Tests that different providers set the correct model prefix."""

    def _run_generate_for_chunk(self, config: GeneratorConfig) -> dict:
        """Helper: call _generate_for_chunk with a mocked litellm.completion."""
        gen = QuestionGenerator(config)
        gen._litellm_checked = True

        # Create a fake litellm module with a mock completion function.
        mock_completion = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "What is X?"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_completion.return_value = mock_response

        # Inject it before the import inside _generate_for_chunk.
        import sys
        fake_litellm = MagicMock()
        fake_litellm.completion = mock_completion
        sys.modules["litellm"] = fake_litellm

        try:
            gen._generate_for_chunk("content", 1)
            return mock_completion.call_args.kwargs
        finally:
            del sys.modules["litellm"]

    def test_ollama_provider(self) -> None:
        """Ollama provider sets correct model prefix and api_base."""
        config = GeneratorConfig(provider="ollama", model="llama3", api_key="")
        kwargs = self._run_generate_for_chunk(config)
        assert kwargs["model"] == "ollama/llama3"
        assert "api_base" in kwargs

    def test_anthropic_provider(self) -> None:
        """Anthropic provider sets correct model prefix."""
        config = GeneratorConfig(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            api_key="sk-test",
        )
        kwargs = self._run_generate_for_chunk(config)
        assert kwargs["model"] == "anthropic/claude-3-haiku-20240307"

    def test_gemini_provider(self) -> None:
        """Gemini provider sets correct model prefix."""
        config = GeneratorConfig(provider="gemini", model="gemini-pro", api_key="test")
        kwargs = self._run_generate_for_chunk(config)
        assert kwargs["model"] == "gemini/gemini-pro"

    def test_custom_base_url(self) -> None:
        """Custom base_url is passed to litellm."""
        config = GeneratorConfig(
            provider="openai",
            model="my-model",
            api_key="key",
            base_url="http://my-server:8080/v1",
        )
        kwargs = self._run_generate_for_chunk(config)
        assert kwargs["api_base"] == "http://my-server:8080/v1"


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Tests for GeneratorConfig loading through ProbeConfig."""

    def test_load_generator_config(self) -> None:
        """Generator config loads from YAML-like dict."""
        from longprobe.config import ProbeConfig

        config = ProbeConfig.from_dict({
            "generator": {
                "provider": "anthropic",
                "model": "claude-3-haiku-20240307",
                "api_key": "${ANTHROPIC_API_KEY}",
                "num_questions": 25,
                "temperature": 0.5,
            },
        })

        assert config.generator.provider == "anthropic"
        assert config.generator.model == "claude-3-haiku-20240307"
        assert config.generator.num_questions == 25
        assert config.generator.temperature == 0.5

    def test_env_var_expansion_in_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """${ENV_VAR} in api_key is expanded."""
        from longprobe.config import ProbeConfig

        monkeypatch.setenv("MY_LLM_KEY", "secret-456")

        config = ProbeConfig.from_dict({
            "generator": {
                "api_key": "${MY_LLM_KEY}",
            },
        })

        assert config.generator.api_key == "secret-456"

    def test_default_generator_config(self) -> None:
        """Default config has sensible generator defaults."""
        from longprobe.config import ProbeConfig

        config = ProbeConfig.defaults()
        assert config.generator.provider == "openai"
        assert config.generator.model == "gpt-4o-mini"
        assert config.generator.num_questions == 50
