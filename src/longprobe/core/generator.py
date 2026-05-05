"""LLM-based question generation for LongProbe.

Uses ``litellm`` to generate realistic test questions from document content
via any supported LLM provider (OpenAI, Anthropic, Google Gemini, Ollama,
etc.).

Typical usage::

    from longprobe.config import GeneratorConfig
    from longprobe.core.generator import QuestionGenerator

    config = GeneratorConfig(provider="openai", model="gpt-4o-mini")
    gen = QuestionGenerator(config)
    questions = gen.generate([("doc1.txt", "Some content...")], num_questions=10)
    for q in questions:
        print(q)
"""

from __future__ import annotations

import logging
import re

from ..config import GeneratorConfig

logger = logging.getLogger(__name__)

# Rough token estimate: 1 token ≈ 4 characters.
_CHARS_PER_TOKEN = 4

# Maximum characters per chunk sent to the LLM (~3000 tokens).
_MAX_CHUNK_CHARS = 3000 * _CHARS_PER_TOKEN

# System prompt for question generation.
_SYSTEM_PROMPT = """\
You are a test engineer generating questions for a RAG (Retrieval-Augmented \
Generation) system. Based on the following document content, generate \
realistic questions that a user might ask about this information.

Requirements:
- Questions should be diverse and cover different topics in the content
- Questions should be specific enough that the answer is contained in the content
- Questions should be natural, as if asked by a real user
- Do NOT include answers, only questions
- Do NOT number the questions
- Output exactly one question per line
- Do not add any prefix, suffix, or extra formatting"""

# User prompt template.
_USER_PROMPT_TEMPLATE = """\
Generate {num_questions} questions about the following content:

---
{content}
---

Remember: one question per line, no numbering, no extra text."""


class QuestionGenerator:
    """Generate test questions from documents using an LLM.

    Args:
        config: Generator configuration (provider, model, API key, etc.).
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self._config = config
        self._litellm_checked = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        documents: list[tuple[str, str]],
        num_questions: int | None = None,
    ) -> list[str]:
        """Generate questions from a list of documents.

        Documents are chunked to fit within the model's context window.
        Questions from all chunks are merged and deduplicated.

        Args:
            documents: List of ``(filename, text)`` tuples.
            num_questions: Number of questions to generate.  Uses config
                default if not specified.

        Returns:
            List of question strings.

        Raises:
            ImportError: If ``litellm`` is not installed.
            RuntimeError: If the LLM API call fails.
        """
        self._ensure_litellm()

        n = num_questions or self._config.num_questions

        # Concatenate all documents with source headers.
        full_text = self._concatenate_documents(documents)

        if not full_text.strip():
            logger.warning("No document text to generate questions from.")
            return []

        # Split into chunks.
        chunks = self._chunk_text(full_text, _MAX_CHUNK_CHARS)
        logger.info(
            "Generating %d questions from %d document(s) in %d chunk(s).",
            n, len(documents), len(chunks),
        )

        # Distribute questions across chunks.
        questions_per_chunk = max(1, n // len(chunks))
        all_questions: list[str] = []

        for i, chunk in enumerate(chunks):
            # For the last chunk, generate remaining questions.
            remaining = n - len(all_questions)
            if i == len(chunks) - 1:
                chunk_target = max(1, remaining)
            else:
                chunk_target = min(questions_per_chunk, n - len(all_questions))

            if chunk_target <= 0:
                break

            chunk_questions = self._generate_for_chunk(chunk, chunk_target)
            all_questions.extend(chunk_questions)

        # Deduplicate (exact match, case-insensitive).
        all_questions = self._deduplicate(all_questions)

        # Trim to exact count.
        return all_questions[:n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_litellm(self) -> None:
        """Lazily check that litellm is available."""
        if self._litellm_checked:
            return
        try:
            import litellm  # noqa: F401
            self._litellm_checked = True
        except ImportError:
            raise ImportError(
                "The 'litellm' package is required for question generation. "
                "Install it with:  pip install litellm\n"
                "Or install LongProbe with:  pip install longprobe[generate]"
            )

    def _generate_for_chunk(self, chunk: str, num_questions: int) -> list[str]:
        """Call the LLM to generate questions from a single chunk."""
        import litellm

        cfg = self._config
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            num_questions=num_questions,
            content=chunk,
        )

        kwargs: dict = {
            "model": f"{cfg.provider}/{cfg.model}" if cfg.provider != "openai" else cfg.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }

        # API key.
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key

        # Custom base URL (for Ollama, etc.).
        if cfg.base_url:
            kwargs["api_base"] = cfg.base_url

        # For non-openai providers that need the model in litellm format.
        if cfg.provider == "ollama":
            kwargs["model"] = f"ollama/{cfg.model}"
            if not cfg.base_url:
                kwargs["api_base"] = "http://localhost:11434"
        elif cfg.provider == "anthropic":
            kwargs["model"] = f"anthropic/{cfg.model}"
        elif cfg.provider == "gemini":
            kwargs["model"] = f"gemini/{cfg.model}"
        elif cfg.provider not in ("openai",):
            # Generic litellm provider prefix.
            kwargs["model"] = f"{cfg.provider}/{cfg.model}"

        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"LLM API call failed: {exc}\n"
                f"Provider: {cfg.provider}, Model: {cfg.model}"
            ) from exc

        # Extract text from response.
        try:
            content = response.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected LLM response format: {exc}"
            ) from exc

        return self._parse_questions(content)

    @staticmethod
    def _concatenate_documents(documents: list[tuple[str, str]]) -> str:
        """Merge documents into a single text with source headers."""
        parts: list[str] = []
        for filename, text in documents:
            if text.strip():
                parts.append(f"[Source: {filename}]\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _chunk_text(text: str, max_chars: int) -> list[str]:
        """Split text into chunks of roughly *max_chars* characters.

        Tries to split on paragraph boundaries (double newlines) to keep
        sentences together.
        """
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        current = ""

        # Split on double newlines (paragraph boundaries).
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            if len(current) + len(para) + 2 <= max_chars:
                current = current + "\n\n" + para if current else para
            else:
                if current:
                    chunks.append(current)
                # If a single paragraph is too long, split by characters.
                if len(para) > max_chars:
                    for i in range(0, len(para), max_chars):
                        chunks.append(para[i : i + max_chars])
                    current = ""
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks if chunks else [text[:max_chars]]

    @staticmethod
    def _parse_questions(raw: str) -> list[str]:
        """Parse LLM output into a list of question strings.

        Handles numbered lists (``1. What is...?``), bullet points, and
        plain one-per-line format.
        """
        lines = raw.strip().splitlines()
        questions: list[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Strip common prefixes: "1. ", "1) ", "- ", "* ".
            cleaned = re.sub(r'^[\d]+[.)]\s*', '', line)
            cleaned = re.sub(r'^[-*]\s*', '', cleaned)

            if cleaned:
                questions.append(cleaned)

        return questions

    @staticmethod
    def _deduplicate(questions: list[str]) -> list[str]:
        """Remove duplicate questions (case-insensitive)."""
        seen: set[str] = set()
        unique: list[str] = []
        for q in questions:
            key = q.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(q)
        return unique
