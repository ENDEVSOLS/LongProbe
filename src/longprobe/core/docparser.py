"""Document text extraction for LongProbe question generation.

Reads files from disk and extracts plain text from a variety of formats
so they can be sent to an LLM for question generation.  Supports two
strategies:

* **markitdown** (preferred): Handles PDF, DOCX, PPTX, XLSX, HTML, images,
  and many more formats via the ``markitdown`` library.
* **Plain-text fallback**: Direct ``open().read()`` for ``.txt``, ``.md``,
  ``.csv``, ``.json`` and other text-based extensions when ``markitdown``
  is not available.

Typical usage::

    parser = DocumentParser()
    docs = parser.parse_directory("./docs")
    for filename, text in docs:
        print(f"{filename}: {len(text)} chars")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Extensions we can always read as plain text (no markitdown needed).
_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl",
    ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".htm", ".css",
    ".rst", ".log", ".sh", ".bash", ".zsh", ".sql", ".go", ".rs",
    ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php",
})

# Binary extensions that require markitdown or similar.
_BINARY_EXTENSIONS = frozenset({
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".odt", ".ods", ".odp", ".rtf", ".epub",
})


class DocumentParser:
    """Extract text from files in various formats.

    Uses ``markitdown`` when available for rich format support (PDF, DOCX,
    etc.).  Falls back to direct text reading for common text-based formats.
    """

    def __init__(self) -> None:
        self._markitdown_available: bool | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(self, path: str) -> str:
        """Extract text from a single file.

        Args:
            path: Filesystem path to the file.

        Returns:
            Extracted text as a string.  Returns an empty string if the
            file cannot be parsed.
        """
        file_path = Path(path)

        if not file_path.is_file():
            logger.warning("File not found: %s", path)
            return ""

        ext = file_path.suffix.lower()

        # Try markitdown first for any file type (it handles both text and binary).
        if self._is_markitdown_available():
            return self._parse_with_markitdown(file_path)

        # Fallback: direct text read for text-based files.
        if ext in _TEXT_EXTENSIONS:
            return self._read_text(file_path)

        # Warn for binary files without markitdown.
        if ext in _BINARY_EXTENSIONS:
            logger.warning(
                "Skipping binary file '%s' — install markitdown for "
                "PDF/DOCX/PPTX support: pip install markitdown",
                path,
            )
            return ""

        # Unknown extension: try text read as last resort.
        return self._read_text(file_path)

    def parse_directory(self, path: str) -> List[Tuple[str, str]]:
        """Recursively parse all supported files in a directory.

        Args:
            path: Filesystem path to the directory.

        Returns:
            List of ``(filename, extracted_text)`` tuples for files that
            yielded non-empty text.
        """
        dir_path = Path(path)

        if not dir_path.is_dir():
            logger.warning("Not a directory: %s", path)
            return []

        results: List[Tuple[str, str]] = []

        for file_path in sorted(dir_path.rglob("*")):
            if not file_path.is_file():
                continue

            # Skip hidden files and common non-document patterns.
            if any(part.startswith(".") for part in file_path.parts):
                continue

            text = self.parse_file(str(file_path))
            if text.strip():
                results.append((str(file_path), text))
            else:
                logger.debug("No text extracted from: %s", file_path)

        return results

    def parse_path(self, path: str) -> List[Tuple[str, str]]:
        """Parse a file or directory.

        If *path* is a file, returns a single-element list.
        If *path* is a directory, recursively parses all files.

        Args:
            path: Filesystem path to a file or directory.

        Returns:
            List of ``(filename, extracted_text)`` tuples.
        """
        if os.path.isfile(path):
            text = self.parse_file(path)
            if text.strip():
                return [(path, text)]
            return []
        elif os.path.isdir(path):
            return self.parse_directory(path)
        else:
            logger.warning("Path does not exist: %s", path)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_markitdown_available(self) -> bool:
        """Check if markitdown is importable (cached after first check)."""
        if self._markitdown_available is None:
            try:
                import markitdown  # noqa: F401
                self._markitdown_available = True
            except ImportError:
                self._markitdown_available = False
        return self._markitdown_available

    def _parse_with_markitdown(self, file_path: Path) -> str:
        """Extract text using markitdown."""
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            result = md.convert(str(file_path))
            return result.text_content or ""
        except Exception as exc:
            logger.warning(
                "markitdown failed to parse '%s': %s. "
                "Falling back to text read.",
                file_path,
                exc,
            )
            # Fallback to text read.
            return self._read_text(file_path)

    @staticmethod
    def _read_text(file_path: Path) -> str:
        """Read a file as UTF-8 text."""
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Could not read file '%s': %s", file_path, exc)
            return ""
