"""Tests for ``longprobe.core.docparser`` — DocumentParser.

Covers text file parsing, directory scanning, markitdown fallback,
and edge cases (missing files, empty directories, binary files).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from longprobe.core.docparser import DocumentParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser() -> DocumentParser:
    """Return a DocumentParser with markitdown disabled (for unit tests)."""
    p = DocumentParser()
    p._markitdown_available = False
    return p


@pytest.fixture
def tmp_dir() -> Path:
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # Text files.
        (td_path / "doc1.txt").write_text("Hello world from doc1", encoding="utf-8")
        (td_path / "doc2.md").write_text("# Markdown Title\n\nSome content.", encoding="utf-8")
        (td_path / "data.csv").write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
        (td_path / "config.json").write_text('{"key": "value"}', encoding="utf-8")

        # Nested directory.
        sub = td_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested content", encoding="utf-8")

        # Hidden file (should be skipped).
        (td_path / ".hidden").write_text("hidden content", encoding="utf-8")

        yield td_path


# ---------------------------------------------------------------------------
# parse_file tests
# ---------------------------------------------------------------------------

class TestParseFile:
    """Tests for DocumentParser.parse_file."""

    def test_parse_txt_file(self, parser: DocumentParser, tmp_dir: Path) -> None:
        """Plain .txt file is read correctly."""
        result = parser.parse_file(str(tmp_dir / "doc1.txt"))
        assert "Hello world from doc1" in result

    def test_parse_md_file(self, parser: DocumentParser, tmp_dir: Path) -> None:
        """Markdown file is read correctly."""
        result = parser.parse_file(str(tmp_dir / "doc2.md"))
        assert "Markdown Title" in result

    def test_parse_csv_file(self, parser: DocumentParser, tmp_dir: Path) -> None:
        """CSV file is read correctly."""
        result = parser.parse_file(str(tmp_dir / "data.csv"))
        assert "Alice" in result

    def test_parse_json_file(self, parser: DocumentParser, tmp_dir: Path) -> None:
        """JSON file is read correctly."""
        result = parser.parse_file(str(tmp_dir / "config.json"))
        assert '"key"' in result

    def test_parse_missing_file(self, parser: DocumentParser) -> None:
        """Missing file returns empty string."""
        result = parser.parse_file("/nonexistent/file.txt")
        assert result == ""

    def test_parse_binary_file_without_markitdown(
        self, parser: DocumentParser, tmp_dir: Path
    ) -> None:
        """Binary file returns empty string with warning when markitdown unavailable."""
        pdf_path = tmp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")
        result = parser.parse_file(str(pdf_path))
        assert result == ""

    @patch("longprobe.core.docparser.DocumentParser._is_markitdown_available")
    def test_parse_with_markitdown(
        self, mock_available: MagicMock, tmp_dir: Path
    ) -> None:
        """When markitdown is available, it is used for parsing."""
        mock_available.return_value = True
        p = DocumentParser()

        with patch.object(p, "_parse_with_markitdown", return_value="Parsed content") as mock_md:
            result = p.parse_file(str(tmp_dir / "doc1.txt"))
            assert result == "Parsed content"
            mock_md.assert_called_once()

    def test_parse_empty_file(self, parser: DocumentParser, tmp_dir: Path) -> None:
        """Empty file returns empty string."""
        empty = tmp_dir / "empty.txt"
        empty.write_text("", encoding="utf-8")
        result = parser.parse_file(str(empty))
        assert result == ""


# ---------------------------------------------------------------------------
# parse_directory tests
# ---------------------------------------------------------------------------

class TestParseDirectory:
    """Tests for DocumentParser.parse_directory."""

    def test_parse_directory_finds_files(
        self, parser: DocumentParser, tmp_dir: Path
    ) -> None:
        """All text files in directory are found."""
        results = parser.parse_directory(str(tmp_dir))
        filenames = [f for f, _ in results]
        assert any("doc1.txt" in f for f in filenames)
        assert any("doc2.md" in f for f in filenames)
        assert any("data.csv" in f in f for f in filenames)

    def test_parse_directory_recursive(
        self, parser: DocumentParser, tmp_dir: Path
    ) -> None:
        """Nested directories are scanned recursively."""
        results = parser.parse_directory(str(tmp_dir))
        filenames = [f for f, _ in results]
        assert any("nested.txt" in f for f in filenames)

    def test_parse_directory_skips_hidden(
        self, parser: DocumentParser, tmp_dir: Path
    ) -> None:
        """Hidden files are skipped."""
        results = parser.parse_directory(str(tmp_dir))
        filenames = [f for f, _ in results]
        assert not any(".hidden" in f for f in filenames)

    def test_parse_directory_skips_empty_files(
        self, parser: DocumentParser, tmp_dir: Path
    ) -> None:
        """Files with no extractable text are excluded from results."""
        empty = tmp_dir / "empty.txt"
        empty.write_text("", encoding="utf-8")
        results = parser.parse_directory(str(tmp_dir))
        filenames = [f for f, _ in results]
        assert not any("empty.txt" in f for f in filenames)

    def test_parse_nonexistent_directory(self, parser: DocumentParser) -> None:
        """Non-existent directory returns empty list."""
        results = parser.parse_directory("/nonexistent/dir")
        assert results == []

    def test_parse_empty_directory(self, parser: DocumentParser) -> None:
        """Empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as td:
            results = parser.parse_directory(td)
            assert results == []


# ---------------------------------------------------------------------------
# parse_path tests
# ---------------------------------------------------------------------------

class TestParsePath:
    """Tests for DocumentParser.parse_path (dispatches file vs directory)."""

    def test_parse_path_file(self, parser: DocumentParser, tmp_dir: Path) -> None:
        """Passing a file path returns a single-element list."""
        results = parser.parse_path(str(tmp_dir / "doc1.txt"))
        assert len(results) == 1
        assert "Hello world from doc1" in results[0][1]

    def test_parse_path_directory(self, parser: DocumentParser, tmp_dir: Path) -> None:
        """Passing a directory recursively parses files."""
        results = parser.parse_path(str(tmp_dir))
        assert len(results) >= 3

    def test_parse_path_nonexistent(self, parser: DocumentParser) -> None:
        """Non-existent path returns empty list."""
        results = parser.parse_path("/nonexistent/path")
        assert results == []


# ---------------------------------------------------------------------------
# Markitdown integration tests
# ---------------------------------------------------------------------------

class TestMarkitdownFallback:
    """Tests for markitdown availability and fallback behavior."""

    def test_markitdown_check_caches(self) -> None:
        """markitdown availability is cached after first check."""
        p = DocumentParser()
        # First check
        p._is_markitdown_available()
        # Second check should use cache
        assert p._markitdown_available is not None

    def test_markitdown_error_falls_back_to_text(self, tmp_dir: Path) -> None:
        """If markitdown raises, falls back to text read."""
        p = DocumentParser()
        p._markitdown_available = True  # Pretend markitdown is available

        # Mock the markitdown module to raise.
        with patch("longprobe.core.docparser.MarkItDown", create=True) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.convert.side_effect = Exception("markitdown crashed")
            mock_cls.return_value = mock_instance

            result = p.parse_file(str(tmp_dir / "doc1.txt"))
            # Should fall back to text read and get the content.
            assert "Hello world from doc1" in result
