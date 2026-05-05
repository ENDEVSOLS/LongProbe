"""
Tests for ``longprobe.core.scorer`` — RecallScorer, QuestionResult, ProbeReport.

Covers every match strategy (id, text, semantic), recall calculation edge
cases, threshold pass/fail logic, and the full ``score_all`` pipeline with a
mock retriever.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from longprobe.core.golden import GoldenQuestion, GoldenSet
from longprobe.core.scorer import ProbeReport, QuestionResult, RecallScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer() -> RecallScorer:
    """Return a RecallScorer with a typical threshold."""
    return RecallScorer(recall_threshold=0.8)


@pytest.fixture
def golden_set() -> GoldenSet:
    """Return a small GoldenSet with two id-mode questions."""
    return GoldenSet(
        name="test-suite",
        version="1.0.0",
        questions=[
            GoldenQuestion(
                id="q1",
                question="What is Python?",
                required_chunks=["chunk-a", "chunk-b", "chunk-c"],
                match_mode="id",
                top_k=5,
            ),
            GoldenQuestion(
                id="q2",
                question="What is Rust?",
                required_chunks=["chunk-d", "chunk-e"],
                match_mode="id",
                top_k=3,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# id match mode
# ---------------------------------------------------------------------------

class TestIdMatchMode:

    def test_exact_id_match(self, scorer):
        """All required chunk IDs are returned → perfect recall."""
        q = GoldenQuestion(
            id="q1",
            question="test?",
            required_chunks=["c1", "c2", "c3"],
            match_mode="id",
        )
        retrieved = [
            {"id": "c1", "text": "first"},
            {"id": "c2", "text": "second"},
            {"id": "c3", "text": "third"},
        ]
        result = scorer.score(q, retrieved)

        assert result.recall_score == 1.0
        assert result.found_chunks == ["c1", "c2", "c3"]
        assert result.missing_chunks == []

    def test_partial_id_match(self, scorer):
        """Only some required IDs are returned → partial recall."""
        q = GoldenQuestion(
            id="q2",
            question="test?",
            required_chunks=["c1", "c2", "c3"],
            match_mode="id",
        )
        retrieved = [
            {"id": "c1", "text": "first"},
            {"id": "other", "text": "irrelevant"},
        ]
        result = scorer.score(q, retrieved)

        assert result.recall_score == pytest.approx(1.0 / 3.0)
        assert result.found_chunks == ["c1"]
        assert set(result.missing_chunks) == {"c2", "c3"}

    def test_no_id_match(self, scorer):
        """None of the required IDs are returned → zero recall."""
        q = GoldenQuestion(
            id="q3",
            question="test?",
            required_chunks=["c1", "c2"],
            match_mode="id",
        )
        retrieved = [
            {"id": "x1", "text": "wrong"},
            {"id": "x2", "text": "also wrong"},
        ]
        result = scorer.score(q, retrieved)

        assert result.recall_score == 0.0
        assert result.found_chunks == []
        assert result.missing_chunks == ["c1", "c2"]


# ---------------------------------------------------------------------------
# text match mode
# ---------------------------------------------------------------------------

class TestTextMatchMode:

    def test_case_insensitive_substring_match(self, scorer):
        """Match is case-insensitive and whitespace-tolerant."""
        q = GoldenQuestion(
            id="q1",
            question="test?",
            required_chunks=["Python is great"],
            match_mode="text",
        )
        retrieved = [
            {"id": "d1", "text": "I think PYTHON IS GREAT for scripting"},
        ]
        result = scorer.score(q, retrieved)

        assert result.recall_score == 1.0
        assert result.found_chunks == ["Python is great"]

    def test_text_not_found(self, scorer):
        """No substring match → chunk is missing."""
        q = GoldenQuestion(
            id="q2",
            question="test?",
            required_chunks=["Rust is blazingly fast"],
            match_mode="text",
        )
        retrieved = [
            {"id": "d1", "text": "Python is dynamically typed"},
        ]
        result = scorer.score(q, retrieved)

        assert result.recall_score == 0.0
        assert result.found_chunks == []

    def test_text_partial_match_among_many(self, scorer):
        """Only matching chunks are counted when there are several required."""
        q = GoldenQuestion(
            id="q3",
            question="test?",
            required_chunks=["hello world", "foo bar", "no match here"],
            match_mode="text",
        )
        retrieved = [
            {"id": "d1", "text": "the hello world program is classic"},
            {"id": "d2", "text": "foo bar baz"},
        ]
        result = scorer.score(q, retrieved)

        assert result.recall_score == pytest.approx(2.0 / 3.0)
        assert "hello world" in result.found_chunks
        assert "foo bar" in result.found_chunks
        assert "no match here" in result.missing_chunks


# ---------------------------------------------------------------------------
# semantic match mode
# ---------------------------------------------------------------------------

class TestSemanticMatchMode:

    def test_identical_text_high_similarity(self, scorer):
        """Identical text should have cosine similarity of 1.0."""
        q = GoldenQuestion(
            id="q1",
            question="test?",
            required_chunks=["the cat sat on the mat"],
            match_mode="semantic",
            semantic_threshold=0.95,
        )
        retrieved = [
            {"id": "d1", "text": "the cat sat on the mat"},
        ]
        result = scorer.score(q, retrieved)

        assert result.recall_score == 1.0

    def test_similar_text_above_threshold(self, scorer):
        """Texts with significant word overlap should exceed a low threshold."""
        q = GoldenQuestion(
            id="q2",
            question="test?",
            required_chunks=["machine learning algorithms process data"],
            match_mode="semantic",
            semantic_threshold=0.3,
        )
        # Significant overlap but not identical
        retrieved = [
            {"id": "d1", "text": "machine learning algorithms analyze large datasets"},
        ]
        result = scorer.score(q, retrieved)

        assert result.recall_score == 1.0

    def test_dissimilar_text_below_threshold(self, scorer):
        """Completely unrelated text should have low similarity."""
        q = GoldenQuestion(
            id="q3",
            question="test?",
            required_chunks=["quantum physics wave particle duality"],
            match_mode="semantic",
            semantic_threshold=0.5,
        )
        retrieved = [
            {"id": "d1", "text": "baking chocolate chip cookies is fun"},
        ]
        result = scorer.score(q, retrieved)

        assert result.recall_score == 0.0

    def test_empty_retrieved_docs_semantic(self, scorer):
        """No retrieved docs → zero recall in semantic mode."""
        q = GoldenQuestion(
            id="q4",
            question="test?",
            required_chunks=["some text"],
            match_mode="semantic",
            semantic_threshold=0.5,
        )
        result = scorer.score(q, [])

        assert result.recall_score == 0.0


# ---------------------------------------------------------------------------
# Recall calculation edge cases
# ---------------------------------------------------------------------------

class TestRecallCalculation:

    def test_zero_required_chunks_gives_recall_one(self, scorer):
        """Division-by-zero guard: 0 required chunks → recall of 1.0."""
        q = GoldenQuestion(
            id="q1",
            question="test?",
            required_chunks=[],
            match_mode="id",
        )
        result = scorer.score(q, [])

        assert result.recall_score == 1.0

    def test_all_found_gives_recall_one(self, scorer):
        q = GoldenQuestion(
            id="q2",
            question="test?",
            required_chunks=["a", "b", "c"],
            match_mode="id",
        )
        retrieved = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "extra"}]
        result = scorer.score(q, retrieved)

        assert result.recall_score == 1.0

    def test_none_found_gives_recall_zero(self, scorer):
        q = GoldenQuestion(
            id="q3",
            question="test?",
            required_chunks=["a", "b"],
            match_mode="id",
        )
        result = scorer.score(q, [{"id": "x"}])

        assert result.recall_score == 0.0

    def test_partial_recall_is_fraction(self, scorer):
        """recall = found / required for non-integer fractions."""
        q = GoldenQuestion(
            id="q4",
            question="test?",
            required_chunks=["a", "b", "c", "d", "e"],
            match_mode="id",
        )
        retrieved = [{"id": "a"}, {"id": "c"}, {"id": "e"}]
        result = scorer.score(q, retrieved)

        assert result.recall_score == pytest.approx(3.0 / 5.0)


# ---------------------------------------------------------------------------
# Threshold pass / fail
# ---------------------------------------------------------------------------

class TestThreshold:

    def test_recall_at_or_above_threshold_is_passed(self):
        scorer = RecallScorer(recall_threshold=0.5)
        q = GoldenQuestion(
            id="q1",
            question="test?",
            required_chunks=["a", "b"],
            match_mode="id",
        )
        # Found 1 of 2 → recall = 0.5, which equals the threshold
        result = scorer.score(q, [{"id": "a"}])

        assert result.passed is True

    def test_recall_below_threshold_is_failed(self):
        scorer = RecallScorer(recall_threshold=0.8)
        q = GoldenQuestion(
            id="q2",
            question="test?",
            required_chunks=["a", "b", "c"],
            match_mode="id",
        )
        # Found 1 of 3 → recall ≈ 0.33 < 0.8
        result = scorer.score(q, [{"id": "a"}])

        assert result.passed is False

    def test_perfect_recall_always_passes(self):
        scorer = RecallScorer(recall_threshold=1.0)
        q = GoldenQuestion(
            id="q3",
            question="test?",
            required_chunks=["x"],
            match_mode="id",
        )
        result = scorer.score(q, [{"id": "x"}])

        assert result.passed is True


# ---------------------------------------------------------------------------
# missing_chunks and found_chunks
# ---------------------------------------------------------------------------

class TestChunkTracking:

    def test_missing_chunks_identifies_unfound(self, scorer):
        q = GoldenQuestion(
            id="q1",
            question="test?",
            required_chunks=["a", "b", "c"],
            match_mode="id",
        )
        result = scorer.score(q, [{"id": "b"}])

        assert sorted(result.missing_chunks) == ["a", "c"]

    def test_found_chunks_identifies_found(self, scorer):
        q = GoldenQuestion(
            id="q2",
            question="test?",
            required_chunks=["a", "b", "c"],
            match_mode="id",
        )
        result = scorer.score(q, [{"id": "a"}, {"id": "c"}, {"id": "extra"}])

        assert sorted(result.found_chunks) == ["a", "c"]


# ---------------------------------------------------------------------------
# score_all with mock retriever
# ---------------------------------------------------------------------------

class TestScoreAll:

    def test_score_all_with_mock_retriever(self, scorer, golden_set):
        """Verify all fields in ProbeReport when using a mock retriever."""

        def mock_retriever(query: str, top_k: int) -> list[dict]:
            # For q1 ("What is Python?"), return 2 of 3 required chunks
            if "Python" in query:
                return [
                    {"id": "chunk-a", "text": "Python is ..."},
                    {"id": "chunk-b", "text": "CPython ..."},
                ]
            # For q2 ("What is Rust?"), return all required chunks
            return [
                {"id": "chunk-d", "text": "Rust is ..."},
                {"id": "chunk-e", "text": "Cargo ..."},
            ]

        report = scorer.score_all(golden_set, mock_retriever)

        # --- Basic structure checks ---
        assert isinstance(report, ProbeReport)
        assert report.golden_set_name == "test-suite"
        assert report.golden_set_version == "1.0.0"
        assert len(report.results) == 2

        # --- overall_recall ---
        # q1: 2/3 ≈ 0.6667, q2: 2/2 = 1.0 → mean = (0.6667 + 1.0) / 2
        expected_recall = (2.0 / 3.0 + 1.0) / 2.0
        assert report.overall_recall == pytest.approx(expected_recall, rel=1e-3)

        # --- pass_rate ---
        # q1 recall ≈ 0.667 < 0.8 → failed; q2 recall = 1.0 → passed
        assert report.pass_rate == pytest.approx(0.5)

        # --- Per-question results ---
        q1_result = report.results[0]
        assert q1_result.question_id == "q1"
        assert q1_result.recall_score == pytest.approx(2.0 / 3.0)
        assert q1_result.found_chunks == ["chunk-a", "chunk-b"]
        assert q1_result.missing_chunks == ["chunk-c"]
        assert q1_result.passed is False

        q2_result = report.results[1]
        assert q2_result.question_id == "q2"
        assert q2_result.recall_score == 1.0
        assert q2_result.found_chunks == ["chunk-d", "chunk-e"]
        assert q2_result.missing_chunks == []
        assert q2_result.passed is True

    def test_latency_ms_is_populated(self, scorer, golden_set):
        """latency_ms should be a positive float for each result."""

        def mock_retriever(query: str, top_k: int) -> list[dict]:
            return [{"id": "fake", "text": "lorem"}]

        report = scorer.score_all(golden_set, mock_retriever)

        for result in report.results:
            assert result.latency_ms >= 0.0
            assert isinstance(result.latency_ms, float)

    def test_timestamp_is_valid_iso_format(self, scorer, golden_set):
        """The report timestamp must be a parseable ISO 8601 string."""

        def mock_retriever(query: str, top_k: int) -> list[dict]:
            return []

        report = scorer.score_all(golden_set, mock_retriever)

        # datetime.fromisoformat should not raise
        parsed = datetime.fromisoformat(report.timestamp)
        assert parsed is not None

    def test_score_all_empty_golden_set(self, scorer):
        """An empty golden set should produce recall=1.0 and pass_rate=1.0."""
        empty_gs = GoldenSet(name="empty", version="0.0.0", questions=[])

        report = scorer.score_all(empty_gs, lambda q, k: [])

        assert report.overall_recall == 1.0
        assert report.pass_rate == 1.0
        assert report.results == []


# ---------------------------------------------------------------------------
# RecallScorer constructor validation
# ---------------------------------------------------------------------------

class TestScorerInit:

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="recall_threshold must be between"):
            RecallScorer(recall_threshold=1.5)

    def test_threshold_zero_ok(self):
        scorer = RecallScorer(recall_threshold=0.0)
        assert scorer.recall_threshold == 0.0

    def test_threshold_one_ok(self):
        scorer = RecallScorer(recall_threshold=1.0)
        assert scorer.recall_threshold == 1.0
