"""
Recall scoring engine for LongProbe RAG regression testing.

Evaluates retrieval quality by comparing retrieved document chunks against
the expected (golden) set of chunks, computing per-question recall scores
and aggregating them into a comprehensive :class:`ProbeReport`.

Three match strategies are supported:

* **id** – exact string match on chunk identifiers.
* **text** – case-insensitive substring matching on document text.
* **semantic** – word-frequency cosine similarity with a configurable threshold.

Typical usage::

    scorer = RecallScorer(recall_threshold=0.8)
    report = scorer.score_all(golden_set, retriever_fn=my_retriever)
    if report.regression_detected:
        print("WARNING: recall regression detected!")
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .golden import GoldenQuestion, GoldenSet


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    """Scoring outcome for a single golden question.

    Attributes:
        question_id: Identifier of the golden question that was evaluated.
        question: The natural-language question text.
        recall_score: Fraction of required chunks that were found (0.0–1.0).
        retrieved_chunk_ids: IDs of the chunks returned by the retriever.
        required_chunks: The full list of chunks expected to be retrieved.
        missing_chunks: Required chunks that were *not* found.
        found_chunks: Required chunks that *were* found.
        passed: Whether recall met the configured threshold.
        latency_ms: Wall-clock time of the retrieval call in milliseconds.
    """

    question_id: str
    question: str
    recall_score: float
    retrieved_chunk_ids: list[str]
    required_chunks: list[str]
    missing_chunks: list[str]
    found_chunks: list[str]
    passed: bool
    latency_ms: float


@dataclass
class ProbeReport:
    """Aggregated report for an entire golden-set evaluation run.

    Attributes:
        golden_set_name: Name of the golden set that was evaluated.
        golden_set_version: Version string of the golden set.
        timestamp: ISO 8601 timestamp (UTC) of when the evaluation ran.
        overall_recall: Mean recall across all questions.
        pass_rate: Fraction of questions whose recall met the threshold.
        results: Per-question scoring details.
        baseline_recall: Previously recorded recall value (if available)
            used to detect regressions.
        recall_delta: Difference between current and baseline recall.
        regression_detected: Whether a statistically meaningful regression
            was observed (``recall_delta`` is negative).
    """

    golden_set_name: str
    golden_set_version: str
    timestamp: str
    overall_recall: float
    pass_rate: float
    results: list[QuestionResult]
    baseline_recall: float | None = None
    recall_delta: float | None = None
    regression_detected: bool = False


# ---------------------------------------------------------------------------
# RecallScorer
# ---------------------------------------------------------------------------


class RecallScorer:
    """Scores retrieval results against a golden set using configurable
    matching strategies.

    Args:
        recall_threshold: Minimum recall (0.0–1.0) for a question to be
            considered *passed*.
    """

    def __init__(self, recall_threshold: float = 0.8) -> None:
        if not 0.0 <= recall_threshold <= 1.0:
            raise ValueError(
                f"recall_threshold must be between 0.0 and 1.0, "
                f"got {recall_threshold}"
            )
        self.recall_threshold = recall_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        question: GoldenQuestion,
        retrieved_docs: list[dict],
    ) -> QuestionResult:
        """Score a single question against its retrieved documents.

        Args:
            question: The golden question containing expected chunks and
                matching configuration.
            retrieved_docs: List of dicts returned by the retriever, each
                with at least ``"id"`` and ``"text"`` keys.

        Returns:
            A :class:`QuestionResult` with recall and diagnostic details.
        """
        required_chunks = list(question.required_chunks)
        retrieved_ids = [doc.get("id", "") for doc in retrieved_docs]

        # Dispatch to the appropriate matching strategy.
        match_mode = question.match_mode

        if match_mode == "id":
            found = self._id_match(required_chunks, retrieved_ids)
        elif match_mode == "text":
            found = self._text_match(required_chunks, retrieved_docs)
        elif match_mode == "semantic":
            found = self._semantic_match(
                required_chunks, retrieved_docs, question.semantic_threshold
            )
        else:
            raise ValueError(
                f"Unsupported match_mode '{match_mode}'. "
                f"Expected 'id', 'text', or 'semantic'."
            )

        # Compute recall with division-by-zero guard.
        if len(required_chunks) == 0:
            recall = 1.0
        else:
            recall = len(found) / len(required_chunks)

        missing_chunks = [c for c in required_chunks if c not in found]

        return QuestionResult(
            question_id=question.id,
            question=question.question,
            recall_score=recall,
            retrieved_chunk_ids=retrieved_ids,
            required_chunks=required_chunks,
            missing_chunks=missing_chunks,
            found_chunks=found,
            passed=recall >= self.recall_threshold,
            latency_ms=0.0,
        )

    def score_all(
        self,
        golden_set: GoldenSet,
        retriever_fn: Callable[[str, int], list[dict]],
        top_k_override: int | None = None,
    ) -> ProbeReport:
        """Score every question in a golden set and produce an aggregated report.

        Args:
            golden_set: The golden set to evaluate.
            retriever_fn: A callable ``(query, top_k) -> list[dict]`` that
                invokes the retrieval pipeline under test.
            top_k_override: If set, overrides the per-question ``top_k``
                value for every question in the golden set.

        Returns:
            A :class:`ProbeReport` summarising the evaluation run.
        """
        results: list[QuestionResult] = []

        for question in golden_set.questions:
            top_k = top_k_override if top_k_override is not None else question.top_k

            start = time.perf_counter()
            retrieved_docs = retriever_fn(question.question, top_k)
            elapsed = time.perf_counter() - start

            result = self.score(question, retrieved_docs)
            result.latency_ms = elapsed * 1000.0
            results.append(result)

        # Aggregate metrics.
        if results:
            overall_recall = sum(r.recall_score for r in results) / len(results)
            pass_rate = sum(1 for r in results if r.passed) / len(results)
        else:
            overall_recall = 1.0
            pass_rate = 1.0

        return ProbeReport(
            golden_set_name=golden_set.name,
            golden_set_version=golden_set.version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_recall=overall_recall,
            pass_rate=pass_rate,
            results=results,
        )

    # ------------------------------------------------------------------
    # Match strategies
    # ------------------------------------------------------------------

    def _id_match(
        self,
        required_chunks: list[str],
        retrieved_ids: list[str],
    ) -> list[str]:
        """Exact string match on chunk identifiers.

        Args:
            required_chunks: Chunk IDs that are expected.
            retrieved_ids: Chunk IDs returned by the retriever.

        Returns:
            Subset of *required_chunks* that appear in *retrieved_ids*.
        """
        retrieved_set = set(retrieved_ids)
        return [c for c in required_chunks if c in retrieved_set]

    def _text_match(
        self,
        required_chunks: list[str],
        retrieved_docs: list[dict],
    ) -> list[str]:
        """Case-insensitive, whitespace-normalised substring matching.

        For each required chunk text, check whether it appears as a
        substring of any retrieved document's ``"text"`` field after
        both sides are lowercased and normalised for whitespace.

        Args:
            required_chunks: Chunk texts that are expected.
            retrieved_docs: Documents returned by the retriever.

        Returns:
            Subset of *required_chunks* that matched at least one doc.
        """
        # Normalise all retrieved doc texts once.
        normalised_docs: list[str] = []
        for doc in retrieved_docs:
            text = doc.get("text", "")
            normalised_docs.append(" ".join(text.lower().split()))

        found: list[str] = []
        for chunk in required_chunks:
            normalised_chunk = " ".join(chunk.lower().split())
            if any(normalised_chunk in doc_text for doc_text in normalised_docs):
                found.append(chunk)

        return found

    def _semantic_match(
        self,
        required_chunks: list[str],
        retrieved_docs: list[dict],
        threshold: float,
    ) -> list[str]:
        """Word-frequency cosine similarity matching.

        For each required chunk, compute cosine similarity against every
        retrieved document's text. If the maximum similarity across all
        retrieved docs meets or exceeds *threshold*, the chunk is
        considered found.

        Args:
            required_chunks: Chunk texts that are expected.
            retrieved_docs: Documents returned by the retriever.
            threshold: Minimum cosine similarity to count as a match.

        Returns:
            Subset of *required_chunks* that met the similarity threshold.
        """
        retrieved_texts: list[str] = [doc.get("text", "") for doc in retrieved_docs]

        found: list[str] = []
        for chunk in required_chunks:
            max_sim = 0.0
            for rtext in retrieved_texts:
                sim = self._cosine_similarity(chunk, rtext)
                if sim > max_sim:
                    max_sim = sim
            if max_sim >= threshold:
                found.append(chunk)

        return found

    # ------------------------------------------------------------------
    # Similarity helper
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(text1: str, text2: str) -> float:
        """Compute word-frequency cosine similarity between two texts.

        Uses a simple bag-of-words representation: texts are tokenised on
        whitespace, word counts are tallied, and the cosine of the angle
        between the resulting count vectors is returned.

        No external dependencies (numpy, scipy, …) are used.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Cosine similarity in the range [0.0, 1.0]. Returns ``0.0`` if
            either text is empty or contains no words.
        """
        words1 = text1.lower().split()
        words2 = text2.lower().split()

        if not words1 or not words2:
            return 0.0

        counter1 = Counter(words1)
        counter2 = Counter(words2)

        # Dot product over the shared vocabulary.
        dot = sum(counter1[w] * counter2[w] for w in counter1 if w in counter2)

        # Magnitudes.
        mag1 = math.sqrt(sum(v * v for v in counter1.values()))
        mag2 = math.sqrt(sum(v * v for v in counter2.values()))

        if mag1 == 0.0 or mag2 == 0.0:
            return 0.0

        return dot / (mag1 * mag2)
