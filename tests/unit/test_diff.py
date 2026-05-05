"""
Tests for ``longprobe.core.diff`` — DiffReporter, RegressionDiff, and formatters.

Covers regression detection, improvement detection, unchanged questions,
chunk diff computation, overall delta, and all three output formats
(JSON, table, GitHub Actions).
"""

from __future__ import annotations

import json

import pytest

from longprobe.core.diff import (
    ChunkImprovement,
    ChunkRegression,
    DiffReporter,
    RegressionDiff,
)
from longprobe.core.scorer import ProbeReport, QuestionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_question_result(
    question_id: str,
    question: str,
    recall_score: float,
    found_chunks: list[str],
    retrieved_chunk_ids: list[str] | None = None,
    required_chunks: list[str] | None = None,
) -> QuestionResult:
    """Build a QuestionResult with sensible defaults for missing fields."""
    if retrieved_chunk_ids is None:
        retrieved_chunk_ids = list(found_chunks)
    if required_chunks is None:
        required_chunks = list(found_chunks) + ["missing-placeholder"]
        # Adjust required_chunks so recall_score is consistent
        if recall_score == 1.0:
            required_chunks = list(found_chunks)
        elif recall_score == 0.0:
            required_chunks = ["nonexistent-1", "nonexistent-2"]
            found_chunks = []
        else:
            # Derive required_chunks from found + missing
            pass  # use caller's found_chunks + required_chunks as-is

    missing_chunks = [c for c in required_chunks if c not in found_chunks]

    return QuestionResult(
        question_id=question_id,
        question=question,
        recall_score=recall_score,
        retrieved_chunk_ids=retrieved_chunk_ids,
        required_chunks=required_chunks,
        missing_chunks=missing_chunks,
        found_chunks=found_chunks,
        passed=recall_score >= 0.8,
        latency_ms=10.0,
    )


def _make_report(
    name: str,
    version: str,
    overall_recall: float,
    results: list[QuestionResult],
) -> ProbeReport:
    """Build a ProbeReport with the given results and computed pass_rate."""
    pass_rate = (
        sum(1 for r in results if r.passed) / len(results) if results else 1.0
    )
    return ProbeReport(
        golden_set_name=name,
        golden_set_version=version,
        timestamp="2025-01-01T00:00:00+00:00",
        overall_recall=overall_recall,
        pass_rate=pass_rate,
        results=results,
    )


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

class TestRegressionDetection:

    def test_recall_drop_appears_in_regressions(self):
        """When current recall < baseline recall, the question is a regression."""
        reporter = DiffReporter()

        baseline_result = _make_question_result(
            question_id="q1",
            question="What is Python?",
            recall_score=1.0,
            found_chunks=["chunk-a", "chunk-b"],
            required_chunks=["chunk-a", "chunk-b"],
        )
        current_result = _make_question_result(
            question_id="q1",
            question="What is Python?",
            recall_score=0.5,
            found_chunks=["chunk-a"],
            required_chunks=["chunk-a", "chunk-b"],
        )

        baseline = _make_report("gs", "1.0", 1.0, [baseline_result])
        current = _make_report("gs", "1.0", 0.5, [current_result])

        diff = reporter.diff(current, baseline)

        assert len(diff.regressions) == 1
        assert diff.regressions[0].question_id == "q1"
        assert diff.regressions[0].baseline_recall == 1.0
        assert diff.regressions[0].current_recall == 0.5
        assert len(diff.improvements) == 0


# ---------------------------------------------------------------------------
# Improvement detection
# ---------------------------------------------------------------------------

class TestImprovementDetection:

    def test_recall_increase_appears_in_improvements(self):
        """When current recall > baseline recall, the question is an improvement."""
        reporter = DiffReporter()

        baseline_result = _make_question_result(
            question_id="q2",
            question="What is Rust?",
            recall_score=0.5,
            found_chunks=["chunk-d"],
            required_chunks=["chunk-d", "chunk-e"],
        )
        current_result = _make_question_result(
            question_id="q2",
            question="What is Rust?",
            recall_score=1.0,
            found_chunks=["chunk-d", "chunk-e"],
            required_chunks=["chunk-d", "chunk-e"],
        )

        baseline = _make_report("gs", "1.0", 0.5, [baseline_result])
        current = _make_report("gs", "1.0", 1.0, [current_result])

        diff = reporter.diff(current, baseline)

        assert len(diff.improvements) == 1
        assert diff.improvements[0].question_id == "q2"
        assert diff.improvements[0].baseline_recall == 0.5
        assert diff.improvements[0].current_recall == 1.0
        assert len(diff.regressions) == 0


# ---------------------------------------------------------------------------
# Unchanged detection
# ---------------------------------------------------------------------------

class TestUnchangedDetection:

    def test_same_recall_appears_in_unchanged(self):
        """When recall is identical, the question appears in the unchanged list."""
        reporter = DiffReporter()

        result = _make_question_result(
            question_id="q3",
            question="What is Go?",
            recall_score=0.8,
            found_chunks=["chunk-f"],
            required_chunks=["chunk-f", "chunk-g"],
        )

        baseline = _make_report("gs", "1.0", 0.8, [result])
        current = _make_report("gs", "1.0", 0.8, [result])

        diff = reporter.diff(current, baseline)

        assert "q3" in diff.unchanged
        assert len(diff.regressions) == 0
        assert len(diff.improvements) == 0

    def test_within_epsilon_is_unchanged(self):
        """Tiny floating-point differences (within epsilon) are treated as unchanged."""
        reporter = DiffReporter()

        baseline_result = _make_question_result(
            question_id="q4",
            question="test",
            recall_score=0.8,
            found_chunks=["c1"],
            required_chunks=["c1", "c2"],
        )
        # Recall differs by 1e-10 — should be unchanged
        current_result = _make_question_result(
            question_id="q4",
            question="test",
            recall_score=0.8 + 1e-10,
            found_chunks=["c1"],
            required_chunks=["c1", "c2"],
        )

        baseline = _make_report("gs", "1.0", 0.8, [baseline_result])
        current = _make_report("gs", "1.0", 0.8 + 1e-10, [current_result])

        diff = reporter.diff(current, baseline)

        assert "q4" in diff.unchanged
        assert len(diff.regressions) == 0
        assert len(diff.improvements) == 0


# ---------------------------------------------------------------------------
# newly_lost_chunks
# ---------------------------------------------------------------------------

class TestNewlyLostChunks:

    def test_newly_lost_chunks_computed_correctly(self):
        """newly_lost_chunks = baseline found chunks ∖ current found chunks."""
        reporter = DiffReporter()

        baseline_result = _make_question_result(
            question_id="q1",
            question="test",
            recall_score=1.0,
            found_chunks=["chunk-a", "chunk-b", "chunk-c"],
            required_chunks=["chunk-a", "chunk-b", "chunk-c"],
        )
        current_result = _make_question_result(
            question_id="q1",
            question="test",
            recall_score=1.0 / 3.0,
            found_chunks=["chunk-a"],
            required_chunks=["chunk-a", "chunk-b", "chunk-c"],
        )

        baseline = _make_report("gs", "1.0", 1.0, [baseline_result])
        current = _make_report("gs", "1.0", 1.0 / 3.0, [current_result])

        diff = reporter.diff(current, baseline)

        assert len(diff.regressions) == 1
        lost = diff.regressions[0].newly_lost_chunks
        assert sorted(lost) == ["chunk-b", "chunk-c"]


# ---------------------------------------------------------------------------
# newly_found_chunks
# ---------------------------------------------------------------------------

class TestNewlyFoundChunks:

    def test_newly_found_chunks_computed_correctly(self):
        """newly_found_chunks = current found chunks ∖ baseline found chunks."""
        reporter = DiffReporter()

        baseline_result = _make_question_result(
            question_id="q2",
            question="test",
            recall_score=0.5,
            found_chunks=["chunk-a"],
            required_chunks=["chunk-a", "chunk-b"],
        )
        current_result = _make_question_result(
            question_id="q2",
            question="test",
            recall_score=1.0,
            found_chunks=["chunk-a", "chunk-b"],
            required_chunks=["chunk-a", "chunk-b"],
        )

        baseline = _make_report("gs", "1.0", 0.5, [baseline_result])
        current = _make_report("gs", "1.0", 1.0, [current_result])

        diff = reporter.diff(current, baseline)

        assert len(diff.improvements) == 1
        found = diff.improvements[0].newly_found_chunks
        assert found == ["chunk-b"]


# ---------------------------------------------------------------------------
# overall_delta
# ---------------------------------------------------------------------------

class TestOverallDelta:

    def test_positive_delta_for_improvement(self):
        reporter = DiffReporter()

        baseline = _make_report(
            "gs", "1.0", 0.5,
            [
                _make_question_result("q1", "test", 0.5, ["c1"], ["c1", "c2"]),
            ],
        )
        current = _make_report(
            "gs", "1.0", 1.0,
            [
                _make_question_result("q1", "test", 1.0, ["c1", "c2"], ["c1", "c2"]),
            ],
        )

        diff = reporter.diff(current, baseline)

        assert diff.overall_delta == pytest.approx(0.5)

    def test_negative_delta_for_regression(self):
        reporter = DiffReporter()

        baseline = _make_report(
            "gs", "1.0", 1.0,
            [
                _make_question_result("q1", "test", 1.0, ["c1", "c2"], ["c1", "c2"]),
            ],
        )
        current = _make_report(
            "gs", "1.0", 0.5,
            [
                _make_question_result("q1", "test", 0.5, ["c1"], ["c1", "c2"]),
            ],
        )

        diff = reporter.diff(current, baseline)

        assert diff.overall_delta == pytest.approx(-0.5)

    def test_zero_delta_for_unchanged(self):
        reporter = DiffReporter()

        result = _make_question_result("q1", "test", 0.8, ["c1"], ["c1", "c2"])
        baseline = _make_report("gs", "1.0", 0.8, [result])
        current = _make_report("gs", "1.0", 0.8, [result])

        diff = reporter.diff(current, baseline)

        assert diff.overall_delta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# format_json
# ---------------------------------------------------------------------------

class TestFormatJson:

    def test_format_json_returns_valid_json(self):
        reporter = DiffReporter()
        diff = RegressionDiff(
            overall_delta=-0.1,
            regressions=[
                ChunkRegression(
                    question_id="q1",
                    question="test question",
                    newly_lost_chunks=["chunk-x"],
                    baseline_recall=1.0,
                    current_recall=0.5,
                )
            ],
            improvements=[],
            unchanged=["q2"],
        )

        output = reporter.format_json(diff)

        # Must be parseable JSON
        parsed = json.loads(output)
        assert parsed["overall_delta"] == -0.1
        assert len(parsed["regressions"]) == 1
        assert parsed["regressions"][0]["question_id"] == "q1"
        assert parsed["regressions"][0]["newly_lost_chunks"] == ["chunk-x"]
        assert parsed["improvements"] == []
        assert parsed["unchanged"] == ["q2"]

    def test_format_json_is_pretty_printed(self):
        """JSON output should be human-readable (indented)."""
        reporter = DiffReporter()
        diff = RegressionDiff()
        output = reporter.format_json(diff)

        # Pretty-printed JSON has newlines
        assert "\n" in output


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------

class TestFormatTable:

    def test_format_table_returns_non_empty_string(self):
        reporter = DiffReporter()
        diff = RegressionDiff(
            overall_delta=0.05,
            regressions=[],
            improvements=[],
            unchanged=["q1", "q2"],
        )

        output = reporter.format_table(diff)

        assert isinstance(output, str)
        assert len(output) > 0

    def test_format_table_contains_regression_info(self):
        reporter = DiffReporter()
        diff = RegressionDiff(
            overall_delta=-0.2,
            regressions=[
                ChunkRegression(
                    question_id="q-reg",
                    question="What went wrong?",
                    newly_lost_chunks=["lost-1"],
                    baseline_recall=1.0,
                    current_recall=0.0,
                ),
            ],
        )

        output = reporter.format_table(diff)

        assert "q-reg" in output
        assert "Regressions" in output

    def test_format_table_contains_improvement_info(self):
        reporter = DiffReporter()
        diff = RegressionDiff(
            overall_delta=0.3,
            improvements=[
                ChunkImprovement(
                    question_id="q-imp",
                    question="What got better?",
                    newly_found_chunks=["new-1"],
                    baseline_recall=0.0,
                    current_recall=1.0,
                ),
            ],
        )

        output = reporter.format_table(diff)

        assert "q-imp" in output
        assert "Improvements" in output


# ---------------------------------------------------------------------------
# format_github
# ---------------------------------------------------------------------------

class TestFormatGithub:

    def test_regression_contains_error_annotation(self):
        """Regressions must use the ::error workflow command."""
        reporter = DiffReporter()
        diff = RegressionDiff(
            overall_delta=-0.1,
            regressions=[
                ChunkRegression(
                    question_id="q-fail",
                    question="Failed question",
                    newly_lost_chunks=["lost-a"],
                    baseline_recall=1.0,
                    current_recall=0.5,
                ),
            ],
        )

        output = reporter.format_github(diff)

        assert "::error" in output
        assert "q-fail" in output
        assert "lost-a" in output

    def test_improvement_contains_notice_annotation(self):
        """Improvements must use the ::notice workflow command."""
        reporter = DiffReporter()
        diff = RegressionDiff(
            overall_delta=0.2,
            improvements=[
                ChunkImprovement(
                    question_id="q-win",
                    question="Improved question",
                    newly_found_chunks=["new-b"],
                    baseline_recall=0.5,
                    current_recall=1.0,
                ),
            ],
        )

        output = reporter.format_github(diff)

        assert "::notice" in output
        assert "q-win" in output

    def test_github_output_contains_summary_group(self):
        """The output must wrap a summary in ::group:: / ::endgroup::."""
        reporter = DiffReporter()
        diff = RegressionDiff(overall_delta=0.0)

        output = reporter.format_github(diff)

        assert "::group::LongProbe Regression Summary" in output
        assert "::endgroup::" in output

    def test_github_output_contains_overall_delta(self):
        reporter = DiffReporter()
        diff = RegressionDiff(overall_delta=-0.1234)

        output = reporter.format_github(diff)

        assert "-0.1234" in output


# ---------------------------------------------------------------------------
# New question not in baseline → skipped
# ---------------------------------------------------------------------------

class TestNewQuestionHandling:

    def test_new_question_not_in_baseline_is_ignored(self):
        """Questions that exist in current but not baseline should not appear
        in regressions, improvements, or unchanged."""
        reporter = DiffReporter()

        current_result = _make_question_result(
            question_id="q-new",
            question="Brand new question",
            recall_score=1.0,
            found_chunks=["c1"],
            required_chunks=["c1"],
        )

        baseline = _make_report("gs", "1.0", 0.5, [])
        current = _make_report("gs", "1.0", 1.0, [current_result])

        diff = reporter.diff(current, baseline)

        assert len(diff.regressions) == 0
        assert len(diff.improvements) == 0
        assert len(diff.unchanged) == 0


# ---------------------------------------------------------------------------
# Mixed scenario
# ---------------------------------------------------------------------------

class TestMixedScenario:

    def test_mixed_regression_improvement_and_unchanged(self):
        """A single golden set can produce all three categories."""
        reporter = DiffReporter()

        q1_baseline = _make_question_result(
            "q1", "regression Q", 1.0,
            found_chunks=["a", "b"], required_chunks=["a", "b"],
        )
        q1_current = _make_question_result(
            "q1", "regression Q", 0.5,
            found_chunks=["a"], required_chunks=["a", "b"],
        )
        q2_baseline = _make_question_result(
            "q2", "improvement Q", 0.5,
            found_chunks=["c"], required_chunks=["c", "d"],
        )
        q2_current = _make_question_result(
            "q2", "improvement Q", 1.0,
            found_chunks=["c", "d"], required_chunks=["c", "d"],
        )
        q3_result = _make_question_result(
            "q3", "unchanged Q", 0.8,
            found_chunks=["e"], required_chunks=["e", "f"],
        )

        baseline = _make_report(
            "gs", "1.0", (1.0 + 0.5 + 0.8) / 3,
            [q1_baseline, q2_baseline, q3_result],
        )
        current = _make_report(
            "gs", "1.0", (0.5 + 1.0 + 0.8) / 3,
            [q1_current, q2_current, q3_result],
        )

        diff = reporter.diff(current, baseline)

        assert len(diff.regressions) == 1
        assert diff.regressions[0].question_id == "q1"
        assert len(diff.improvements) == 1
        assert diff.improvements[0].question_id == "q2"
        assert "q3" in diff.unchanged
