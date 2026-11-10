"""Acceptance tests for LP-01: enforcing configured top-K in evaluation."""

from longprobe.core.diff import DiffReporter
from longprobe.core.golden import GoldenQuestion, GoldenSet
from longprobe.core.reporting import fingerprint, report_from_dict, report_to_dict
from longprobe.core.scorer import RecallScorer


def test_k1_ignores_evidence_at_rank_2():
    """Case 1: When K=1, evidence present at rank 2 is ignored; recall is zero."""
    question = GoldenQuestion(
        id="q1",
        question="What is the refund policy?",
        required_chunks=["chunk-refund"],
        match_mode="id",
        top_k=1,
    )
    golden_set = GoldenSet("test-suite", "1.0", [question])
    scorer = RecallScorer(recall_threshold=0.8)

    # Retriever returns 3 results, with the expected chunk at rank 2 (index 1)
    retrieved = [
        {"id": "chunk-unrelated", "text": "Unrelated content", "score": 0.9},
        {"id": "chunk-refund", "text": "Refunds within 30 days", "score": 0.8},
        {"id": "chunk-extra", "text": "Extra info", "score": 0.7},
    ]

    report = scorer.score_all(golden_set, lambda q, k: retrieved)
    res = report.results[0]

    assert res.recall_score == 0.0
    assert not res.passed
    assert res.returned_count == 3
    assert res.evaluated_count == 1
    assert res.retrieved_chunk_ids == ["chunk-unrelated"]
    assert res.missing_chunks == ["chunk-refund"]
    assert res.found_chunks == []


def test_k2_finds_evidence_at_rank_2():
    """Case 2: When K=2 on the same results, expected evidence at rank 2 is found."""
    question = GoldenQuestion(
        id="q1",
        question="What is the refund policy?",
        required_chunks=["chunk-refund"],
        match_mode="id",
        top_k=2,
    )
    golden_set = GoldenSet("test-suite", "1.0", [question])
    scorer = RecallScorer(recall_threshold=0.8)

    retrieved = [
        {"id": "chunk-unrelated", "text": "Unrelated content", "score": 0.9},
        {"id": "chunk-refund", "text": "Refunds within 30 days", "score": 0.8},
        {"id": "chunk-extra", "text": "Extra info", "score": 0.7},
    ]

    report = scorer.score_all(golden_set, lambda q, k: retrieved)
    res = report.results[0]

    assert res.recall_score == 1.0
    assert res.passed
    assert res.returned_count == 3
    assert res.evaluated_count == 2
    assert res.retrieved_chunk_ids == ["chunk-unrelated", "chunk-refund"]
    assert res.missing_chunks == []
    assert res.found_chunks == ["chunk-refund"]


def test_top_k_override_fewer_results_and_empty():
    """Case 3: K override, fewer results than requested, and empty results."""
    question = GoldenQuestion(
        id="q1",
        question="What is the refund policy?",
        required_chunks=["chunk-refund"],
        match_mode="id",
        top_k=5,
    )
    golden_set = GoldenSet("test-suite", "1.0", [question])
    scorer = RecallScorer(recall_threshold=0.8)

    retrieved = [
        {"id": "chunk-unrelated", "text": "Unrelated", "score": 0.9},
        {"id": "chunk-refund", "text": "Refunds", "score": 0.8},
    ]

    # Override top_k to 1 on a question with default top_k=5
    report = scorer.score_all(golden_set, lambda q, k: retrieved, top_k_override=1)
    res = report.results[0]
    assert res.evaluated_count == 1
    assert res.recall_score == 0.0

    # Fewer results than requested (K=5, only 2 returned)
    report_few = scorer.score_all(golden_set, lambda q, k: retrieved)
    res_few = report_few.results[0]
    assert res_few.returned_count == 2
    assert res_few.evaluated_count == 2
    assert res_few.recall_score == 1.0

    # Zero results returned
    report_empty = scorer.score_all(golden_set, lambda q, k: [])
    res_empty = report_empty.results[0]
    assert res_empty.returned_count == 0
    assert res_empty.evaluated_count == 0
    assert res_empty.recall_score == 0.0


def test_legacy_baseline_readable_and_strict_compatibility_rejected():
    """Case 5: Legacy v1 baseline can still be read, but strict diff flags it explicitly."""
    question = GoldenQuestion(
        id="q1",
        question="What is the refund policy?",
        required_chunks=["chunk-refund"],
        match_mode="id",
        top_k=2,
    )
    golden_set = GoldenSet("test-suite", "1.0", [question])
    scorer = RecallScorer(recall_threshold=0.8)

    # Current report (v2 matcher)
    current_report = scorer.score_all(
        golden_set,
        lambda q, k: [{"id": "chunk-refund", "text": "Refunds", "score": 1.0}],
    )

    # Simulate a legacy baseline dictionary where matcher_version is v1 and counts are absent
    legacy_dict = report_to_dict(current_report)
    legacy_dict["results"][0]["contract"]["matcher_version"] = "longprobe-lexical-v1"
    # Recompute fingerprint for v1 contract
    legacy_dict["results"][0]["contract"]["fingerprint"] = fingerprint(legacy_dict["results"][0]["contract"])
    legacy_dict["results"][0].pop("returned_count", None)
    legacy_dict["results"][0].pop("evaluated_count", None)

    # Reading the legacy report succeeds without error
    legacy_report = report_from_dict(legacy_dict)
    assert legacy_report.results[0].evaluated_count is None

    # Comparing against the legacy baseline rejects strict compatibility
    diff = DiffReporter().diff(current_report, legacy_report)
    assert not diff.compatible
    assert any("matcher version" in r for r in diff.compatibility_reasons)
