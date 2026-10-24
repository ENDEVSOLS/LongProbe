"""Promotion and comparison decisions exercised through shared interfaces."""

import copy
from dataclasses import asdict

import pytest

from longprobe.config import ConfigurationError, ProbeConfig
from longprobe.core.baseline import BaselineStore
from longprobe.core.diff import DiffReporter
from longprobe.core.golden import GoldenQuestion, GoldenSet
from longprobe.core.policy import BaselineQualityError, assess_baseline, require_baseline_quality
from longprobe.core.scorer import RecallScorer


def make_report(ids=("q1",), found=("a",), *, top_k=5, required=("a", "b"), status="approved"):
    suite = GoldenSet(
        "suite", "1", [GoldenQuestion(qid, qid, list(required), status=status) for qid in ids]
    )
    return RecallScorer().score_all(suite, lambda query, k: [{"id": c} for c in found], top_k)


def test_default_gate_cannot_hide_failure_with_high_average():
    suite = GoldenSet(
        "suite",
        "1",
        [
            GoldenQuestion("q1", "q1", ["a"]),
            GoldenQuestion("q2", "q2", ["a", "b", "c", "d", "e"]),
        ],
    )
    report = RecallScorer().score_all(suite, lambda q, k: [{"id": c} for c in "abc"])
    assert report.overall_recall == 0.8
    with pytest.raises(BaselineQualityError, match="q2"):
        require_baseline_quality(report)
    report.policy["mode"] = "aggregate"
    assert assess_baseline(report).passed
    assert assess_baseline(report).failed_questions == ["q2"]


def test_drafts_never_promoted_even_with_aggregate_mode():
    report = make_report(found=("a", "b"), status="draft")
    report.policy["mode"] = "aggregate"
    assert not assess_baseline(report).passed


@pytest.mark.parametrize(
    "config",
    [
        {"scoring": {"question_thresholds": {"q": True}}},
        {"scoring": {"question_thresholds": {"q": float("nan")}}},
        {"scoring": {"question_thresholds": {"q": 1.1}}},
        {"scoring": {"question_thresholds": []}},
        {"baseline": {"quality_gate": "anything"}},
    ],
)
def test_invalid_policy_settings(config):
    with pytest.raises(ConfigurationError):
        ProbeConfig.from_dict(config)


def test_per_question_threshold_is_recorded_and_used():
    suite = GoldenSet("suite", "1", [GoldenQuestion("q1", "q1", ["a", "b"])])
    report = RecallScorer(0.4, {"q1": 1}).score_all(suite, lambda q, k: [{"id": "a"}])
    assert not report.results[0].passed
    assert report.results[0].contract["recall_threshold"] == 1
    assert not assess_baseline(report).passed


def test_unknown_threshold_id_is_not_silently_ignored():
    with pytest.raises(ValueError, match="unknown question"):
        RecallScorer(0.8, {"typo": 1}).score_all(
            GoldenSet("s", "1", [GoldenQuestion("q", "Q", ["a"])]), lambda q, k: []
        )


def test_equal_recall_evidence_swap_is_regression_and_shows_gains():
    baseline = make_report(found=("a",))
    current = make_report(found=("b",))
    diff = DiffReporter().diff(current, baseline)
    assert diff.compatible
    assert diff.overall_delta == 0
    assert len(diff.regressions) == 1
    assert diff.regressions[0].newly_lost_chunks == ["a"]
    assert diff.evidence_changes[0]["newly_found_chunks"] == ["b"]
    assert not diff.improvements


@pytest.mark.parametrize(
    "current,field",
    [
        (lambda: make_report(ids=("q1", "q2")), "added_questions"),
        (lambda: make_report(ids=("q2",)), "removed_questions"),
        (lambda: make_report(top_k=10), "changed_questions"),
        (lambda: make_report(required=("a", "c")), "changed_questions"),
    ],
)
def test_coverage_and_contract_changes_prevent_misleading_delta(current, field):
    diff = DiffReporter().diff(current(), make_report())
    assert not diff.compatible
    assert getattr(diff, field)
    assert diff.overall_delta is None


def test_legacy_diagnostics_do_not_certify_comparability():
    baseline = make_report()
    baseline.schema_version = None
    baseline.results[0].contract = None
    diff = DiffReporter().diff(make_report(), baseline)
    assert not diff.compatible
    assert diff.overall_delta is None
    assert "Legacy" in diff.compatibility_reasons[0]


def test_store_and_reporter_use_the_same_comparison(tmp_path):
    current, baseline = make_report(found=("b",)), make_report()
    assert BaselineStore(str(tmp_path / "b.db")).diff(current, baseline) == asdict(
        DiffReporter().diff(current, baseline)
    )


def test_suite_version_and_reordering_are_handled_explicitly():
    baseline = make_report(ids=("q1", "q2"))
    current = copy.deepcopy(baseline)
    current.results.reverse()
    assert DiffReporter().diff(current, baseline).compatible
    current.golden_set_version = "2"
    assert not DiffReporter().diff(current, baseline).compatible
