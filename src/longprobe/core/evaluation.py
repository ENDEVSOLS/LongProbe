"""Shared evaluation module: selection, recorded policy, and comparison.

Callers supply retrieval and baseline storage. No presentation or process exit
belongs here; a quality failure is report data, not an execution exception.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict

from longprobe.config import ConfigurationError, ProbeConfig

from .baseline import BaselineStore
from .diff import DiffReporter
from .golden import GoldenSet
from .scorer import EmptyEvaluationError, ProbeReport, RecallScorer


class BaselineNotFoundError(ValueError):
    """An explicitly requested reference does not exist."""


class EvaluationExecutionError(RuntimeError):
    """Retrieval failed operationally, rather than producing a quality miss."""


def select_questions(
    golden_set: GoldenSet,
    config: ProbeConfig,
    *,
    tags: list[str] | None = None,
    include_drafts: bool = False,
) -> GoldenSet:
    suite = GoldenSet.from_dict(asdict(golden_set))
    unknown = set(config.scoring.question_thresholds) - {q.id for q in suite.questions}
    if unknown:
        raise ConfigurationError("scoring.question_thresholds references unknown question IDs.")
    if tags:
        suite = suite.filter_by_tags(tags)
    if not include_drafts:
        suite = suite.filter_by_status("approved")
    if not suite.questions:
        raise EmptyEvaluationError(
            "No questions remain after tag/status selection; add or approve questions, or select --include-drafts for analysis."
        )
    return suite


def evaluate(
    golden_set: GoldenSet,
    retrieve: Callable[[str, int], list[dict]],
    config: ProbeConfig,
    *,
    store: BaselineStore | None = None,
    tags: list[str] | None = None,
    include_drafts: bool = False,
    top_k_override: int | None = None,
    baseline_label: str | None = None,
    compare: bool = True,
) -> ProbeReport:
    settings = deepcopy(config)
    settings.validate_for("evaluate", owns_retriever=False)
    suite = select_questions(golden_set, settings, tags=tags, include_drafts=include_drafts)
    if top_k_override is not None and (type(top_k_override) is not int or top_k_override <= 0):
        raise ConfigurationError("top_k must be a positive integer.")
    baseline = None
    identity = {
        "status": "not_requested",
        "requested_label": None,
        "resolved_label": None,
        "timestamp": None,
    }
    if compare and (baseline_label is not None or settings.baseline.auto_compare):
        label = baseline_label if baseline_label is not None else "latest"
        if not isinstance(label, str) or not label.strip():
            raise ConfigurationError("Baseline label must be nonempty.")
        if store is None:
            raise ConfigurationError("Baseline storage is required for the selected comparison.")
        baseline = store.load(label)
        identity.update(status="not_found", requested_label=label)
        if baseline is None and baseline_label is not None:
            raise BaselineNotFoundError(
                "Requested baseline was not found; save it explicitly or select an existing label."
            )
        if baseline is not None:
            identity.update(status="compared", resolved_label=label, timestamp=baseline.timestamp)
    selected_ids = {q.id for q in suite.questions}
    scorer = RecallScorer(
        settings.scoring.recall_threshold,
        {k: v for k, v in settings.scoring.question_thresholds.items() if k in selected_ids},
    )

    def retrieve_checked(query, top_k):
        try:
            return retrieve(query, top_k)
        except ImportError:
            raise
        except Exception:
            raise EvaluationExecutionError(
                "Retrieval failed; check the configured endpoint and service diagnostics."
            ) from None

    report = scorer.score_all(suite, retrieve_checked, top_k_override)
    report.policy = {
        "mode": settings.baseline.quality_gate,
        "recall_threshold": settings.scoring.recall_threshold,
    }
    report.selection = {
        "tags": sorted(set(tags or [])),
        "include_drafts": include_drafts,
        "evaluated_question_ids": sorted(selected_ids),
    }
    report.baseline_identity = identity
    if baseline is not None:
        comparison = DiffReporter().diff(report, baseline)
        report.comparison = asdict(comparison)
        report.baseline_recall = baseline.overall_recall
        report.recall_delta = comparison.overall_delta
        report.regression_detected = bool(comparison.regressions)
        if not comparison.compatible:
            report.baseline_identity["status"] = "incompatible"
    failed = sorted(r.question_id for r in report.results if not r.passed)
    reasons = []
    if failed:
        reasons.append("One or more evaluated questions failed.")
    if report.regression_detected:
        reasons.append("Required evidence regressed against the selected baseline.")
    if identity["status"] == "incompatible":
        reasons.append("Baseline comparison is incompatible or unverified.")
    report.quality = {
        "passed": not reasons,
        "mode": "evaluation",
        "failed_questions": failed,
        "reasons": reasons,
    }
    return report
