"""One quality decision for baseline promotion through every supported caller."""

from dataclasses import asdict, dataclass, field

from .reporting import ReportError, report_to_dict
from .scorer import ProbeReport


@dataclass
class GateDecision:
    passed: bool
    mode: str
    failed_questions: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


class BaselineQualityError(ValueError):
    """A completed evaluation is not eligible for baseline promotion."""

    def __init__(self, decision: GateDecision):
        self.decision = decision
        super().__init__("Baseline was not saved: " + "; ".join(decision.reasons))


def assess_baseline(report: ProbeReport) -> GateDecision:
    """Assess recorded policy and evidence, never a mutable aggregate alone."""
    report_to_dict(report)
    if report.schema_version != 2:
        raise ReportError(
            "Legacy reports lack verified settings; run a new evaluation before promotion."
        )
    policy = report.policy
    mode = policy.get("mode")
    threshold = policy.get("recall_threshold")
    if (
        mode not in {"all_questions", "aggregate"}
        or isinstance(threshold, bool)
        or not isinstance(threshold, (float, int))
        or not 0 <= threshold <= 1
    ):
        raise ReportError("Report baseline quality policy is invalid.")
    failed = sorted(
        r.question_id for r in report.results if r.recall_score < r.contract["recall_threshold"]
    )
    drafts = [r.question_id for r in report.results if r.contract["status"] != "approved"]
    reasons = []
    if drafts:
        reasons.append("Draft questions require review and approval before baseline promotion")
    if mode == "all_questions" and failed:
        reasons.append("Not all evaluated questions pass: " + ", ".join(failed))
    if (
        mode == "aggregate"
        and sum(r.recall_score for r in report.results) / len(report.results) < threshold
    ):
        reasons.append("Overall recall is below the configured aggregate threshold")
    return GateDecision(not reasons, mode, failed, reasons)


def require_baseline_quality(report: ProbeReport) -> None:
    decision = assess_baseline(report)
    report.quality = asdict(decision)
    if not decision.passed:
        raise BaselineQualityError(decision)
