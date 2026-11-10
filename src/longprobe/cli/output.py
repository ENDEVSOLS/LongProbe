"""Machine output and annotations derived from one validated evaluation."""

import json
import sys

from longprobe.config import ConfigurationError
from longprobe.core.reporting import report_to_dict
from longprobe.core.scorer import ProbeReport


def validate_output(output: str) -> None:
    if output not in {"table", "json", "github"}:
        raise ConfigurationError("Output must be table, json, or github.")


def write_json(payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.write(encoded + "\n")


def redact_report(report: ProbeReport) -> dict:
    """Produce a safe, redacted dictionary of the evaluation report for shared CI surfaces.

    Excludes sensitive question text, chunk contents, and raw evidence strings,
    retaining only safe identifiers, recall scores, counts, verdicts, and diff metrics.
    """
    payload = {
        "schema_version": report.schema_version,
        "golden_set_name": report.golden_set_name,
        "golden_set_version": report.golden_set_version,
        "timestamp": report.timestamp,
        "overall_recall": report.overall_recall,
        "pass_rate": report.pass_rate,
        "baseline_recall": report.baseline_recall,
        "recall_delta": report.recall_delta,
        "regression_detected": report.regression_detected,
        "baseline_identity": report.baseline_identity,
        "policy": report.policy,
        "quality": report.quality,
        "redacted": True,
        "total_questions": len(report.results),
        "passed_questions": sum(r.passed for r in report.results),
        "failed_questions": sum(not r.passed for r in report.results),
        "status": "passed" if (report.quality and report.quality.get("passed")) else "failed",
        "results": [
            {
                "question_id": r.question_id,
                "passed": r.passed,
                "recall_score": r.recall_score,
                "required_count": len(r.required_chunks),
                "found_count": len(r.found_chunks),
                "missing_count": len(r.missing_chunks),
                "returned_count": r.returned_count,
                "evaluated_count": r.evaluated_count,
                "latency_ms": r.latency_ms,
            }
            for r in report.results
        ],
    }
    if report.comparison is not None:
        comp = dict(report.comparison)
        if "regressions" in comp:
            comp["regressions"] = [
                {
                    "question_id": item["question_id"],
                    "baseline_recall": item.get("baseline_recall"),
                    "current_recall": item.get("current_recall"),
                    "lost_count": len(item.get("newly_lost_chunks", [])),
                }
                for item in comp["regressions"]
            ]
        if "improvements" in comp:
            comp["improvements"] = [
                {
                    "question_id": item["question_id"],
                    "baseline_recall": item.get("baseline_recall"),
                    "current_recall": item.get("current_recall"),
                    "found_count": len(item.get("newly_found_chunks", [])),
                }
                for item in comp["improvements"]
            ]
        if "evidence_changes" in comp:
            comp["evidence_changes"] = [
                {
                    "question_id": item["question_id"],
                    "lost_count": len(item.get("newly_lost_chunks", [])),
                    "found_count": len(item.get("newly_found_chunks", [])),
                }
                for item in comp["evidence_changes"]
            ]
        payload["comparison"] = comp
        payload["diff"] = comp
    return payload


def report_payload(report: ProbeReport, *, redact: bool = False) -> dict:
    if redact:
        return redact_report(report)
    payload = report_to_dict(report)
    payload.update(
        golden_set=report.golden_set_name,
        status=("passed" if report.quality.get("passed") else "failed")
        if report.quality
        else "legacy",
        total_questions=len(report.results),
        passed_questions=sum(r.passed for r in report.results),
        failed_questions=sum(not r.passed for r in report.results),
    )
    if report.comparison is not None:
        payload["diff"] = report.comparison
    return payload


def report_exit_code(report: ProbeReport, *, blocking: bool = True) -> int:
    if report.comparison is not None and not report.comparison.get("compatible", False):
        return 2
    return 1 if blocking and not report.quality.get("passed", False) else 0


def escape_annotation(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def github_annotations(report: ProbeReport, *, redact: bool = False) -> str:
    lines = []
    for result in report.results:
        level = "notice" if result.passed else "error"
        if redact:
            message = (
                f"[{result.question_id}] recall={result.recall_score:.0%}; "
                f"missing {len(result.missing_chunks)} of {len(result.required_chunks)} chunk(s)"
            )
        else:
            message = (
                f"[{result.question_id}] recall={result.recall_score:.0%}; "
                f"missing: {', '.join(result.missing_chunks) or 'none'}"
            )
        lines.append(f"::{level} title=LongProbe Retrieval::{escape_annotation(message)}")
    comparison = report.comparison or {}
    if comparison and not comparison.get("compatible", False):
        lines.append(
            "::error title=Incompatible comparison::"
            + escape_annotation("; ".join(comparison.get("compatibility_reasons", [])))
        )
    for change in comparison.get("regressions", []):
        if redact:
            message = (
                f"[{change['question_id']}] Required evidence lost: "
                f"{len(change.get('newly_lost_chunks', []))} chunk(s)"
            )
        else:
            message = f"[{change['question_id']}] Required evidence lost: {', '.join(change['newly_lost_chunks'])}"
        lines.append("::error title=Retrieval Regression::" + escape_annotation(message))
    identity = report.baseline_identity
    message = f"Baseline: {identity.get('resolved_label') or identity.get('requested_label') or 'none'}; comparison: {identity.get('status', 'not_requested')}; timestamp: {identity.get('timestamp') or 'none'}"
    lines.append("::notice title=LongProbe Reference::" + escape_annotation(message))
    return "\n".join(lines)
