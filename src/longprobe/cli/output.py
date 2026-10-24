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


def report_payload(report: ProbeReport) -> dict:
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


def github_annotations(report: ProbeReport) -> str:
    lines = []
    for result in report.results:
        level = "notice" if result.passed else "error"
        message = f"[{result.question_id}] recall={result.recall_score:.0%}; missing: {', '.join(result.missing_chunks) or 'none'}"
        lines.append(f"::{level} title=LongProbe Retrieval::{escape_annotation(message)}")
    comparison = report.comparison or {}
    if comparison and not comparison.get("compatible", False):
        lines.append(
            "::error title=Incompatible comparison::"
            + escape_annotation("; ".join(comparison.get("compatibility_reasons", [])))
        )
    for change in comparison.get("regressions", []):
        message = f"[{change['question_id']}] Required evidence lost: {', '.join(change['newly_lost_chunks'])}"
        lines.append("::error title=Retrieval Regression::" + escape_annotation(message))
    identity = report.baseline_identity
    message = f"Baseline: {identity.get('resolved_label') or identity.get('requested_label') or 'none'}; comparison: {identity.get('status', 'not_requested')}; timestamp: {identity.get('timestamp') or 'none'}"
    lines.append("::notice title=LongProbe Reference::" + escape_annotation(message))
    return "\n".join(lines)
