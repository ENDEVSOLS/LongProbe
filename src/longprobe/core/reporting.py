"""Versioned report encoding shared by SQLite, CLI output, and report readers.

Unversioned reports remain readable without inventing evaluation settings.
Only reports with verified contracts can establish strict comparability.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .golden import GoldenQuestion
from .scorer import ProbeReport, QuestionResult

SCHEMA_VERSION = 2
MATCHER_VERSION = "longprobe-lexical-v1"


class ReportError(ValueError):
    """Malformed or unsupported report; messages never include raw contents."""


def fingerprint(contract: dict) -> str:
    content = {key: value for key, value in contract.items() if key != "fingerprint"}
    try:
        encoded = json.dumps(content, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, TypeError):
        raise ReportError("Report contract contains invalid values.") from None
    return hashlib.sha256(encoded).hexdigest()


def question_contract(question: GoldenQuestion, top_k: int, threshold: float) -> dict:
    contract = {
        "question": question.question,
        "required_chunks": sorted(question.required_chunks),
        "match_mode": question.match_mode,
        "semantic_threshold": question.semantic_threshold
        if question.match_mode == "semantic"
        else None,
        "top_k": top_k,
        "recall_threshold": threshold,
        "matcher_version": MATCHER_VERSION,
        "status": question.status,
    }
    contract["fingerprint"] = fingerprint(contract)
    return contract


def _number(value, name: str, *, ratio: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReportError(f"Report {name} must be a finite number.")
    if value < 0 or (ratio and value > 1):
        raise ReportError(f"Report {name} is out of range.")
    return float(value)


def _strings(value, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ReportError(f"Report {name} must be a list of strings.")
    return list(value)


def report_from_dict(data: dict) -> ProbeReport:
    """Validate a stored report or CLI success envelope without lossy coercions."""
    try:
        if not isinstance(data, dict):
            raise ReportError("Report must be a JSON object.")
        version = data.get("schema_version")
        if version is not None and (type(version) is not int or version != SCHEMA_VERSION):
            raise ReportError("Unsupported report schema version.")
        raw_results = data["results"]
        if not isinstance(raw_results, list) or not raw_results:
            raise ReportError("Report must contain evaluated questions.")
        results = []
        seen = set()
        for raw in raw_results:
            qid = raw["question_id"]
            if not isinstance(qid, str) or not qid.strip() or qid in seen:
                raise ReportError("Report question IDs must be nonempty and unique.")
            seen.add(qid)
            if not isinstance(raw["question"], str) or type(raw["passed"]) is not bool:
                raise ReportError("Report question text or passing status is invalid.")
            result = QuestionResult(
                question_id=qid,
                question=raw["question"],
                recall_score=_number(raw["recall_score"], "recall_score"),
                retrieved_chunk_ids=_strings(
                    raw.get("retrieved_chunk_ids", []), "retrieved_chunk_ids"
                ),
                required_chunks=_strings(raw["required_chunks"], "required_chunks"),
                missing_chunks=_strings(raw["missing_chunks"], "missing_chunks"),
                found_chunks=_strings(raw["found_chunks"], "found_chunks"),
                passed=raw["passed"],
                latency_ms=_number(raw["latency_ms"], "latency_ms", ratio=False),
                contract=raw.get("contract"),
            )
            if version == SCHEMA_VERSION:
                _validate_contract(result)
            results.append(result)
        name = data.get("golden_set_name", data.get("golden_set"))
        for item in (name, data["golden_set_version"], data["timestamp"]):
            if not isinstance(item, str) or not item.strip():
                raise ReportError("Report suite identity and timestamp must be nonempty strings.")
        report = ProbeReport(
            golden_set_name=name,
            golden_set_version=data["golden_set_version"],
            timestamp=data["timestamp"],
            results=results,
            overall_recall=_number(data["overall_recall"], "overall_recall"),
            pass_rate=_number(data["pass_rate"], "pass_rate"),
            baseline_recall=None
            if data.get("baseline_recall") is None
            else _number(data["baseline_recall"], "baseline_recall"),
            recall_delta=data.get("recall_delta"),
            regression_detected=data.get("regression_detected", False),
            schema_version=version,
        )
        for key in ("selection", "policy", "baseline_identity", "quality"):
            value = data.get(key, {})
            if not isinstance(value, dict):
                raise ReportError(f"Report {key} must be an object.")
            setattr(report, key, value)
        report.comparison = data.get("comparison", data.get("diff"))
        if report.comparison is not None and not isinstance(report.comparison, dict):
            raise ReportError("Report comparison must be an object.")
        if report.quality:
            if type(report.quality.get("passed")) is not bool or report.quality.get("mode") not in {
                "evaluation",
                "all_questions",
                "aggregate",
            }:
                raise ReportError("Report quality decision is invalid.")
            _strings(report.quality.get("failed_questions"), "quality.failed_questions")
            _strings(report.quality.get("reasons"), "quality.reasons")
            if (
                report.quality["passed"]
                and report.quality["mode"] in {"evaluation", "all_questions"}
                and any(not r.passed for r in results)
            ):
                raise ReportError("Report passing verdict contradicts failed questions.")
        if report.comparison is not None:
            comparison = report.comparison
            if type(comparison.get("compatible")) is not bool:
                raise ReportError("Report comparison compatibility is invalid.")
            for key in (
                "compatibility_reasons",
                "added_questions",
                "removed_questions",
                "unchanged",
            ):
                _strings(comparison.get(key), f"comparison.{key}")
            for key in ("regressions", "improvements", "changed_questions", "evidence_changes"):
                if not isinstance(comparison.get(key), list) or any(
                    not isinstance(item, dict) for item in comparison[key]
                ):
                    raise ReportError("Report comparison changes must be lists of objects.")
                for item in comparison[key]:
                    if (
                        not isinstance(item.get("question_id"), str)
                        or not item["question_id"].strip()
                    ):
                        raise ReportError("Report comparison changes require question IDs.")
                    if key in {"regressions", "improvements"}:
                        _number(item.get("baseline_recall"), "comparison.baseline_recall")
                        _number(item.get("current_recall"), "comparison.current_recall")
                    if key in {"regressions", "evidence_changes"}:
                        _strings(item.get("newly_lost_chunks"), "comparison.newly_lost_chunks")
                    if key in {"improvements", "evidence_changes"}:
                        _strings(item.get("newly_found_chunks"), "comparison.newly_found_chunks")
                    if key == "changed_questions":
                        _strings(item.get("fields"), "comparison.fields")
            delta = comparison.get("overall_delta")
            if not comparison["compatible"] and delta is not None:
                raise ReportError(
                    "Incompatible comparisons cannot carry a comparable overall delta."
                )
            if delta is not None and (
                isinstance(delta, bool)
                or not isinstance(delta, (float, int))
                or not math.isfinite(delta)
                or abs(delta) > 1
            ):
                raise ReportError("Report comparison delta is invalid.")
            if (
                report.quality.get("passed")
                and report.quality.get("mode") == "evaluation"
                and (not comparison["compatible"] or comparison["regressions"])
            ):
                raise ReportError("Report passing verdict contradicts comparison failures.")
        if type(report.regression_detected) is not bool:
            raise ReportError("Report regression flag must be a boolean.")
        if report.recall_delta is not None:
            delta = report.recall_delta
            if (
                isinstance(delta, bool)
                or not isinstance(delta, (int, float))
                or not math.isfinite(delta)
                or abs(delta) > 1
            ):
                raise ReportError("Report recall_delta must be a finite difference in [-1, 1].")
        if version == SCHEMA_VERSION:
            if report.policy.get("mode") not in {"all_questions", "aggregate"}:
                raise ReportError("Report baseline quality policy is missing or invalid.")
            _number(report.policy.get("recall_threshold"), "policy.recall_threshold")
            mean = sum(r.recall_score for r in results) / len(results)
            rate = sum(r.passed for r in results) / len(results)
            if abs(mean - report.overall_recall) > 1e-9 or abs(rate - report.pass_rate) > 1e-9:
                raise ReportError("Report aggregates disagree with question results.")
        return report
    except (KeyError, TypeError, AttributeError, OverflowError):
        raise ReportError("Report is missing required fields or contains invalid data.") from None


def _validate_contract(result: QuestionResult) -> None:
    contract = result.contract
    if not isinstance(contract, dict) or contract.get("fingerprint") != fingerprint(contract):
        raise ReportError("Report evaluation contract is missing or has an invalid fingerprint.")
    if contract.get("question") != result.question or contract.get("required_chunks") != sorted(
        result.required_chunks
    ):
        raise ReportError("Report contract disagrees with the evaluated question.")
    if contract.get("match_mode") not in {"id", "text", "semantic"} or not isinstance(
        contract.get("matcher_version"), str
    ):
        raise ReportError("Report matching contract is invalid.")
    if (
        type(contract.get("top_k")) is not int
        or contract["top_k"] <= 0
        or contract.get("status") not in {"draft", "approved"}
    ):
        raise ReportError("Report retrieval limit or question status is invalid.")
    threshold = _number(contract["recall_threshold"], "contract.recall_threshold")
    if contract["match_mode"] == "semantic":
        _number(contract["semantic_threshold"], "contract.semantic_threshold")
    required, found = Counter(result.required_chunks), Counter(result.found_chunks)
    missing = Counter(result.missing_chunks)
    if not required or found - required or required - found != missing:
        raise ReportError("Report required, found, and missing evidence are inconsistent.")
    expected = len(result.found_chunks) / len(result.required_chunks)
    if abs(expected - result.recall_score) > 1e-9 or result.passed != (
        result.recall_score >= threshold
    ):
        raise ReportError(
            "Report score or passing status disagrees with recorded evidence and threshold."
        )


def report_to_dict(report: ProbeReport) -> dict:
    """Validate before writing; retain a single representation across storage routes."""
    data = asdict(report)
    report_from_dict(data)
    return data


def read_report(path: str | Path) -> ProbeReport:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        raise ReportError("Report must contain a valid UTF-8 JSON document.") from None
    return report_from_dict(data)
