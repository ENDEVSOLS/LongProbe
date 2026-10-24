"""Contracts and legacy preservation at the report storage interface."""

import copy
import json
import sqlite3
from dataclasses import asdict

import pytest

from longprobe.core.baseline import BaselineStore
from longprobe.core.golden import GoldenQuestion, GoldenSet
from longprobe.core.reporting import (
    ReportError,
    fingerprint,
    question_contract,
    read_report,
    report_from_dict,
    report_to_dict,
)
from longprobe.core.scorer import RecallScorer


@pytest.fixture
def report():
    suite = GoldenSet("policies", "1", [GoldenQuestion("refund", "Refund?", ["a", "b"])])
    return RecallScorer().score_all(suite, lambda query, top_k: [{"id": "a"}])


def test_json_and_sqlite_preserve_contracts(tmp_path, report):
    report.selection = {"tags": ["critical"], "include_drafts": False}
    report.baseline_identity = {"status": "not_found", "requested_label": "latest"}
    data = report_to_dict(report)
    assert report_from_dict(data) == report
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data))
    assert read_report(path) == report
    store = BaselineStore(str(tmp_path / "baselines.db"))
    store.save(report)
    assert store.load() == report


def test_legacy_row_is_read_without_rewriting(tmp_path, report):
    data = asdict(report)
    for key in (
        "schema_version",
        "selection",
        "policy",
        "baseline_identity",
        "comparison",
        "quality",
    ):
        data.pop(key)
    for result in data["results"]:
        result.pop("contract")
    encoded = json.dumps(data)
    store = BaselineStore(str(tmp_path / "old.db"))
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "INSERT INTO baselines (label, golden_set_name, golden_set_version, timestamp, report_json) VALUES (?, ?, ?, ?, ?)",
            ("old", "policies", "1", report.timestamp, encoded),
        )
    loaded = store.load("old")
    assert loaded.schema_version is None
    assert loaded.results[0].contract is None
    assert store.list_labels()[0]["label"] == "old"
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT report_json FROM baselines").fetchone()[0] == encoded


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version=99),
        lambda d: d.update(schema_version=True),
        lambda d: d.update(results=[]),
        lambda d: d.update(overall_recall=float("nan")),
        lambda d: d.update(pass_rate=True),
        lambda d: d.update(overall_recall=1),
        lambda d: d["results"].append(copy.deepcopy(d["results"][0])),
        lambda d: d["results"][0].update(passed="false"),
        lambda d: d["results"][0].update(found_chunks=["unknown"]),
        lambda d: d["results"][0].update(contract=None),
        lambda d: d["results"][0]["contract"].update(top_k=20),
        lambda d: d.update(policy={}),
        lambda d: d.update(quality={"passed": "false", "mode": "evaluation"}),
        lambda d: d.update(
            quality={"passed": True, "mode": "evaluation", "failed_questions": [], "reasons": []}
        ),
        lambda d: d.update(comparison={"compatible": "true"}),
    ],
)
def test_rejects_malformed_or_unsupported_reports(report, change):
    data = asdict(report)
    change(data)
    with pytest.raises(ReportError):
        report_from_dict(data)


def test_contract_order_is_stable_and_settings_are_significant():
    a = GoldenQuestion("q", "Question", ["b", "a"])
    b = GoldenQuestion("q", "Question", ["a", "b"])
    assert question_contract(a, 5, 0.8) == question_contract(b, 5, 0.8)
    contract = question_contract(a, 5, 0.8)
    assert contract["fingerprint"] != question_contract(a, 10, 0.8)["fingerprint"]
    assert contract["fingerprint"] != question_contract(a, 5, 0.9)["fingerprint"]
    assert fingerprint(contract) == contract["fingerprint"]


def test_failed_write_does_not_replace_baseline(tmp_path, report):
    store = BaselineStore(str(tmp_path / "baselines.db"))
    store.save(report)
    broken = copy.deepcopy(report)
    broken.results[0].recall_score = 1
    with pytest.raises(ReportError):
        store.save(broken)
    assert store.load() == report
