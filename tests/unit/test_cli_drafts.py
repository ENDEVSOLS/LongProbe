"""Tests for CLI --include-drafts flag and draft notices."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from longprobe.cli.main import check, diff


def _setup_draft_test_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    goldens_path = tmp_path / "goldens.yaml"
    config_path = tmp_path / "longprobe.yaml"
    db_path = tmp_path / "baselines.db"

    goldens_data = {
        "name": "test-drafts",
        "version": "1.0",
        "questions": [
            {"id": "q1", "question": "Q1", "required_chunks": ["chunk-1"], "status": "approved"},
            {"id": "q2", "question": "Q2", "required_chunks": ["chunk-2"], "status": "draft"},
        ],
    }
    goldens_path.write_text(yaml.dump(goldens_data))

    config_data = {
        "retriever": {"type": "http", "http": {"url": "http://localhost:8000/retrieve"}},
        "scoring": {"recall_threshold": 0.50, "fail_on_regression": False},
        "baseline": {"db_path": str(db_path), "auto_compare": False},
    }
    config_path.write_text(yaml.dump(config_data))

    return goldens_path, config_path, db_path


def test_check_excludes_drafts_by_default(tmp_path: Path, monkeypatch, capsys):
    goldens_path, config_path, _ = _setup_draft_test_files(tmp_path)

    mock_adapter = MagicMock()
    mock_adapter.retrieve.return_value = [{"id": "chunk-1"}]
    monkeypatch.setattr("longprobe.cli.main._create_adapter_from_config", lambda cfg: mock_adapter)

    check(
        goldens=goldens_path,
        config=config_path,
        output="json",
        top_k=None,
        threshold=None,
        tag=[],
        include_drafts=False,
    )

    captured = capsys.readouterr()
    assert (
        "Notice: 1 draft question(s) excluded" in captured.out
        or "Notice: 1 draft question(s) excluded" in captured.err
    )


def test_check_includes_drafts_with_flag(tmp_path: Path, monkeypatch):
    goldens_path, config_path, _ = _setup_draft_test_files(tmp_path)

    calls = []

    def mock_retrieve(q, top_k=5):
        calls.append(q)
        return [{"id": "chunk-1"}]

    mock_adapter = MagicMock()
    mock_adapter.retrieve.side_effect = mock_retrieve
    monkeypatch.setattr("longprobe.cli.main._create_adapter_from_config", lambda cfg: mock_adapter)

    check(
        goldens=goldens_path,
        config=config_path,
        output="table",
        top_k=None,
        threshold=None,
        tag=[],
        include_drafts=True,
    )

    assert len(calls) == 2


def test_diff_live_run_supports_include_drafts(tmp_path: Path, monkeypatch):
    goldens_path, config_path, db_path = _setup_draft_test_files(tmp_path)

    # Save a dummy baseline first
    from longprobe.core.baseline import BaselineStore
    from longprobe.core.scorer import ProbeReport, QuestionResult

    store = BaselineStore(db_path=str(db_path))
    dummy_report = ProbeReport(
        golden_set_name="test-drafts",
        golden_set_version="1.0",
        timestamp="2025-01-01T00:00:00+00:00",
        overall_recall=1.0,
        pass_rate=1.0,
        results=[
            QuestionResult(
                question_id="q1",
                question="Q1",
                recall_score=1.0,
                retrieved_chunk_ids=["chunk-1"],
                required_chunks=["chunk-1"],
                missing_chunks=[],
                found_chunks=["chunk-1"],
                passed=True,
                latency_ms=10.0,
            )
        ],
    )
    store.save(dummy_report, label="latest")

    calls = []

    def mock_retrieve(q, top_k=5):
        calls.append(q)
        return [{"id": "chunk-1"}]

    mock_adapter = MagicMock()
    mock_adapter.retrieve.side_effect = mock_retrieve
    monkeypatch.setattr("longprobe.cli.main._create_adapter_from_config", lambda cfg: mock_adapter)

    with pytest.raises(SystemExit) as exc:
        diff(
            baseline_a=None,
            baseline_b=None,
            baseline_label="latest",
            goldens=goldens_path,
            config=config_path,
            output="table",
            top_k=None,
            threshold=None,
            tag=[],
            include_drafts=True,
        )
    assert exc.value.code == 2  # Legacy reference and added draft are incompatible.

    assert len(calls) == 2
