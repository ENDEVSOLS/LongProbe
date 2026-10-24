"""Shared caller behavior, explicit references, and defensive promotion state."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
import yaml

from longprobe import LongProbe
from longprobe.cli.main import _run_probe
from longprobe.config import ConfigurationError, ProbeConfig
from longprobe.core.baseline import BaselineStore
from longprobe.core.evaluation import BaselineNotFoundError, evaluate
from longprobe.core.golden import GoldenQuestion, GoldenSet
from longprobe.core.policy import BaselineQualityError


@pytest.fixture
def setup(tmp_path):
    suite = GoldenSet("policies", "1", [GoldenQuestion("q1", "Q1", ["a", "b"])])
    suite.to_yaml(str(tmp_path / "goldens.yaml"))
    data = {
        "retriever": {"type": "http", "http": {"url": "http://localhost/retrieve"}},
        "baseline": {"db_path": str(tmp_path / "b.db"), "auto_compare": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return suite, path, ProbeConfig.from_dict(data)


def test_cli_and_python_use_equivalent_settings_and_results(tmp_path, monkeypatch, setup):
    suite, path, _ = setup
    adapter = Mock()
    adapter.retrieve.return_value = [{"id": "a"}]
    monkeypatch.setattr("longprobe.cli.main._create_adapter_from_config", lambda cfg: adapter)
    cli = _run_probe(tmp_path / "goldens.yaml", path)[2]
    python = LongProbe(adapter, golden_set=suite, config_path=str(path)).run()
    for report in (cli, python):
        assert report.overall_recall == 0.5
        assert report.policy == {"mode": "all_questions", "recall_threshold": 0.8}
        assert not report.quality["passed"]
    assert cli.results[0].contract == python.results[0].contract
    assert cli.selection == python.selection
    assert cli.baseline_identity == python.baseline_identity
    assert adapter.retrieve.call_count == 2


def test_returned_report_and_config_mutation_cannot_weaken_promotion(setup):
    suite, path, _ = setup
    adapter = Mock()
    adapter.retrieve.return_value = [{"id": "a"}]
    probe = LongProbe(adapter, golden_set=suite, config_path=str(path))
    report = probe.run()
    report.policy["mode"] = "aggregate"
    report.policy["recall_threshold"] = 0
    report.results[0].passed = True
    report.overall_recall = report.pass_rate = 1
    probe.config.baseline.quality_gate = "aggregate"
    probe.config.scoring.recall_threshold = 0
    with pytest.raises(BaselineQualityError):
        probe.save_baseline()
    assert probe.baseline_store.load() is None


def test_passing_snapshot_survives_returned_report_mutation(setup):
    suite, path, _ = setup
    adapter = Mock()
    adapter.retrieve.return_value = [{"id": "a"}, {"id": "b"}]
    probe = LongProbe(adapter, golden_set=suite, config_path=str(path))
    report = probe.run()
    report.results.clear()
    probe.save_baseline("release")
    assert len(probe.baseline_store.load("release").results) == 1
    assert probe.baseline_store.load("latest") is None


@pytest.mark.parametrize("label", ["", " ", None])
def test_unusable_baseline_labels_cannot_be_promoted(setup, label):
    suite, path, _ = setup
    adapter = Mock()
    adapter.retrieve.return_value = [{"id": "a"}, {"id": "b"}]
    probe = LongProbe(adapter, golden_set=suite, config_path=str(path))
    probe.run()
    with pytest.raises(ValueError, match="label"):
        probe.save_baseline(label)
    assert probe.baseline_store.list_labels() == []


def test_missing_explicit_reference_fails_before_retrieval(setup):
    suite, _, config = setup
    retrieve = Mock()
    store = BaselineStore(config.baseline.db_path)
    with pytest.raises(BaselineNotFoundError):
        evaluate(suite, retrieve, config, store=store, baseline_label="missing")
    retrieve.assert_not_called()


def test_optional_reference_absence_is_visible(setup):
    suite, _, config = setup
    config.baseline.auto_compare = True
    report = evaluate(
        suite,
        lambda q, k: [{"id": "a"}, {"id": "b"}],
        config,
        store=BaselineStore(config.baseline.db_path),
    )
    assert report.baseline_identity["status"] == "not_found"
    assert report.baseline_identity["requested_label"] == "latest"
    assert report.quality["passed"]


def test_explicit_reference_does_not_compare_with_latest_first(setup):
    suite, _, config = setup
    reference = evaluate(suite, lambda q, k: [{"id": "a"}, {"id": "b"}], config, compare=False)
    store = Mock()
    store.load.return_value = reference
    report = evaluate(
        suite, lambda q, k: [{"id": "a"}], config, store=store, baseline_label="release"
    )
    store.load.assert_called_once_with("release")
    assert report.regression_detected
    assert report.baseline_identity["resolved_label"] == "release"


def test_filtered_override_is_validated_against_full_suite(setup):
    suite, _, config = setup
    suite.questions.append(GoldenQuestion("q2", "Q2", ["a"], status="draft"))
    config.scoring.question_thresholds = {"q2": 1}
    report = evaluate(suite, lambda q, k: [{"id": "a"}, {"id": "b"}], config, compare=False)
    assert len(report.results) == 1
    config.scoring.question_thresholds = {"unknown": 1}
    retrieve = Mock()
    with pytest.raises(ConfigurationError, match="unknown question"):
        evaluate(suite, retrieve, config, compare=False)
    retrieve.assert_not_called()


def test_rebaseline_does_not_require_legacy_comparison_to_pass(setup):
    suite, path, config = setup
    config.baseline.auto_compare = True
    reference = evaluate(suite, lambda q, k: [{"id": "a"}], config, compare=False)
    legacy = deepcopy(reference)
    legacy.schema_version = None
    legacy.results[0].contract = None
    store = BaselineStore(config.baseline.db_path)
    store.save(legacy)
    adapter = Mock()
    adapter.retrieve.return_value = [{"id": "a"}, {"id": "b"}]
    probe = LongProbe(adapter, golden_set=suite, config_path=str(path))
    probe.config.baseline.auto_compare = True
    assert probe.run().baseline_identity["status"] == "incompatible"
    probe.save_baseline()
    assert store.load().schema_version == 2
