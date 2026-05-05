"""
LongProbe — RAG retrieval regression testing harness.

Define Golden Questions once. Run ``longprobe check`` on every commit.
Get an exact diff of which document chunks were lost — before your users notice.
"""

from longprobe.adapters import create_adapter
from longprobe.adapters.base import AbstractRetrieverAdapter
from longprobe.config import ProbeConfig
from longprobe.core.baseline import BaselineStore
from longprobe.core.diff import DiffReporter, RegressionDiff
from longprobe.core.embedder import QueryEmbedder
from longprobe.core.golden import GoldenQuestion, GoldenSet
from longprobe.core.scorer import ProbeReport, QuestionResult, RecallScorer

__version__ = "0.1.0"

__all__ = [
    "AbstractRetrieverAdapter",
    "BaselineStore",
    "DiffReporter",
    "GoldenQuestion",
    "GoldenSet",
    "ProbeConfig",
    "ProbeReport",
    "QueryEmbedder",
    "QuestionResult",
    "RecallScorer",
    "RegressionDiff",
    "create_adapter",
]


class LongProbe:
    """
    High-level facade for running RAG regression probes.

    Usage::

        from longprobe import LongProbe, ChromaAdapter

        adapter = ChromaAdapter(collection_name="my_docs", persist_directory="./db")
        probe = LongProbe(adapter=adapter, goldens_path="goldens.yaml")
        report = probe.run()
        print(f"Overall recall: {report.overall_recall:.2f}")
    """

    def __init__(
        self,
        adapter: "AbstractRetrieverAdapter",
        goldens_path: str = "goldens.yaml",
        config_path: str = "longprobe.yaml",
        recall_threshold: float | None = None,
    ):
        from pathlib import Path

        self.adapter = adapter
        self.golden_set = GoldenSet.from_yaml(goldens_path)

        config_file = Path(config_path)
        if config_file.exists():
            self.config = ProbeConfig.from_yaml(str(config_file))
        else:
            self.config = ProbeConfig.defaults()

        if recall_threshold is not None:
            self.config.scoring.recall_threshold = recall_threshold

        self.scorer = RecallScorer(
            recall_threshold=self.config.scoring.recall_threshold
        )
        self.baseline_store = BaselineStore(
            db_path=self.config.baseline.db_path
        )
        self.diff_reporter = DiffReporter()
        self._last_report: ProbeReport | None = None

    def run(self, top_k_override: int | None = None) -> ProbeReport:
        """Run the probe against the golden set and return a report."""
        report = self.scorer.score_all(
            self.golden_set,
            self.adapter.retrieve,
            top_k_override=top_k_override,
        )

        # Auto-compare against baseline
        if self.config.baseline.auto_compare:
            baseline = self.baseline_store.load("latest")
            if baseline is not None:
                report.baseline_recall = baseline.overall_recall
                report.recall_delta = report.overall_recall - baseline.overall_recall
                diff_result = self.baseline_store.diff(report, baseline)
                report.regression_detected = len(diff_result["regressions"]) > 0

        self._last_report = report
        return report

    def save_baseline(self, label: str = "latest") -> None:
        """Save the last run report as a baseline."""
        if self._last_report is None:
            raise RuntimeError("No report to save. Run probe.run() first.")
        self.baseline_store.save(self._last_report, label=label)

    def diff(self, baseline_label: str = "latest") -> dict:
        """Compare last run against a saved baseline."""
        if self._last_report is None:
            raise RuntimeError("No report to diff. Run probe.run() first.")
        baseline = self.baseline_store.load(baseline_label)
        if baseline is None:
            raise ValueError(f"Baseline '{baseline_label}' not found.")
        return self.baseline_store.diff(self._last_report, baseline)

    def get_missing_chunks(self) -> dict[str, list[str]]:
        """Return a dict mapping question_id to its missing chunks."""
        if self._last_report is None:
            raise RuntimeError("No report available. Run probe.run() first.")
        return {
            r.question_id: r.missing_chunks
            for r in self._last_report.results
            if r.missing_chunks
        }
