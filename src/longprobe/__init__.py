"""
LongProbe — RAG retrieval regression testing harness.

Define Golden Questions once. Run ``longprobe check`` on every commit.
Get an exact diff of which document chunks were lost — before your users notice.
"""

from copy import deepcopy
from importlib.metadata import version

from longprobe.adapters import (
    ChromaAdapter,
    HttpAdapter,
    LangChainRetrieverAdapter,
    LlamaIndexRetrieverAdapter,
    PineconeAdapter,
    QdrantAdapter,
    create_adapter,
)
from longprobe.adapters.base import AbstractRetrieverAdapter
from longprobe.config import ProbeConfig
from longprobe.core.baseline import BaselineStore
from longprobe.core.diff import DiffReporter, RegressionDiff
from longprobe.core.embedder import QueryEmbedder
from longprobe.core.evaluation import evaluate
from longprobe.core.golden import GoldenQuestion, GoldenSet
from longprobe.core.policy import require_baseline_quality
from longprobe.core.scorer import ProbeReport, QuestionResult, RecallScorer

__version__ = version("longprobe")

__all__ = [
    "AbstractRetrieverAdapter",
    "BaselineStore",
    "ChromaAdapter",
    "DiffReporter",
    "GoldenQuestion",
    "GoldenSet",
    "HttpAdapter",
    "LangChainRetrieverAdapter",
    "LlamaIndexRetrieverAdapter",
    "PineconeAdapter",
    "ProbeConfig",
    "ProbeReport",
    "QdrantAdapter",
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
        include_drafts: bool = False,
        golden_set: "GoldenSet | None" = None,
    ):
        from pathlib import Path

        self.adapter = adapter
        if golden_set is not None:
            raw_set = golden_set
        else:
            raw_set = GoldenSet.from_yaml(goldens_path)
        self.golden_set = raw_set
        self.include_drafts = include_drafts

        config_file = Path(config_path)
        try:
            self.config = ProbeConfig.from_yaml(str(config_file))
        except FileNotFoundError:
            if config_file != Path("longprobe.yaml") or config_file.is_symlink():
                raise
            self.config = ProbeConfig.defaults()

        if recall_threshold is not None:
            self.config.scoring.recall_threshold = recall_threshold
        self.config.validate_for("evaluate", owns_retriever=False)

        self.scorer = RecallScorer(recall_threshold=self.config.scoring.recall_threshold)
        self.baseline_store = BaselineStore(db_path=self.config.baseline.db_path)
        self.diff_reporter = DiffReporter()
        self._last_report: ProbeReport | None = None

    def run(
        self,
        top_k_override: int | None = None,
        *,
        baseline_label: str | None = None,
        tags: list[str] | None = None,
    ) -> ProbeReport:
        """Evaluate with shared rules; retain a defensive baseline candidate."""
        self._last_report = None
        report = evaluate(
            self.golden_set,
            self.adapter.retrieve,
            self.config,
            store=self.baseline_store,
            tags=tags,
            include_drafts=self.include_drafts,
            top_k_override=top_k_override,
            baseline_label=baseline_label,
        )
        self._last_report = deepcopy(report)
        return report

    def save_baseline(self, label: str = "latest") -> None:
        """Explicitly promote the latest completed evaluation under its policy."""
        if self._last_report is None:
            raise RuntimeError("No report to save. Run probe.run() first.")
        candidate = deepcopy(self._last_report)
        require_baseline_quality(candidate)
        self.baseline_store.save(candidate, label=label)

    def diff(self, baseline_label: str = "latest") -> dict:
        """Compare the retained run against an explicit saved reference."""
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
            r.question_id: r.missing_chunks for r in self._last_report.results if r.missing_chunks
        }
