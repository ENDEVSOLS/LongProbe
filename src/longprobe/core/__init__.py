"""
longprobe.core — regression-testing primitives for RAG pipelines.

This package exposes the public data models used throughout LongProbe:

* **GoldenQuestion** / **GoldenSet** — ground-truth question sets loaded from
  YAML.
* **QuestionResult** / **ProbeReport** — evaluation results produced by the
  scorer.
* **QueryEmbedder** — interface for computing query embeddings at scoring
  time.
"""

from longprobe.core.embedder import QueryEmbedder
from longprobe.core.golden import GoldenQuestion, GoldenSet
from longprobe.core.scorer import ProbeReport, QuestionResult

__all__ = [
    "GoldenQuestion",
    "GoldenSet",
    "ProbeReport",
    "QueryEmbedder",
    "QuestionResult",
]
