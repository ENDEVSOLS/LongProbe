from __future__ import annotations

import logging
from typing import Any

from .base import AbstractRetrieverAdapter

logger = logging.getLogger(__name__)


class LlamaIndexRetrieverAdapter(AbstractRetrieverAdapter):
    """Adapter for LlamaIndex retriever objects.

    Accepts any LlamaIndex retriever and normalises its
    ``NodeWithScore`` results into the LongProbe format.
    """

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve documents using a LlamaIndex retriever."""
        try:
            from llama_index.core import QueryBundle  # noqa: F401
        except ImportError:
            raise ImportError(
                "llama-index-core is required for LlamaIndexRetrieverAdapter. "
                "Install it with:  pip install llama-index-core"
            )

        nodes_with_scores = self.retriever.retrieve(query)

        results: list[dict[str, Any]] = []
        for node in nodes_with_scores:
            results.append(
                {
                    "id": node.node.node_id,
                    "text": node.node.get_content(),
                    "score": float(node.score or 0.0),
                    "metadata": dict(node.node.metadata),
                }
            )

        return results[:top_k]

    def health_check(self) -> bool:
        """Check that the underlying retriever has a ``retrieve`` method."""
        return hasattr(self.retriever, "retrieve")
