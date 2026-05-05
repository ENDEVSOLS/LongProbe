from __future__ import annotations

import logging
from typing import Any

from .base import AbstractRetrieverAdapter

logger = logging.getLogger(__name__)


class LangChainRetrieverAdapter(AbstractRetrieverAdapter):
    """Adapter for LangChain BaseRetriever objects.

    Accepts any LangChain retriever (duck-typed) and normalises its
    output into the LongProbe result format.
    """

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve documents using a LangChain retriever.

        Tries the modern ``invoke`` API first, then falls back to the
        legacy ``get_relevant_documents`` method.
        """
        try:
            from langchain_core.documents import Document  # noqa: F401 – ensure importable
        except ImportError:
            raise ImportError(
                "langchain-core is required for LangChainRetrieverAdapter. "
                "Install it with:  pip install langchain-core"
            )

        # Invoke the retriever
        documents = []
        try:
            documents = self.retriever.invoke(query)
        except AttributeError:
            # Legacy LangChain API
            documents = self.retriever.get_relevant_documents(query)

        results: list[dict[str, Any]] = []
        for i, d in enumerate(documents):
            doc_id = d.metadata.get("chunk_id", d.metadata.get("source", f"doc_{i}"))
            results.append(
                {
                    "id": doc_id,
                    "text": d.page_content,
                    "score": float(d.metadata.get("score", 0.0)),
                    "metadata": dict(d.metadata),
                }
            )

        return results[:top_k]

    def health_check(self) -> bool:
        """Check that the underlying retriever is callable."""
        try:
            return (
                hasattr(self.retriever, "invoke")
                or hasattr(self.retriever, "get_relevant_documents")
            )
        except Exception:
            return False
