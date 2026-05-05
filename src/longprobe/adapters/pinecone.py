from __future__ import annotations

import logging
from typing import Any

from .base import AbstractRetrieverAdapter

logger = logging.getLogger(__name__)


class PineconeAdapter(AbstractRetrieverAdapter):
    """Adapter for Pinecone vector indexes.

    Supports querying by embedding vector directly.  The Pinecone SDK
    is imported lazily so that the adapter can be instantiated without
    the library being present.
    """

    def __init__(
        self,
        index_name: str,
        api_key: str = "",
        namespace: str = "",
        top_k: int = 10,
    ) -> None:
        self.index_name = index_name
        self.api_key = api_key
        self.namespace = namespace
        self.top_k = top_k

    def retrieve(
        self,
        query_embedding: list[float] | None = None,
        query: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query the Pinecone index for the most similar vectors.

        Args:
            query_embedding: Pre-computed query vector to search with.
            query: Text query (reserved for future text-based query support).
            top_k: Number of results to return.

        Returns:
            List of result dicts normalised to the LongProbe format.
        """
        if query_embedding is None:
            if query is None:
                raise ValueError(
                    "Either 'query_embedding' or 'query' must be provided."
                )
            # Future: embed query text.  For now, require an embedding.
            raise NotImplementedError(
                "Text-based queries are not yet supported. "
                "Please provide a 'query_embedding'."
            )

        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "pinecone-client is required for PineconeAdapter. "
                "Install it with:  pip install pinecone-client"
            )

        pc = Pinecone(api_key=self.api_key)
        index = pc.Index(self.index_name)

        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace or None,
            include_metadata=True,
        )

        results: list[dict[str, Any]] = []
        for match in response.matches:
            results.append(
                {
                    "id": match.id,
                    "text": match.metadata.get("text", "") if match.metadata else "",
                    "score": float(match.score or 0.0),
                    "metadata": dict(match.metadata) if match.metadata else {},
                }
            )

        return results

    def health_check(self) -> bool:
        """Check connectivity by describing the configured index."""
        try:
            from pinecone import Pinecone
        except ImportError:
            return False

        try:
            pc = Pinecone(api_key=self.api_key)
            pc.Index(self.index_name).describe_index_stats()
            return True
        except Exception:
            logger.debug("Pinecone health check failed", exc_info=True)
            return False
