from __future__ import annotations

import logging
from typing import Any

from .base import AbstractRetrieverAdapter

logger = logging.getLogger(__name__)


class ChromaAdapter(AbstractRetrieverAdapter):
    """Adapter for Chroma vector databases.

    Supports both text-based and embedding-based queries.  The Chroma
    client SDK is imported lazily and cached after the first access.
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str = "",
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily create and cache a Chroma client."""
        if self._client is not None:
            return self._client

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaAdapter. "
                "Install it with:  pip install chromadb"
            )

        if self.persist_directory:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self._client = chromadb.Client()

        return self._client

    def retrieve(
        self,
        query: str | list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query a Chroma collection.

        Args:
            query: Either a text string or a pre-computed embedding vector.
            top_k: Number of results to return.

        Returns:
            List of result dicts normalised to the LongProbe format.
            Scores are converted from Chroma distances (``1 - distance``).
        """
        client = self._get_client()
        collection = client.get_collection(self.collection_name)

        if isinstance(query, str):
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        elif isinstance(query, list):
            results = collection.query(
                query_embeddings=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        else:
            raise TypeError(
                f"Query must be a str or list[float], got {type(query).__name__}"
            )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        mapped: list[dict[str, Any]] = []
        for i in range(len(ids)):
            mapped.append(
                {
                    "id": ids[i],
                    "text": documents[i] if documents and i < len(documents) else "",
                    "score": 1.0 - (distances[i] if distances and i < len(distances) else 0.0),
                    "metadata": (
                        metadatas[i]
                        if metadatas and i < len(metadatas)
                        else {}
                    ),
                }
            )

        return mapped

    def health_check(self) -> bool:
        """Check connectivity by listing collections."""
        try:
            client = self._get_client()
            client.list_collections()
            return True
        except Exception:
            logger.debug("Chroma health check failed", exc_info=True)
            return False
