from abc import ABC, abstractmethod
from typing import Any


class AbstractRetrieverAdapter(ABC):
    """Base class for all retriever adapters."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve documents for a query.

        Returns list of dicts with keys:
        - "id": str - document/chunk identifier
        - "text": str - document text content
        - "score": float - relevance score
        - "metadata": dict - additional metadata
        """
        ...

    def health_check(self) -> bool:
        """Check if the retriever is reachable. Override for custom health checks."""
        return True
