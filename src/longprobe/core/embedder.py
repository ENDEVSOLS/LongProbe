"""
Query Embedder module for LongProbe.

Provides a unified interface for generating query embeddings with pluggable
backends (OpenAI, HuggingFace, or a local zero-vector fallback for testing).
All returned vectors are L2-normalized to unit length.
"""

from __future__ import annotations

import math
import warnings
from typing import List


class QueryEmbedder:
    """Generate normalized embedding vectors for query strings.

    Supports three provider backends:

    - ``"openai"``       – OpenAI text-embedding API
    - ``"huggingface"``  – HuggingFace ``sentence-transformers`` models
    - ``"local"``        – Zero-vector fallback (no external dependencies)

    Parameters
    ----------
    provider:
        Embedding backend to use.  One of ``"openai"``, ``"huggingface"``,
        or ``"local"``.
    model:
        Model identifier.  When empty the provider-specific default is used.
    api_key:
        API key required by cloud providers (OpenAI).  Ignored for
        ``"huggingface"`` and ``"local"``.
    dimensions:
        Output embedding dimensions.  Only meaningful for the ``"local"``
        provider (which uses it directly) and OpenAI models that support the
        ``dimensions`` parameter.  ``0`` means "use the model's native size".
    batch_size:
        Maximum number of queries processed in a single API call (reserved
        for future use; current implementation calls ``embed`` one-by-one).
    """

    # Default model identifiers per provider
    _DEFAULT_MODELS: dict[str, str] = {
        "openai": "text-embedding-3-small",
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        "local": "",  # dimensions param controls vector size
    }

    def __init__(
        self,
        provider: str = "local",
        model: str = "",
        api_key: str = "",
        dimensions: int = 0,
        batch_size: int = 32,
    ) -> None:
        if provider not in ("openai", "huggingface", "local"):
            raise ValueError(
                f"Unknown embedding provider '{provider}'. "
                f"Supported providers are: 'openai', 'huggingface', 'local'."
            )

        self.provider: str = provider
        self.model: str = model or self._DEFAULT_MODELS[provider]
        self.api_key: str = api_key
        self.dimensions: int = dimensions
        self.batch_size: int = batch_size

        # Lazy-loaded backend instances
        self._hf_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, query: str) -> List[float]:
        """Embed a single query string and return a normalized vector.

        Parameters
        ----------
        query:
            The text to embed.

        Returns
        -------
        List[float]
            An L2-normalized embedding vector.
        """
        if not query or not query.strip():
            # Return a zero vector for empty queries to avoid API errors
            dim = self.dimensions or self._native_dimensions()
            return [0.0] * dim

        if self.provider == "openai":
            vector = self._embed_openai(query)
        elif self.provider == "huggingface":
            vector = self._embed_huggingface(query)
        else:
            vector = self._embed_local(query)

        return self.normalize(vector)

    def embed_batch(self, queries: List[str]) -> List[List[float]]:
        """Embed multiple query strings.

        Parameters
        ----------
        queries:
            A list of text strings to embed.

        Returns
        -------
        List[List[float]]
            A list of L2-normalized embedding vectors, one per input query.
        """
        return [self.embed(q) for q in queries]

    @staticmethod
    def normalize(vector: List[float]) -> List[float]:
        """L2-normalize *vector* to a unit vector.

        Parameters
        ----------
        vector:
            The raw embedding vector.

        Returns
        -------
        List[float]
            The vector scaled so that its Euclidean norm equals 1.
            If the input is a zero vector it is returned unchanged.
        """
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            return vector[:]
        return [x / norm for x in vector]

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _embed_openai(self, query: str) -> List[float]:
        """Call the OpenAI embeddings API for a single query string."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the OpenAI embedding "
                "provider. Install it with: pip install openai"
            )

        client_kwargs: dict = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key

        client = openai.OpenAI(**client_kwargs)

        create_kwargs: dict = {
            "input": query,
            "model": self.model,
        }
        if self.dimensions > 0:
            create_kwargs["dimensions"] = self.dimensions

        response = client.embeddings.create(**create_kwargs)
        return response.data[0].embedding

    def _embed_huggingface(self, query: str) -> List[float]:
        """Embed using a HuggingFace ``sentence-transformers`` model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "The 'sentence-transformers' package is required for the "
                "HuggingFace embedding provider. Install it with: "
                "pip install sentence-transformers"
            )

        # Cache the model instance so we only load it once.
        if self._hf_model is None:
            self._hf_model = SentenceTransformer(self.model)

        embedding = self._hf_model.encode(query)
        return embedding.tolist()

    def _embed_local(self, query: str) -> List[float]:
        """Return a zero vector (fallback for testing without API keys)."""
        dim = self.dimensions or 384
        warnings.warn(
            "Local embedding mode is active — returning a zero vector of "
            f"dimension {dim}. This is intended for testing only and will "
            "NOT produce meaningful similarity scores.",
            UserWarning,
            stacklevel=3,
        )
        return [0.0] * dim

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _native_dimensions(self) -> int:
        """Return the default embedding dimension for the active provider."""
        if self.provider == "local":
            return self.dimensions or 384
        if self.provider == "openai":
            # text-embedding-3-small -> 1536, text-embedding-3-large -> 3072
            known: dict[str, int] = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            return known.get(self.model, 1536)
        if self.provider == "huggingface":
            # all-MiniLM-L6-v2 -> 384
            known: dict[str, int] = {
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "sentence-transformers/all-mpnet-base-v2": 768,
            }
            return known.get(self.model, 384)
        return 384
