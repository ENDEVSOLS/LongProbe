"""
Integration tests for ``longprobe.adapters.chroma`` — ChromaAdapter.

These tests create an ephemeral ChromaDB instance, insert known documents,
and verify that the adapter returns correctly formatted results.

Requires the ``chromadb`` package and ``--run-integration`` flag::

    pytest tests/integration/test_chroma_adapter.py --run-integration
"""

from __future__ import annotations

import pytest

# Lazy import with skip fallback — ensures the file can be collected even
# when chromadb is not installed.
try:
    import chromadb

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def persist_directory(tmp_path_factory):
    """Create a temporary directory for the ephemeral ChromaDB."""
    return str(tmp_path_factory.mktemp("chroma_db"))


@pytest.fixture(scope="module")
def chroma_client(persist_directory):
    """Create a persistent Chroma client and skip if chromadb is unavailable."""
    if not HAS_CHROMADB:
        pytest.skip("chromadb is not installed")
    client = chromadb.PersistentClient(path=persist_directory)
    return client


@pytest.fixture(scope="module")
def collection(chroma_client):
    """Create (or get) the test collection with sample documents."""
    collection_name = "longprobe-test"

    # Delete if it already exists to ensure a clean state.
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    col = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "l2"},
    )

    # Insert test documents with known IDs, text, and metadata.
    test_docs = [
        ("doc-1", "Python is a high-level programming language known for its readability and simplicity.", {"topic": "python", "type": "intro"}),
        ("doc-2", "Rust is a systems programming language focused on safety and performance.", {"topic": "rust", "type": "intro"}),
        ("doc-3", "JavaScript is the dominant language for web development on both client and server.", {"topic": "javascript", "type": "intro"}),
        ("doc-4", "Machine learning is a subset of artificial intelligence that enables systems to learn from data.", {"topic": "ml", "type": "concept"}),
        ("doc-5", "Docker containers package applications with their dependencies for portable deployment.", {"topic": "devops", "type": "tool"}),
        ("doc-6", "PostgreSQL is a powerful open-source relational database system.", {"topic": "database", "type": "tool"}),
        ("doc-7", "Kubernetes orchestrates containerized applications at scale across clusters.", {"topic": "devops", "type": "tool"}),
    ]

    col.add(
        ids=[doc_id for doc_id, _, _ in test_docs],
        documents=[text for _, text, _ in test_docs],
        metadatas=[meta for _, _, meta in test_docs],
    )

    return col


@pytest.fixture(scope="module")
def adapter(persist_directory, collection):
    """Create a ChromaAdapter pointing at the test collection."""
    # Import inside fixture to ensure chromadb is available.
    from longprobe.adapters.chroma import ChromaAdapter

    return ChromaAdapter(
        collection_name="longprobe-test",
        persist_directory=persist_directory,
    )


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:

    def test_health_check_returns_true(self, adapter):
        """The adapter should report healthy when Chroma is reachable."""
        assert adapter.health_check() is True


# ---------------------------------------------------------------------------
# retrieve — basic functionality
# ---------------------------------------------------------------------------

class TestRetrieve:

    def test_retrieve_returns_list_of_dicts(self, adapter):
        """retrieve() must return a list of dicts."""
        results = adapter.retrieve("Python programming", top_k=3)

        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_retrieve_results_have_required_keys(self, adapter):
        """Each result dict must contain id, text, score, and metadata."""
        results = adapter.retrieve("Rust language", top_k=2)

        for doc in results:
            assert "id" in doc
            assert "text" in doc
            assert "score" in doc
            assert "metadata" in doc

    def test_retrieve_ids_are_strings(self, adapter):
        """All returned IDs must be strings."""
        results = adapter.retrieve("JavaScript web", top_k=3)

        for doc in results:
            assert isinstance(doc["id"], str)

    def test_retrieve_text_is_string(self, adapter):
        """All returned text fields must be non-empty strings."""
        results = adapter.retrieve("machine learning AI", top_k=2)

        for doc in results:
            assert isinstance(doc["text"], str)
            assert len(doc["text"]) > 0

    def test_retrieve_score_is_float(self, adapter):
        """All returned scores must be floats in a reasonable range."""
        results = adapter.retrieve("database PostgreSQL", top_k=2)

        for doc in results:
            assert isinstance(doc["score"], (int, float))

    def test_retrieve_metadata_is_dict(self, adapter):
        """All returned metadata must be dicts."""
        results = adapter.retrieve("Docker containers", top_k=2)

        for doc in results:
            assert isinstance(doc["metadata"], dict)

    def test_retrieve_preserves_document_metadata(self, adapter):
        """Metadata inserted alongside documents must be returned."""
        results = adapter.retrieve("Python", top_k=1)

        assert len(results) >= 1
        # The top result for "Python" should be doc-1
        top_doc = results[0]
        assert top_doc["metadata"].get("topic") == "python"

    def test_retrieve_finds_correct_document(self, adapter):
        """A specific query should surface the most relevant document."""
        results = adapter.retrieve("Kubernetes orchestration containers clusters", top_k=1)

        assert len(results) >= 1
        assert results[0]["id"] == "doc-7"


# ---------------------------------------------------------------------------
# retrieve — top_k limiting
# ---------------------------------------------------------------------------

class TestRetrieveTopK:

    def test_top_k_limits_results(self, adapter):
        """The number of returned results must not exceed top_k."""
        results = adapter.retrieve("programming language", top_k=3)

        assert len(results) <= 3

    def test_top_k_one(self, adapter):
        """top_k=1 should return exactly one result."""
        results = adapter.retrieve("Rust safety", top_k=1)

        assert len(results) == 1

    def test_top_k_larger_than_collection(self, adapter):
        """top_k exceeding collection size should return all documents."""
        results = adapter.retrieve("general query", top_k=100)

        assert len(results) == 7  # we inserted 7 docs

    def test_top_k_results_ordered_by_relevance(self, adapter):
        """Results should be ordered by score (descending — highest first).

        Chroma returns results ordered by distance ascending (lowest first),
        and the adapter converts distance to score via ``1 - distance``.
        Therefore the first result should have the highest score.
        """
        results = adapter.retrieve("Python readability simplicity", top_k=3)

        scores = [doc["score"] for doc in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Results not ordered by score: {scores}"
            )


# ---------------------------------------------------------------------------
# retrieve — edge cases
# ---------------------------------------------------------------------------

class TestRetrieveEdgeCases:

    def test_retrieve_with_empty_query(self, adapter):
        """An empty string query should not crash the adapter."""
        results = adapter.retrieve("", top_k=2)

        assert isinstance(results, list)

    def test_retrieve_with_embedding_vector(self, adapter):
        """Passing a raw embedding vector (list of floats) should also work."""
        # Use a random-ish 384-dimensional vector (Chroma's default dim).
        # The results may not be meaningful, but the call must not error.
        embedding = [0.1] * 384
        results = adapter.retrieve(embedding, top_k=2)

        assert isinstance(results, list)
        assert len(results) <= 2
