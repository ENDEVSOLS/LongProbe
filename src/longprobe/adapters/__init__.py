from .base import AbstractRetrieverAdapter
from .chroma import ChromaAdapter
from .http import HttpAdapter
from .langchain import LangChainRetrieverAdapter
from .llamaindex import LlamaIndexRetrieverAdapter
from .pinecone import PineconeAdapter
from .qdrant import QdrantAdapter


def create_adapter(adapter_type: str, **kwargs) -> AbstractRetrieverAdapter:
    """
    Factory function to create a retriever adapter.

    Args:
        adapter_type: One of "langchain", "llamaindex", "pinecone", "qdrant",
            "chroma", "http"
        **kwargs: Passed to the adapter constructor.
                  For "langchain" and "llamaindex": pass "retriever" kwarg
                  For "http": pass "config" kwarg (HttpRetrieverConfig)
                  For others: pass connection config kwargs

    Returns:
        AbstractRetrieverAdapter instance

    Raises:
        ValueError: If adapter_type is unknown
    """
    adapters = {
        "langchain": LangChainRetrieverAdapter,
        "llamaindex": LlamaIndexRetrieverAdapter,
        "pinecone": PineconeAdapter,
        "qdrant": QdrantAdapter,
        "chroma": ChromaAdapter,
        "http": HttpAdapter,
    }

    adapter_cls = adapters.get(adapter_type)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown adapter type '{adapter_type}'. "
            f"Must be one of: {', '.join(adapters.keys())}"
        )

    return adapter_cls(**kwargs)


__all__ = [
    "AbstractRetrieverAdapter",
    "ChromaAdapter",
    "HttpAdapter",
    "LangChainRetrieverAdapter",
    "LlamaIndexRetrieverAdapter",
    "PineconeAdapter",
    "QdrantAdapter",
    "create_adapter",
]
