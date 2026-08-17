# LangChain Integration Guide

LongProbe provides native, first-class integration with **LangChain** vector stores and retrievers. This guide explains how to connect LongProbe to your LangChain RAG pipelines to measure retrieval recall, detect missing document chunks, and prevent regressions in CI/CD.

---

## 1. Installation

To use LongProbe with LangChain, install the `[langchain]` extra package:

```bash
pip install "longprobe[langchain]"
```

Or using `uv`:

```bash
uv add "longprobe[langchain]"
```

---

## 2. Integration Patterns

### Pattern A: Native Retriever Adapter (Recommended)

LongProbe automatically adapts any LangChain `BaseRetriever` or `VectorStore` (such as `Chroma`, `FAISS`, `PGVector`, `Qdrant`, or `Pinecone`).

When you wrap a LangChain retriever, LongProbe automatically extracts `Document.page_content` and `Document.metadata` to evaluate recall:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from longprobe import LongProbe
from longprobe.adapters import LangChainAdapter

# 1. Initialize your existing LangChain vector store
vectorstore = Chroma(
    collection_name="corporate_docs",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
)

# 2. Create a LangChain retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Wrap with LongProbe LangChainAdapter
adapter = LangChainAdapter(retriever)

# 4. Run retrieval evaluation against Golden Questions
probe = LongProbe(
    adapter=adapter,
    goldens_path="goldens.yaml"
)

report = probe.run()
print(f"Overall Recall: {report.overall_recall:.1%}")
print(f"Pass Rate: {report.pass_rate:.1%}")
```

#### How Chunk Extraction Works:
When LongProbe queries a LangChain retriever:
1. `retriever.invoke(query)` (or `get_relevant_documents`) is called.
2. LongProbe extracts:
   * **Text**: `doc.page_content` $\rightarrow$ Mapped to `text` for ground-truth matching.
   * **ID**: `doc.metadata.get("id")` or `doc.metadata.get("chunk_id")` $\rightarrow$ Mapped to `id` for exact ID matching.
   * **Metadata**: Full `doc.metadata` dictionary.
3. LongProbe compares retrieved text/IDs against required ground-truth chunks using `match_mode: text`, `match_mode: id`, or `match_mode: semantic`.

---

### Pattern B: Custom LCEL Chain or Callable Wrapping

If your LangChain pipeline uses custom LCEL (LangChain Expression Language) runnables, RAG fusion, or multi-step chains, wrap the retrieval step in a lightweight Python callable:

```python
from longprobe import LongProbe
from longprobe.adapters import CustomAdapter

# Define a function that takes a query string and returns a list of chunk dicts
def langchain_rag_chain_retriever(query: str, top_k: int = 5) -> list[dict]:
    # Invoke your custom LangChain retriever or chain
    docs = retriever.invoke(query)
    
    return [
        {
            "id": doc.metadata.get("chunk_id", f"doc_{i}"),
            "text": doc.page_content,
            "metadata": doc.metadata,
        }
        for i, doc in enumerate(docs[:top_k])
    ]

# Wrap in CustomAdapter
adapter = CustomAdapter(retrieval_fn=langchain_rag_chain_retriever)

probe = LongProbe(adapter=adapter, goldens_path="goldens.yaml")
report = probe.run()
```

---

## 3. End-to-End Walkthrough

Here is a complete, runnable example using LangChain document chunking and vector storage:

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from longprobe import LongProbe
from longprobe.adapters import LangChainAdapter

# Step 1: Ingest & Chunk Documents with LangChain
loader = TextLoader("refund_policy.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Step 2: Index Chunks into VectorStore
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    collection_name="policy_docs"
)

# Step 3: Run LongProbe Quality Check
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
adapter = LangChainAdapter(retriever)

probe = LongProbe(adapter=adapter, goldens_path="goldens.yaml")
report = probe.run()

# Step 4: Check Results & Baseline
if report.overall_recall < 0.85:
    print("⚠️ Quality Gate Warning: Retrieval recall below 85%!")
else:
    print("✅ Quality Gate Passed!")
```

---

## 4. Continuous Integration & Pytest

Add LangChain regression checks directly to your test suite (`pytest`):

```python
import pytest
from longprobe import LongProbe
from longprobe.adapters import LangChainAdapter

def test_langchain_retrieval_quality(my_langchain_retriever):
    adapter = LangChainAdapter(my_langchain_retriever)
    probe = LongProbe(adapter=adapter, goldens_path="goldens.yaml")
    
    report = probe.run()
    
    # Assert recall threshold in CI/CD
    assert report.overall_recall >= 0.85, f"Retrieval recall dropped to {report.overall_recall:.1%}"
```

Run in terminal:
```bash
uv run pytest tests/test_rag_pipeline.py
```

---

## 5. Best Practices for LangChain Pipelines

1. **Include Chunk IDs in Metadata**: When using LangChain text splitters, add a unique ID to `doc.metadata["id"]` or `doc.metadata["chunk_id"]`. This allows `match_mode: id` evaluation in LongProbe.
2. **Tune `search_kwargs={"k": ...}`**: Ensure `k` matches the `top_k` specified in your `goldens.yaml` questions.
3. **Test Text Splitter Changes**: Whenever you adjust `chunk_size` or `chunk_overlap` in LangChain, run `longprobe check` to ensure key information isn't split across chunk boundaries.
