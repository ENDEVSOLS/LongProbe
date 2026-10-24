# LangChain integration

Wrap your application's LangChain retriever with `LangChainRetrieverAdapter`. Install `longprobe[langchain]` and any provider-specific packages your application uses. For the unreleased v0.2.0 candidate, install the checkout as explained in [installation](../getting-started/installation.md).

## Connect an existing retriever

```python
from longprobe import LongProbe
from longprobe.adapters import LangChainRetrieverAdapter

# application_retriever is your already configured LangChain retriever.
probe = LongProbe(
    adapter=LangChainRetrieverAdapter(application_retriever),
    goldens_path="goldens.yaml",
    config_path="longprobe.yaml",
)
report = probe.run()
assert report.quality["passed"], report.quality["reasons"]
```

The application supplies `application_retriever`; LongProbe does not construct your ingestion pipeline or configure its embeddings. Use a configuration file for scoring and baseline policy. A supplied adapter avoids connecting to an unused built-in backend.

The adapter calls `invoke(query)`, with a fallback to `get_relevant_documents(query)`. It maps `page_content` to text and copies metadata. IDs come from `metadata["chunk_id"]`, then `metadata["source"]`, then a positional `doc_N` fallback. Use stable, unique `chunk_id` values for ID matching. A document source shared by multiple chunks is not a unique chunk ID.

`top_k` truncates returned results; it does not increase the upstream retriever's configured result limit. Configure your application's retriever to return enough candidates.

## Baselines and pytest

After reviewing the golden evidence, explicitly save a passing candidate:

```python
probe.save_baseline(label="production-v1")
report = probe.run(baseline_label="production-v1")
assert report.quality["passed"], report.quality["reasons"]
```

A good overall mean alone does not guarantee passing quality. The shared verdict includes individual failures, regressions, and incompatible comparisons. Promotion uses its recorded baseline policy.

For pytest, override the supported `longprobe_adapter` fixture to return `LangChainRetrieverAdapter(application_retriever)`, then request `longprobe_report` or enable automatic enforcement. See [pytest integration](pytest.md) for session reuse and required configuration.

For a custom retrieval chain, subclass `AbstractRetrieverAdapter` and implement `retrieve(query, top_k)` returning dictionaries with `id`, `text`, `score`, and `metadata`. There is no built-in `CustomAdapter` class. See the [Python guide](../guide/python-api.md) and [baseline management](../guide/baseline-management.md).
