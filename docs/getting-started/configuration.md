# Configuration

LongProbe uses two main configuration files:

- `goldens.yaml` - Defines your golden question set
- `longprobe.yaml` - Configures retriever, embedder, and scoring

## Configuration File: longprobe.yaml

### Complete Example

```yaml
retriever:
  type: "chroma"
  chroma:
    persist_directory: "./chroma_db"
    collection: "my_documents"

embedder:
  provider: "local"
  model: "text-embedding-3-small"
  dimensions: 1536

scoring:
  recall_threshold: 0.8
  fail_on_regression: true

baseline:
  db_path: ".longprobe/baselines.db"
  auto_compare: true
```

## Retriever Configuration

### ChromaDB

```yaml
retriever:
  type: "chroma"
  chroma:
    persist_directory: "./chroma_db"
    collection: "my_documents"
```

### Pinecone

```yaml
retriever:
  type: "pinecone"
  pinecone:
    index_name: "my-index"
    api_key: "${PINECONE_API_KEY}"
    namespace: ""
```

### Qdrant

```yaml
retriever:
  type: "qdrant"
  qdrant:
    collection: "my_collection"
    host: "localhost"
    port: 6333
    api_key: "${QDRANT_API_KEY}"
```

### HTTP API

```yaml
retriever:
  type: "http"
  http:
    url: "http://localhost:8000/api/retrieve"
    method: "POST"
    headers:
      Authorization: "Bearer ${API_TOKEN}"
    body_template: '{"query": "{question}", "top_k": {top_k}}'
    response_mapping:
      results_path: "data.chunks"
      text_field: "content"
      id_field: "id"
```

## Embedder Configuration

### OpenAI

```yaml
embedder:
  provider: "openai"
  model: "text-embedding-3-small"
  api_key: "${OPENAI_API_KEY}"
  dimensions: 1536
```

### Hugging Face

```yaml
embedder:
  provider: "huggingface"
  model: "sentence-transformers/all-MiniLM-L6-v2"
```

### Local (No API)

```yaml
embedder:
  provider: "local"
  model: "text-embedding-3-small"
```

## Scoring Configuration

```yaml
scoring:
  recall_threshold: 0.8        # Minimum recall to pass
  fail_on_regression: true     # Exit 1 if regression detected
```

## Baseline Configuration

```yaml
baseline:
  db_path: ".longprobe/baselines.db"  # SQLite database path
  auto_compare: true                   # Auto-compare vs latest baseline
```

## Environment Variables

Use `${VAR_NAME}` syntax to reference environment variables:

```yaml
retriever:
  type: "pinecone"
  pinecone:
    api_key: "${PINECONE_API_KEY}"
```

Set in your environment:

```bash
export PINECONE_API_KEY="your-key-here"
```

## Configuration Reference

| Section | Field | Type | Default | Description |
|---------|-------|------|---------|-------------|
| `retriever` | `type` | string | `"chroma"` | Adapter type |
| `retriever.chroma` | `persist_directory` | string | `""` | ChromaDB path |
| `retriever.chroma` | `collection` | string | `""` | Collection name |
| `retriever.pinecone` | `index_name` | string | `""` | Pinecone index |
| `retriever.pinecone` | `api_key` | string | `""` | API key |
| `retriever.qdrant` | `host` | string | `"localhost"` | Qdrant host |
| `retriever.qdrant` | `port` | int | `6333` | Qdrant port |
| `embedder` | `provider` | string | `"openai"` | Embedding provider |
| `embedder` | `model` | string | `""` | Model name |
| `embedder` | `dimensions` | int | `0` | Embedding dimensions |
| `scoring` | `recall_threshold` | float | `0.8` | Min recall to pass |
| `scoring` | `fail_on_regression` | bool | `true` | Exit 1 on regression |
| `baseline` | `db_path` | string | `.longprobe/baselines.db` | SQLite path |
| `baseline` | `auto_compare` | bool | `true` | Auto-compare vs baseline |

## Next Steps

- [Golden Questions](../guide/golden-questions.md) - Define your test cases
- [CLI Reference](../guide/cli-reference.md) - Learn CLI commands
- [Vector Stores](../integrations/vector-stores.md) - Adapter details
