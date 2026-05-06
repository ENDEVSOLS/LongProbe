# Match Modes

LongProbe supports three match modes for verifying retrieved chunks.

## ID Match

Exact string match on chunk/document IDs.

```yaml
match_mode: "id"
required_chunks:
  - "doc_a_chunk_3"
  - "doc_b_chunk_7"
```

## Text Match

Case-insensitive substring matching.

```yaml
match_mode: "text"
required_chunks:
  - "30-day money-back guarantee"
```

## Semantic Match

Word-frequency cosine similarity.

```yaml
match_mode: "semantic"
semantic_threshold: 0.80
required_chunks:
  - "Only the CFO and CEO may authorize payments"
```

## Choosing a Match Mode

- **ID**: When you control chunk IDs
- **Text**: For exact phrases
- **Semantic**: When wording may vary
