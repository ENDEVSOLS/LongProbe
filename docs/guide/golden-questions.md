# Golden Questions

Golden questions are the foundation of LongProbe testing. They define what your RAG system should retrieve.

## What is a Golden Question?

A golden question consists of:

- **Question**: The query to test
- **Required Chunks**: Document chunks that must be retrieved
- **Match Mode**: How to verify chunks (ID, text, or semantic)
- **Top-K**: Number of results to retrieve
- **Tags**: Optional metadata for organization

## Basic Example

```yaml
questions:
  - id: "q1"
    question: "What is the refund policy?"
    match_mode: "text"
    required_chunks:
      - "30-day money-back guarantee"
      - "full refund within 30 days"
    top_k: 5
    tags: ["policy", "critical"]
```

## Match Modes

See [Match Modes](match-modes.md) for detailed explanations.

## Best Practices

1. **Start Small** - Begin with 5-10 critical questions
2. **Be Specific** - Test specific facts, not general topics
3. **Use Tags** - Organize by importance or category
4. **Test Edge Cases** - Include difficult queries
5. **Update Regularly** - Add questions when bugs are found

## Next Steps

- [Match Modes](match-modes.md) - Understand matching strategies
- [CLI Reference](cli-reference.md) - Run your tests
