# Quick Start

Get up and running with LongProbe in 5 minutes.

## 1. Install LongProbe

```bash
pip install longprobe
```

## 2. Initialize Your Project

```bash
longprobe init
```

This creates three files:

- `.longprobe/` - Directory for baseline storage
- `goldens.yaml` - Your golden question set
- `longprobe.yaml` - Configuration file

## 3. Define Golden Questions

Edit `goldens.yaml` to define your test cases:

```yaml
name: "my-rag-golden-set"
version: "1.0"

questions:
  - id: "q1"
    question: "What is the refund policy?"
    match_mode: "text"
    required_chunks:
      - "30-day money-back guarantee"
      - "full refund within 30 days"
    top_k: 5
    tags: ["policy", "critical"]

  - id: "q2"
    question: "How do I reset my password?"
    match_mode: "text"
    required_chunks:
      - "click forgot password"
      - "check your email"
    top_k: 5
```

## 4. Configure Your Vector Store

Edit `longprobe.yaml`:

```yaml
retriever:
  type: "chroma"
  chroma:
    persist_directory: "./chroma_db"
    collection: "my_documents"

embedder:
  provider: "local"
  model: "text-embedding-3-small"

scoring:
  recall_threshold: 0.8
  fail_on_regression: true
```

## 5. Run Your First Check

```bash
longprobe check
```

You'll see output like:

```
LongProbe Results
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ ID                   ┃ Question            ┃   Recall ┃ Required ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ q1                   │ What is the refund  │     100% │    2     │   ✓    │
│                      │ policy?             │          │          │        │
├──────────────────────┼─────────────────────┼──────────┼──────────┼────────┤
│ q2                   │ How do I reset my   │     100% │    2     │   ✓    │
│                      │ password?           │          │          │        │
└──────────────────────┴─────────────────────┴──────────┴──────────┴────────┘
╭─────────────────────────────────────────────────────────── Summary ───────╮
│ Overall Recall:  1.00                                                     │
│ Pass Rate:       1.00  (2/2)                                              │
╰───────────────────────────────────────────────────────────────────────────╯
```

## 6. Save a Baseline

```bash
longprobe baseline save --label v1.0
```

## 7. Make Changes and Compare

After making changes to your RAG system:

```bash
longprobe diff --baseline v1.0
```

This shows you exactly what changed:

```
LongProbe Results
╭─────────────────────────────────────────────────────────── Summary ───────╮
│ Overall Recall:  0.95                                                     │
│ Pass Rate:       0.95  (19/20)                                            │
│ Regressions:     1 question(s) degraded                                   │
│ vs Baseline:     -0.05                                                    │
╰───────────────────────────────────────────────────────────────────────────╯

Regressions detected:
  q5: Recall dropped from 100% to 50%
    Lost chunks: ["specific technical detail"]
```

## Common Workflows

### Development Loop

```bash
# Watch mode - auto re-run on changes
longprobe watch
```

### CI/CD Integration

```bash
# Fail pipeline on regression
longprobe check --output github
```

### Python API

```python
from longprobe import LongProbe
from longprobe.adapters import create_adapter

adapter = create_adapter("chroma", collection_name="docs", persist_directory="./db")
probe = LongProbe(adapter=adapter, goldens_path="goldens.yaml")

report = probe.run()
assert report.overall_recall >= 0.85
```

## Next Steps

- [Configuration Guide](configuration.md) - Detailed configuration options
- [Golden Questions](../guide/golden-questions.md) - Learn how to write effective golden questions
- [Match Modes](../guide/match-modes.md) - Understand different matching strategies
- [CLI Reference](../guide/cli-reference.md) - Complete CLI documentation
