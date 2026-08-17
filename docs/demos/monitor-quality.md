# Monitor RAG Quality Demo

The **Monitor RAG Quality** demonstration illustrates how to continuously measure RAG retrieval performance against saved Golden Questions using LongProbe.

---

## Overview

In production RAG systems, monitoring retrieval quality is critical to ensure that changes to documents, embeddings, or vector databases do not silently degrade recall. 

This demo connects LongProbe to a vector store (ChromaDB), evaluates a set of approved golden questions, and renders an interactive Rich TUI table displaying recall scores, pass/fail status, and retrieved chunk counts.

---

## Running the Demo

To run the RAG quality monitoring demo locally:

```bash
uv run python demo_run/monitor_rag_quality.py
```

---

## Interactive TUI Output

When executed, LongProbe performs three live setup and evaluation phases:

1. **Setup**: Connects to the vector store and seeds/loads sample documents.
2. **Loading**: Loads approved golden questions from `goldens.yaml`.
3. **Testing**: Queries the vector store for each question and calculates recall.

### Sample Output Display:

```text
🔬 RAG Quality Check with LongProbe

╭──────────────────────────────────────── RAG Quality Check ─────────────────────────────────────────╮
│   Setup                                      ✓                    Connected to vector store        │
│   Loading                                    ✓                    Loaded 3 golden questions        │
│   Testing                                    ✓                    All 3 tests complete             │
│                                                                                                    │
│ ✓ Tests complete!                                                                                  │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────────────────── 📊 Test Results ──────────────────────────────────────────╮
│  Question                                                                      Recall        Status  │
│  What are the enterprise payment terms?                                         50.0%        ✗ FAIL  │
│  What is the refund policy?                                                    100.0%        ✓ PASS  │
│  How is data encrypted?                                                        100.0%        ✓ PASS  │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭───────────────────────────────────────────── 📈 Summary ───────────────────────────────────────────╮
│ Overall Recall: 83.3%                                                                              │
│ Pass Rate: 66.7%                                                                                   │
│                                                                                                    │
│ ✓ Passed: 2/3                                                                                      │
│ ✗ Failed: 1/3                                                                                      │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯

⚠️  1 test(s) failed - review missing chunks
```

---

## How It Works Under the Hood

```python
import os
from longprobe import LongProbe
from longprobe.adapters import create_adapter

# 1. Connect to vector database adapter
adapter = create_adapter(
    "chroma",
    collection_name="demo_collection",
    persist_directory="./.demo_chroma_db"
)

# 2. Initialize LongProbe with Golden Questions
probe = LongProbe(
    adapter=adapter,
    goldens_path="goldens.yaml"
)

# 3. Execute quality check
report = probe.run()

# 4. Inspect results
print(f"Overall Recall: {report.overall_recall:.1%}")
for result in report.results:
    print(f"Question: {result.question_id} -> Recall: {result.recall_score:.1%}")
```

---

## Quality Threshold Enforcement

You can enforce hard quality gates in Python or CLI:

```bash
uv run longprobe check --goldens goldens.yaml --threshold 0.85
```

If overall recall falls below `85%`, LongProbe outputs a quality warning diagnostic panel and exits with status code `1`, preventing deployment of broken RAG pipelines.
