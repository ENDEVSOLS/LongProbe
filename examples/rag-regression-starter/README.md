# LongProbe RAG Regression Starter Kit

A self-contained, reproducible starter kit for **LongProbe 0.2.0** demonstrating sub-second RAG retrieval regression testing.

Runs locally **without API keys, credentials, or external vector databases**. Uses 20 reviewed golden questions and a 3-state local HTTP retrieval fixture to guide you through the full **Pass → Fail → Fix** workflow.

---

## Source Documents & Reuse Rights

The documents in `documents/` represent synthetic enterprise SaaS policies:
- `01_refund_and_billing.md`
- `02_security_and_privacy.md`
- `03_enterprise_sla.md`
- `04_account_governance.md`

> **Note on Data & Reuse:** All documents and questions are 100% synthetic scenarios created by EnDevSols released under the **MIT License**. No customer data, PII, API tokens, or proprietary embeddings are used.

---

## Quick Setup (Python 3.11+)

```bash
# 1. Create and activate a fresh virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install LongProbe 0.2.0
pip install "longprobe==0.2.0"

# 3. Enter the starter directory
cd examples/rag-regression-starter
```

---

## Automated 1-Command Demo

Run the entire 3-stage workflow with a single command:

```bash
python run_demo.py
```

This automated runner spins up the local fixture, runs all 3 stages, validates exit codes (`0` → `1` → `0`), and saves clean terminal logs to `sample_outputs/`.

---

## Manual Step-by-Step Workflow

If you prefer to run the commands manually in your own terminal sessions:

### Step 1: Start the Local Fixture
In a separate terminal, start the fixture in known-good mode:
```bash
python fixture_server.py --port 8000 --mode good
```

### Step 2: Run the 3 Workflow Stages

| Stage | Command pattern | Expected result |
|---|---|---|
| **1. Known good** | `longprobe check`<br>`longprobe baseline save --label launch-demo` | `20/20` questions pass (`100.0%` recall). Baseline `launch-demo` is saved to SQLite. Exit code `0`. |
| **2. Degraded** | Switch fixture mode:<br>`python fixture_server.py --mode degraded`<br>Then run:<br>`longprobe check --baseline launch-demo --output github`<br>`longprobe diff --baseline launch-demo` | Exits with **code `1`**. Recall drops to `85.0%` (`17/20`). GitHub Actions annotations and terminal diff pinpoint the 3 missing chunks (`q01`, `q02`, `q11`). |
| **3. Corrected** | Switch fixture mode:<br>`python fixture_server.py --mode corrected`<br>Then re-run:<br>`longprobe check --baseline launch-demo`<br>`longprobe diff --baseline launch-demo` | Exits with **code `0`**. `20/20` questions pass (`100.0%` recall). Diff reports 0 regressions against `launch-demo`. |

---

## Configuration Settings

The starter uses `longprobe.yaml`:
```yaml
retriever:
  type: http
  http:
    url: "http://127.0.0.1:8000/retrieve"
    method: "POST"
    timeout: 10
    response_mapping:
      results_path: "data.chunks"
      id_field: "chunk_id"
      text_field: "content"
      score_field: "similarity"

scoring:
  recall_threshold: 0.90
  fail_on_regression: true

baseline:
  db_path: ".longprobe/baselines.db"
  auto_compare: true
  quality_gate: all_questions
```

- **`recall_threshold: 0.90`**: Requires at least 90% overall recall for evaluation to pass.
- **`fail_on_regression: true`**: Causes `longprobe check` to exit nonzero whenever any question degrades compared to the selected baseline.
- **`quality_gate: all_questions`**: Baseline promotion requires every evaluated question to pass.

---

## Measured Timings & Performance

Benchmarks recorded on standard developer hardware (Ubuntu Linux, Python 3.12, 20 Golden Questions):
- **Full evaluation check (`20 questions`):** `~0.15 seconds` (sub-second feedback)
- **Baseline save & SQLite transaction:** `~0.04 seconds`
- **Baseline diff comparison:** `~0.03 seconds`

---

## Known Limits & Scope

1. **Golden Questions Scope:** LongProbe is designed to run fast regression tests against a fixed, reviewed set of Golden Questions in your local loop and CI/CD pipelines. It does not inspect arbitrary live user queries in production.
2. **Semantic Mode:** In LongProbe 0.2.0, `match_mode: "semantic"` uses lexical cosine similarity (word-frequency overlap) rather than neural embeddings.
3. **Simulation Note:** The degraded stage in this starter uses a simulated chunking omission in the local fixture to illustrate how dropped chunks are flagged without needing heavy vector DB setup.

---

## Frequently Asked Questions (FAQ)

### 1. How do I start and switch the local fixture?
Run `python fixture_server.py --port 8000 --mode <good|degraded|corrected>`. You can also dynamically switch modes without restarting by sending a POST request:
```bash
curl -X POST http://127.0.0.1:8000/mode -H "Content-Type: application/json" -d '{"mode": "degraded"}'
```

### 2. How do approved golden questions work?
In `goldens.yaml`, each question specifies `status: approved` or `status: draft`. `longprobe check` only evaluates approved questions by default, protecting your test results from unreviewed questions.

### 3. How does baseline selection work?
Baselines are stored in a local SQLite database (`.longprobe/baselines.db`). You explicitly snapshot a trusted run using `longprobe baseline save --label <label>`. During checks or CI pipelines, passing `--baseline <label>` ensures your current code is compared strictly against that reference snapshot.

### 4. How do I interpret a failed check?
When a check fails:
- **Terminal output** displays a color-coded table listing which questions failed, their expected vs. retrieved chunks, and the missing chunk text.
- **GitHub output (`--output github`)** emits `::error` annotations directly on PR lines in GitHub Actions.
- **`longprobe diff`** displays an exact before/after comparison table showing lost chunks and recall drops.

