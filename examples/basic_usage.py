#!/usr/bin/env python3
"""
LongProbe Basic Usage Example

This script demonstrates how to use LongProbe programmatically
to test RAG retrieval quality.

Prerequisites:
    pip install longprobe

Usage:
    python basic_usage.py
    python basic_usage.py --cli     # Demonstrates CLI invocation
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# LongProbe imports — these are the public API surface
# ---------------------------------------------------------------------------
from longprobe import GoldenSet, ProbeResult, ProbeRunner, RetrieverContext
from longprobe.adapters import AbstractRetrieverAdapter
from longprobe.baseline import BaselineManager
from longprobe.cli import main as cli_main
from longprobe.config import LongProbeConfig
from longprobe.matching import MatchMode


# ===========================================================================
# 1. Mock Retriever Adapter
# ===========================================================================
# In a real project you would implement AbstractRetrieverAdapter by wrapping
# your actual RAG pipeline (LangChain, LlamaIndex, custom code, etc.).
# This mock returns deterministic results so the example runs without any
# external dependencies (no database, no LLM API keys).


@dataclass
class RetrievalHit:
    """A single document returned by a retriever."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class MockRetrieverAdapter(AbstractRetrieverAdapter):
    """
    A deterministic mock retriever for demonstration and testing.

    It stores a fixed index of documents and returns results based on
    keyword matching against the query string.  This is intentionally
    simple — a real adapter would call ChromaDB, Pinecone, etc.
    """

    def __init__(self, documents: list[RetrievalHit] | None = None) -> None:
        """
        Args:
            documents: The in-memory document index.  If *None*, a default
                       set of sample legal / corporate documents is loaded.
        """
        self._documents = documents or self._default_documents()

    # -- Public API required by AbstractRetrieverAdapter --------------------

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Retrieve the top-*k* chunks for *query*.

        Returns a list of dicts, each with at least:
            - chunk_id  (str)
            - text      (str)
            - score     (float)
            - metadata  (dict, optional)
        """
        query_lower = query.lower()

        # Score each document by counting how many query tokens appear in it
        scored: list[tuple[float, RetrievalHit]] = []
        query_tokens = set(query_lower.split())
        for doc in self._documents:
            doc_tokens = set(doc.text.lower().split())
            overlap = len(query_tokens & doc_tokens)
            # Normalise score to [0, 1]
            score = overlap / max(len(query_tokens), 1)
            scored.append((score, doc))

        # Sort descending by score, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, doc in scored[:top_k]:
            results.append({
                "chunk_id": doc.chunk_id,
                "text": doc.text,
                "score": round(score, 4),
                "metadata": doc.metadata,
            })

        return results

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _default_documents() -> list[RetrievalHit]:
        """Return a small but realistic set of corporate documents."""
        return [
            RetrievalHit(
                chunk_id="contracts_chunk_42",
                text=(
                    "Termination Clause: Either party may terminate this service agreement "
                    "upon 30 days written notice. In the event of material breach, the "
                    "non-breaching party may terminate immediately. All outstanding fees "
                    "remain due upon termination."
                ),
                score=1.0,
                metadata={"source": "master_service_agreement.pdf", "page": 42},
            ),
            RetrievalHit(
                chunk_id="contracts_chunk_43",
                text=(
                    "Upon termination, the Customer must return all confidential materials "
                    "within 14 business days. Data deletion must be certified in writing. "
                    "Surviving clauses include indemnification, limitation of liability, "
                    "and governing law provisions."
                ),
                score=0.95,
                metadata={"source": "master_service_agreement.pdf", "page": 43},
            ),
            RetrievalHit(
                chunk_id="finance_chunk_10",
                text=(
                    "Payment terms for enterprise clients are net 30 days from invoice "
                    "date. Enterprise clients receive a 15% discount on annual contracts "
                    "exceeding $100,000. Late payments accrue interest at 1.5% per month."
                ),
                score=0.90,
                metadata={"source": "pricing_guide.pdf", "page": 10},
            ),
            RetrievalHit(
                chunk_id="legal_chunk_05",
                text=(
                    "The following officers are authorized to sign contracts on behalf of "
                    "the company: CEO, CFO, and General Counsel. Signatures from any other "
                    "individual require prior written authorization from the Board."
                ),
                score=0.88,
                metadata={"source": "corporate_governance.pdf", "page": 5},
            ),
            RetrievalHit(
                chunk_id="billing_chunk_22",
                text=(
                    "Refund policy for annual subscriptions: Customers are entitled to a "
                    "full refund within 30 days of purchase. After 30 days, a prorated "
                    "refund after 30 days is issued based on remaining months. No refunds "
                    "are available after 6 months."
                ),
                score=0.85,
                metadata={"source": "billing_policy.pdf", "page": 22},
            ),
            RetrievalHit(
                chunk_id="security_chunk_01",
                text=(
                    "User data at rest is encrypted using industry standard algorithms. "
                    "All databases use AES-256 encryption at rest. Encryption keys are "
                    "managed via AWS KMS with automatic rotation every 90 days."
                ),
                score=0.82,
                metadata={"source": "security_whitepaper.pdf", "page": 1},
            ),
            RetrievalHit(
                chunk_id="security_chunk_02",
                text=(
                    "Data in transit is protected with TLS 1.3. All API endpoints enforce "
                    "HTTPS. Certificate pinning is used for mobile clients."
                ),
                score=0.78,
                metadata={"source": "security_whitepaper.pdf", "page": 2},
            ),
            RetrievalHit(
                chunk_id="hr_chunk_15",
                text=(
                    "Employee onboarding requires completion of security awareness training "
                    "within the first week. Annual refresher training is mandatory for all "
                    "staff with access to customer data."
                ),
                score=0.70,
                metadata={"source": "employee_handbook.pdf", "page": 15},
            ),
        ]


# ===========================================================================
# 2. Creating a Golden Set Programmatically (no YAML needed)
# ===========================================================================

def create_golden_set() -> GoldenSet:
    """
    Build a GoldenSet entirely in Python.

    A GoldenSet defines the expected retrieval behaviour: for each question,
    it lists the chunks that *must* appear in the retriever's top-k results.
    This is the same structure you would express in goldens.yaml, but built
    dynamically — useful for generated / parametric test suites.
    """
    golden = GoldenSet(name="demo-golden-set", version="1.0")

    # Question 1 — match by exact chunk ID
    golden.add_question(
        id="q1",
        question="What is the termination clause in the service agreement?",
        match_mode=MatchMode.ID,
        required_chunks=["contracts_chunk_42", "contracts_chunk_43"],
        top_k=5,
        tags=["contracts", "legal", "critical"],
    )

    # Question 2 — match by substring text search
    golden.add_question(
        id="q2",
        question="What are the payment terms for enterprise clients?",
        match_mode=MatchMode.TEXT,
        required_chunks=[
            "net 30 days from invoice date",
            "enterprise clients receive a 15% discount",
        ],
        top_k=5,
        tags=["finance", "enterprise"],
    )

    # Question 3 — semantic similarity (cosine >= threshold)
    golden.add_question(
        id="q3",
        question="Who are the authorized signatories for the company?",
        match_mode=MatchMode.SEMANTIC,
        semantic_threshold=0.80,
        required_chunks=[
            "The following officers are authorized to sign contracts on behalf "
            "of the company: CEO, CFO, and General Counsel."
        ],
        top_k=10,
        tags=["legal", "governance"],
    )

    # Question 4 — text match on refund policy
    golden.add_question(
        id="q4",
        question="What is the refund policy for annual subscriptions?",
        match_mode=MatchMode.TEXT,
        required_chunks=["full refund within 30 days", "prorated refund after 30 days"],
        top_k=5,
        tags=["billing", "policy"],
    )

    # Question 5 — text match on encryption
    golden.add_question(
        id="q5",
        question="How is user data encrypted at rest?",
        match_mode=MatchMode.TEXT,
        required_chunks=["AES-256 encryption", "data at rest is encrypted using industry standard algorithms"],
        top_k=5,
        tags=["security", "compliance"],
    )

    return golden


# ===========================================================================
# 3. Loading a Golden Set from YAML
# ===========================================================================

def load_golden_set_from_yaml(path: str | Path) -> GoldenSet:
    """
    Load a golden set from a YAML file (same schema as goldens.yaml).

    This is what the CLI does under the hood when you run:
        longprobe check --goldens goldens.yaml
    """
    golden = GoldenSet.from_yaml(path)
    return golden


# ===========================================================================
# 4. Running a Probe
# ===========================================================================

def run_probe(golden: GoldenSet, retriever: AbstractRetrieverAdapter) -> ProbeResult:
    """
    Execute the probe: for every question in the golden set, query the
    retriever and check whether the required chunks appear.

    The ProbeRunner orchestrates retrieval, matching, and scoring.
    """
    config = LongProbeConfig(
        scoring=LongProbeConfig.Scoring(
            recall_threshold=0.8,
            fail_on_regression=True,
        ),
    )

    runner = ProbeRunner(config=config, retriever=retriever)

    # The runner returns a ProbeResult containing per-question scores and
    # an overall aggregate.
    result = runner.run(golden)
    return result


# ===========================================================================
# 5. Checking Results
# ===========================================================================

def check_results(result: ProbeResult) -> None:
    """
    Pretty-print the probe results and exit with a non-zero code if any
    question fell below the recall threshold.
    """
    print("\n" + "=" * 64)
    print("  LONGPROBE RESULTS")
    print("=" * 64)

    print(f"\n  Overall Recall : {result.overall_recall:.1%}")
    print(f"  Pass Threshold : {result.threshold:.1%}")
    print(f"  Status         : {'✅ PASSED' if result.passed else '❌ FAILED'}")
    print(f"  Questions      : {result.total_questions}")
    print(f"  Passed         : {result.passed_questions}")
    print(f"  Failed         : {result.failed_questions}")

    # Per-question breakdown
    print("\n" + "-" * 64)
    print(f"  {'ID':<6} {'Recall':>8}   {'Status':<8}   {'Missing Chunks'}")
    print("-" * 64)

    for q in result.question_results:
        missing = ", ".join(q.missing_chunks) if q.missing_chunks else "—"
        status = "✅" if q.passed else "❌"
        print(f"  {q.id:<6} {q.recall:>7.1%}   {status:<8}   {missing}")

    print("-" * 64 + "\n")

    if not result.passed:
        print("⚠️  Some questions did not meet the recall threshold.")
        print("   Review the missing chunks above and adjust your retriever or golden set.\n")
        sys.exit(1)
    else:
        print("🎉  All questions passed! Your retriever is performing well.\n")


# ===========================================================================
# 6. Saving and Comparing Baselines
# ===========================================================================

def save_baseline(result: ProbeResult, label: str = "latest") -> Path:
    """
    Persist the current probe result as a named baseline.

    Baselines are stored in a local SQLite database so you can track
    regressions over time.  When a future probe scores lower than the
    saved baseline on any question, LongProbe flags it as a regression.

    Args:
        result: The probe result to save.
        label:  A human-readable label (default: "latest").

    Returns:
        Path to the baseline database file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "baselines.db"
    # Use a persistent path for real usage
    db_path = Path(".longprobe") / "baselines.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    manager = BaselineManager(db_path=str(db_path))
    manager.save(result=result, label=label)
    print(f"💾  Baseline saved → {db_path}  (label={label!r})")
    return db_path


def compare_baseline(
    result: ProbeResult,
    db_path: str | Path = ".longprobe/baselines.db",
    label: str = "latest",
) -> None:
    """
    Compare the current probe result against a stored baseline.

    LongProbe will report:
      - Regressions: questions that previously passed but now fail
      - Improvements: questions that previously failed but now pass
      - Unchanged: questions with the same pass/fail status
    """
    manager = BaselineManager(db_path=str(db_path))
    comparison = manager.compare(result=result, label=label)

    print("\n" + "=" * 64)
    print("  BASELINE COMPARISON")
    print("=" * 64)

    if comparison.regressions:
        print("\n  📉 REGRESSIONS (previously passing, now failing):")
        for reg in comparison.regressions:
            print(f"     • {reg.question_id}: recall {reg.previous_recall:.1%} → {reg.current_recall:.1%}")
            print(f"       Lost chunks: {', '.join(reg.lost_chunks)}")

    if comparison.improvements:
        print("\n  📈 IMPROVEMENTS (previously failing, now passing):")
        for imp in comparison.improvements:
            print(f"     • {imp.question_id}: recall {imp.previous_recall:.1%} → {imp.current_recall:.1%}")

    if comparison.unchanged:
        print(f"\n  ➡️  Unchanged: {len(comparison.unchanged)} questions")

    if not comparison.regressions:
        print("\n  ✅ No regressions detected.\n")
    else:
        print("\n  ⚠️  Regressions detected — investigate before merging!\n")


# ===========================================================================
# 7. Using the CLI
# ===========================================================================

def demo_cli_invocation() -> None:
    """
    Show how to invoke LongProbe from the command line.

    The CLI is the recommended way to run LongProbe in CI/CD pipelines.
    It reads the golden set from YAML and outputs results in various formats.
    """
    print("\n" + "=" * 64)
    print("  CLI USAGE EXAMPLES")
    print("=" * 64)

    examples = [
        "",
        "  # Basic check (reads longprobe.yaml + goldens.yaml)",
        "  longprobe check",
        "",
        "  # Specify custom paths",
        "  longprobe check --goldens my_goldens.yaml --config my_config.yaml",
        "",
        "  # GitHub Actions output format (posts annotations)",
        "  longprobe check --output github",
        "",
        "  # JSON output for programmatic consumption",
        "  longprobe check --output json --output-file results.json",
        "",
        "  # Save a baseline after a successful run",
        "  longprobe check --save-baseline main-branch",
        "",
        "  # Compare against a named baseline",
        "  longprobe check --compare-baseline main-branch",
        "",
        "  # Run only questions with specific tags",
        "  longprobe check --tags legal,security",
        "",
        "  # Override top-k and threshold from CLI",
        "  longprobe check --top-k 10 --threshold 0.9",
    ]

    print("\n".join(examples))
    print()

    # Actually invoke the CLI with the example golden set to prove it works
    golden_path = Path(__file__).parent / "goldens.yaml"
    config_path = Path(__file__).parent / "longprobe.yaml"

    if golden_path.exists():
        print(f"  🚀  Running: longprobe check --goldens {golden_path}")
        print()
        try:
            # We pass sys.argv so the CLI parses our arguments
            original_argv = sys.argv
            sys.argv = [
                "longprobe",
                "check",
                "--goldens", str(golden_path),
                "--config", str(config_path) if config_path.exists() else "",
                "--output", "text",
            ]
            cli_main()
            sys.argv = original_argv
        except SystemExit:
            # The CLI calls sys.exit on completion — catch it here
            sys.argv = original_argv
        except Exception as exc:
            sys.argv = original_argv
            print(f"  ℹ️  CLI demo skipped ({exc}). This is expected if LongProbe")
            print(f"      is not installed in the current environment.\n")
    else:
        print(f"  ℹ️  {golden_path} not found — skipping live CLI demo.\n")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """
    Run the full LongProbe demonstration pipeline.
    """
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║               LongProbe — Basic Usage Example               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ------------------------------------------------------------------
    # Step 1: Create the mock retriever (wraps your actual RAG pipeline)
    # ------------------------------------------------------------------
    print("\n[1/6] Creating mock retriever …")
    retriever = MockRetrieverAdapter()
    print(f"       Indexed {len(retriever._documents)} documents.\n")

    # ------------------------------------------------------------------
    # Step 2: Build a golden set programmatically
    # ------------------------------------------------------------------
    print("[2/6] Building golden set from Python …")
    golden = create_golden_set()
    print(f"       Created {len(golden.questions)} questions.\n")

    # ------------------------------------------------------------------
    # Step 3: Run the probe
    # ------------------------------------------------------------------
    print("[3/6] Running probe …")
    result = run_probe(golden, retriever)
    print("       Probe complete.\n")

    # ------------------------------------------------------------------
    # Step 4: Check results
    # ------------------------------------------------------------------
    print("[4/6] Checking results …")
    check_results(result)

    # ------------------------------------------------------------------
    # Step 5: Save baseline
    # ------------------------------------------------------------------
    print("[5/6] Saving baseline …")
    save_baseline(result, label="demo-baseline")

    # ------------------------------------------------------------------
    # Step 6: Compare against baseline
    # ------------------------------------------------------------------
    print("[6/6] Comparing against baseline …")
    compare_baseline(result, label="demo-baseline")

    # ------------------------------------------------------------------
    # Bonus: Show CLI usage
    # ------------------------------------------------------------------
    demo_cli_invocation()


if __name__ == "__main__":
    main()
