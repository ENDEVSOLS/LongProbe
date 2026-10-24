"""Self-contained v0.2.0 recording scenarios using the real evaluation API."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from longprobe import LongProbe
from longprobe.adapters import AbstractRetrieverAdapter
from longprobe.core.golden import GoldenQuestion, GoldenSet

# Fixed source evidence keeps recordings reproducible without SDKs or model downloads.
CASES = [
    ("refund", "What is the refund policy?", "Refunds are available within 30 days."),
    ("payment", "What are the payment terms?", "Enterprise payment terms are net 60 days."),
    ("security", "How is data encrypted?", "Data is encrypted at rest using AES-256."),
]


class SampleRetriever(AbstractRetrieverAdapter):
    """Deterministic sample retriever; this is not a vector database benchmark."""

    def __init__(self):
        self.missing = set()
        self.calls = 0

    def retrieve(self, query, top_k=5):
        self.calls += 1
        return [
            {"id": key, "text": evidence, "score": 1.0, "metadata": {"source": "sample"}}
            for key, question, evidence in CASES
            if query == question and key not in self.missing
        ][:top_k]


def results_table(report):
    table = Table(show_header=True, box=None, expand=True, padding=(0, 1))
    table.add_column("Question", ratio=3)
    table.add_column("Recall", justify="right")
    table.add_column("Evidence found / required", justify="center")
    table.add_column("Status", justify="center")
    for result in report.results:
        style = "green" if result.passed else "red"
        table.add_row(
            result.question,
            f"[{style}]{result.recall_score:.1%}[/{style}]",
            f"{len(result.found_chunks)} / {len(result.required_chunks)}",
            f"[{style}]{'✓ PASS' if result.passed else '✗ FAIL'}[/{style}]",
        )
    return table


def progress_table(steps):
    table = Table(show_header=False, box=None, expand=True, padding=(0, 2))
    table.add_column("Step", width=27, style="bold")
    table.add_column("Status", width=3)
    table.add_column("Details", ratio=1)
    for label, status, detail in steps:
        icon, style = {
            "pending": ("○", "dim"),
            "running": ("◌", "cyan"),
            "done": ("✓", "green"),
            "detected": ("!", "yellow"),
        }[status]
        table.add_row(label, f"[{style}]{icon}[/{style}]", Text(detail, style=style))
    return Panel(table, title="Workflow progress", border_style="cyan")


def run_demo(mode):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-delay", action="store_true", help="Run without recording pauses")
    args = parser.parse_args()
    console = Console(no_color=False)

    def pause(seconds=1.0):
        if not args.no_delay:
            time.sleep(seconds)

    title = {
        "workflow": "Complete Workflow",
        "quality": "Monitor RAG Quality",
        "regression": "Detect Retrieval Regressions",
    }[mode]
    console.print(f"\n[bold cyan]🔬 LongProbe — {title}[/bold cyan]")
    console.print("[dim]v0.2.0  •  Reviewed sample documents  •  Isolated demonstration[/dim]\n")
    labels = {
        "workflow": [
            "1. Prepare Golden Questions",
            "2. Review & Approve",
            "3. Evaluate Retrieval",
            "4. Save Baseline",
            "5. Update Retrieval",
            "6. Detect Regression",
        ],
        "quality": ["1. Connect Retriever", "2. Load Golden Questions", "3. Evaluate Retrieval"],
        "regression": [
            "1. Evaluate Retrieval",
            "2. Save Baseline",
            "3. Update Retrieval",
            "4. Compare Baseline",
        ],
    }[mode]
    steps = [(label, "pending", "Waiting…") for label in labels]
    current = 0

    with tempfile.TemporaryDirectory(prefix="longprobe-demo-") as directory:
        root = Path(directory)
        goldens_path = root / "goldens.yaml"
        config_path = root / "longprobe.yaml"
        suite = GoldenSet(
            name="reviewed-sample-documents",
            version="1.0",
            questions=[
                GoldenQuestion(
                    id=key,
                    question=question,
                    required_chunks=[evidence],
                    match_mode="text",
                    status="draft" if mode == "workflow" else "approved",
                    top_k=3,
                )
                for key, question, evidence in CASES
            ],
        )
        suite.to_yaml(str(goldens_path))
        config_path.write_text(
            yaml.safe_dump(
                {
                    "scoring": {"recall_threshold": 0.85},
                    "baseline": {
                        "db_path": str(root / "baselines.db"),
                        "quality_gate": "all_questions",
                        "auto_compare": True,
                    },
                }
            )
        )
        adapter = SampleRetriever()
        if mode == "quality":
            adapter.missing.add("refund")
        with Live(progress_table(steps), console=console, refresh_per_second=10) as live:

            def stage(detail, status="running", extra=None):
                steps[current] = (labels[current], status, detail)
                content = [progress_table(steps)]
                if extra is not None:
                    content.extend([Text(""), extra])
                live.update(Group(*content), refresh=True)
                pause()

            if mode == "workflow":
                questions = Table(show_header=False, box=None, expand=True)
                questions.add_column("Question", ratio=3)
                questions.add_column("Status", width=10)
                for _, question, _ in CASES:
                    questions.add_row(question, "[yellow]DRAFT[/yellow]")
                candidates = Panel(questions, title="Golden questions", border_style="blue")
                stage("Preparing questions and candidate evidence…", extra=candidates)
                stage("3 draft questions prepared", "done", candidates)
                current += 1
                evidence = Table(show_header=False, box=None, expand=True)
                evidence.add_column("Question", ratio=1)
                evidence.add_column("Source evidence", ratio=2)
                for _, question, text in CASES:
                    evidence.add_row(question, text)
                stage(
                    "Review source evidence before approval",
                    extra=Panel(evidence, title="Source review", border_style="blue"),
                )
                for question in suite.questions:
                    question.status = "approved"
                suite.to_yaml(str(goldens_path))
                stage(
                    "3 approved cases • illustrative approval of reviewed sample evidence", "done"
                )
                current += 1
            elif mode == "quality":
                stage("Connecting to the sample retriever…")
                stage("Sample retriever ready", "done")
                current += 1
                stage("Loading reviewed golden questions…")
                stage("3 approved cases • 85% per-question threshold", "done")
                current += 1

            stage("Checking required evidence for each question…")
            probe = LongProbe(
                adapter=adapter, goldens_path=str(goldens_path), config_path=str(config_path)
            )
            report = probe.run()
            passed = sum(result.passed for result in report.results)
            assert adapter.calls == len(CASES)
            if mode == "quality":
                assert passed == 2 and len(report.results) == 3
                assert not report.quality["passed"]
                assert report.quality["failed_questions"] == ["refund"]
            else:
                assert report.quality["passed"], report.quality["reasons"]
            stage(
                f"Recall: {report.overall_recall:.1%} • Passed: {passed} / {len(report.results)}",
                "done" if report.quality["passed"] else "detected",
                Panel(results_table(report), title="Retrieval results", border_style="cyan"),
            )

            if mode == "quality":
                summary = Panel(
                    "[bold red]✗ Retrieval quality checks failed — 1 question needs attention[/bold red]\n"
                    f"Overall recall: {report.overall_recall:.1%}  •  "
                    f"Pass rate: {report.pass_rate:.1%}  •  Threshold: 85%\n"
                    f"Missing evidence: {CASES[0][2]}\n"
                    "[dim]One evaluation • No historical reference • No baseline saved automatically[/dim]",
                    title="Quality summary",
                    border_style="red",
                )
                live.update(
                    Group(
                        progress_table(steps),
                        Text(""),
                        Panel(results_table(report), title="Test results", border_style="cyan"),
                        Text(""),
                        summary,
                    ),
                    refresh=True,
                )
                pause(3)
                return

            current += 1
            stage("Checking baseline eligibility…")
            probe.save_baseline(label="production-v1")
            assert adapter.calls == len(CASES)
            stage("production-v1 saved • all approved questions passed", "done")
            current += 1
            stage("Simulating a retrieval pipeline change…")
            adapter.missing.add("refund")
            stage("Refund evidence removed from sample retrieval", "done")
            current += 1
            stage("Comparing against production-v1…")
            report2 = probe.run(baseline_label="production-v1")
            diff = report2.comparison
            assert not report2.quality["passed"]
            assert diff["compatible"] and len(diff["regressions"]) == 1
            assert adapter.calls == 2 * len(CASES)
            assert probe.baseline_store.load("production-v1").overall_recall == 1.0
            steps[current] = (
                labels[current],
                "detected",
                "1 regression detected • baseline preserved",
            )
            comparison = Table(show_header=False, box=None, expand=True)
            comparison.add_column("Metric", style="bold")
            comparison.add_column("Value")
            comparison.add_row("Reference", report2.baseline_identity["resolved_label"])
            comparison.add_row("Compatibility", "[green]✓ Compatible[/green]")
            comparison.add_row("Regressions", f"[red]{len(diff['regressions'])}[/red]")
            comparison.add_row("Improvements", f"[green]{len(diff['improvements'])}[/green]")
            comparison.add_row(
                "Overall recall", f"100.0% → [yellow]{report2.overall_recall:.1%}[/yellow]"
            )
            comparison_title = "Baseline comparison"
            if mode == "workflow":
                comparison = Table(show_header=True, box=None, expand=True)
                comparison.add_column("Metric", style="bold", ratio=2)
                comparison.add_column("Before", justify="right")
                comparison.add_column("After", justify="right")
                comparison.add_column("Change", justify="right")
                comparison.add_row(
                    "Overall recall",
                    f"[green]{report.overall_recall:.1%}[/green]",
                    f"[yellow]{report2.overall_recall:.1%}[/yellow]",
                    f"[red]{report2.overall_recall - report.overall_recall:+.1%}[/red]",
                )
                comparison.add_row(
                    "Pass rate",
                    f"[green]{report.pass_rate:.1%}[/green]",
                    f"[yellow]{report2.pass_rate:.1%}[/yellow]",
                    f"[red]{report2.pass_rate - report.pass_rate:+.1%}[/red]",
                )
                failed_before = sum(not result.passed for result in report.results)
                failed_after = sum(not result.passed for result in report2.results)
                comparison.add_row(
                    "Failed questions",
                    str(failed_before),
                    f"[red]{failed_after}[/red]",
                    f"[red]+{failed_after - failed_before}[/red]",
                )
                comparison.add_row(
                    "Quality verdict", "[green]PASS[/green]", "[red]FAIL[/red]", "Review required"
                )
                comparison.caption = f"Regressions: {len(diff['regressions'])}  •  Improvements: {len(diff['improvements'])}"
                comparison_title = (
                    f"Before / after — {report2.baseline_identity['resolved_label']} • Compatible"
                )
            lost = diff["regressions"][0]["newly_lost_chunks"][0]
            verdict = Panel(
                f"[bold red]✗ Retrieval quality checks failed[/bold red]\n"
                f"Missing evidence: {lost}\n"
                "[dim]Review the retrieval change. The trusted baseline remains unchanged.[/dim]",
                title="Regression detected",
                border_style="red",
            )
            live.update(
                Group(
                    progress_table(steps),
                    Text(""),
                    Panel(results_table(report2), title="Test results", border_style="cyan"),
                    Text(""),
                    Panel(comparison, title=comparison_title, border_style="cyan"),
                    Text(""),
                    verdict,
                ),
                refresh=True,
            )
            pause(3)
