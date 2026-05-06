#!/usr/bin/env python3
"""RAG Quality Check - LongProbe Python API Demo (TUI Version)"""

import time
from longprobe import LongProbe
from longprobe.adapters import create_adapter
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table

console = Console()

def create_progress_display(phase="setup", questions_count=0, current_test=0, total_tests=0):
    """Create a live progress display"""
    table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    table.add_column("Phase", style="bold", width=15)
    table.add_column("Status", width=5)
    table.add_column("Details", style="")
    
    # Setup phase
    if phase == "setup":
        table.add_row("Setup", "[cyan]⚙️[/cyan]", "[cyan]Connecting to ChromaDB...[/cyan]")
        table.add_row("Loading", "[dim]⏳[/dim]", "[dim]Waiting...[/dim]")
        table.add_row("Testing", "[dim]⏳[/dim]", "[dim]Waiting...[/dim]")
    elif phase == "loading":
        table.add_row("Setup", "[green]✓[/green]", "[green]Connected to vector store[/green]")
        table.add_row("Loading", "[cyan]⚙️[/cyan]", f"[cyan]Loading golden questions...[/cyan]")
        table.add_row("Testing", "[dim]⏳[/dim]", "[dim]Waiting...[/dim]")
    elif phase == "ready":
        table.add_row("Setup", "[green]✓[/green]", "[green]Connected to vector store[/green]")
        table.add_row("Loading", "[green]✓[/green]", f"[green]Loaded {questions_count} golden questions[/green]")
        table.add_row("Testing", "[dim]⏳[/dim]", "[dim]Ready to start...[/dim]")
    elif phase == "testing":
        table.add_row("Setup", "[green]✓[/green]", "[green]Connected to vector store[/green]")
        table.add_row("Loading", "[green]✓[/green]", f"[green]Loaded {questions_count} golden questions[/green]")
        table.add_row("Testing", "[cyan]⚙️[/cyan]", f"[cyan]Running tests... ({current_test}/{total_tests})[/cyan]")
    elif phase == "done":
        table.add_row("Setup", "[green]✓[/green]", "[green]Connected to vector store[/green]")
        table.add_row("Loading", "[green]✓[/green]", f"[green]Loaded {questions_count} golden questions[/green]")
        table.add_row("Testing", "[green]✓[/green]", f"[green]All {total_tests} tests complete[/green]")
    
    return table

def create_results_table(results):
    """Create detailed results table"""
    table = Table(show_header=True, box=None, expand=True)
    table.add_column("Question", style="bold", width=30)
    table.add_column("Recall", justify="right", width=10)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Chunks", justify="center", width=10)
    
    for result in results:
        status = "[green]✓ PASS[/green]" if result.passed else "[red]✗ FAIL[/red]"
        recall_style = "green" if result.recall_score >= 1.0 else "yellow" if result.recall_score >= 0.5 else "red"
        found = len(result.found_chunks)
        required = len(result.required_chunks)
        
        # Truncate long questions
        question = result.question[:27] + "..." if len(result.question) > 30 else result.question
        
        table.add_row(
            question,
            f"[{recall_style}]{result.recall_score:.1%}[/{recall_style}]",
            status,
            f"{found}/{required}"
        )
    
    return table

# Main execution
console.print("\n[bold blue]🔬 RAG Quality Check with LongProbe[/bold blue]\n")

with Live(console=console, refresh_per_second=10) as live:
    # Phase 1: Setup
    status = Group(
        create_progress_display(phase="setup")
    )
    live.update(Panel(status, border_style="cyan", title="RAG Quality Check"))
    time.sleep(0.8)
    
    adapter = create_adapter(
        "chroma",
        collection_name="default",
        persist_directory="./chroma_db"
    )
    
    # Phase 2: Loading
    status = Group(
        create_progress_display(phase="loading")
    )
    live.update(Panel(status, border_style="cyan", title="RAG Quality Check"))
    time.sleep(0.8)
    
    probe = LongProbe(
        adapter=adapter,
        goldens_path="goldens.yaml",
        config_path="longprobe.yaml"
    )
    
    questions_count = len(probe.golden_set.questions)
    
    # Phase 3: Ready
    status = Group(
        create_progress_display(phase="ready", questions_count=questions_count)
    )
    live.update(Panel(status, border_style="cyan", title="RAG Quality Check"))
    time.sleep(0.5)
    
    # Show golden questions
    questions_text = Text("\nGolden Questions:\n", style="bold")
    for i, q in enumerate(probe.golden_set.questions, 1):
        questions_text.append(f"  {i}. {q.question}\n")
        questions_text.append(f"     Required chunks: {len(q.required_chunks)}\n", style="dim")
    
    status = Group(
        create_progress_display(phase="ready", questions_count=questions_count),
        questions_text
    )
    live.update(Panel(status, border_style="cyan", title="RAG Quality Check"))
    time.sleep(1.0)
    
    # Phase 4: Testing
    status = Group(
        create_progress_display(phase="testing", questions_count=questions_count, current_test=0, total_tests=questions_count),
        Text("\n🚀 Running RAG regression tests...", style="cyan")
    )
    live.update(Panel(status, border_style="cyan", title="RAG Quality Check"))
    time.sleep(0.5)
    
    report = probe.run()
    
    # Phase 5: Complete
    status = Group(
        create_progress_display(phase="done", questions_count=questions_count, total_tests=questions_count),
        Text("\n✓ Tests complete!", style="green")
    )
    live.update(Panel(status, border_style="green", title="RAG Quality Check"))
    time.sleep(0.5)

# Show detailed results
console.print()
results_table = create_results_table(report.results)
console.print(Panel(results_table, title="📊 Test Results", border_style="cyan"))

# Summary
summary = Text()
summary.append("Overall Recall: ", style="bold")
summary.append(f"{report.overall_recall:.1%}\n", style="green")
summary.append("Pass Rate: ", style="bold")
summary.append(f"{report.pass_rate:.1%}\n", style="green")

passed = sum(1 for r in report.results if r.passed)
failed = len(report.results) - passed
summary.append("\n")
summary.append(f"✓ Passed: {passed}/{len(report.results)}\n", style="green")
if failed > 0:
    summary.append(f"✗ Failed: {failed}/{len(report.results)}", style="red")

summary_style = "green" if failed == 0 else "yellow"
console.print()
console.print(Panel(summary, title="📈 Summary", border_style=summary_style))

if passed == len(report.results):
    console.print("\n[bold green]✅ All tests passed! RAG quality is good.[/bold green]\n")
else:
    console.print(f"\n[yellow]⚠️  {failed} test(s) failed - review missing chunks[/yellow]\n")
