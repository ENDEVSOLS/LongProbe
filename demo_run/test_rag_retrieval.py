#!/usr/bin/env python3
"""Quick Check - Test RAG Retrieval Quality (TUI Version)"""

import time
import subprocess
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table

console = Console()

def create_progress_display(status="running", results=None):
    """Create a live progress display"""
    table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    table.add_column("Phase", style="bold", width=15)
    table.add_column("Status", width=5)
    table.add_column("Details", style="")
    
    if status == "loading":
        table.add_row(
            "Loading",
            "[cyan]⚙️[/cyan]",
            "[cyan]Reading configuration and golden set...[/cyan]"
        )
        table.add_row(
            "Testing",
            "[dim]⏳[/dim]",
            "[dim]Waiting...[/dim]"
        )
        table.add_row(
            "Results",
            "[dim]⏳[/dim]",
            "[dim]Waiting...[/dim]"
        )
    elif status == "testing":
        table.add_row(
            "Loading",
            "[green]✓[/green]",
            "[green]Configuration loaded[/green]"
        )
        table.add_row(
            "Testing",
            "[cyan]⚙️[/cyan]",
            "[cyan]Running RAG regression tests...[/cyan]"
        )
        table.add_row(
            "Results",
            "[dim]⏳[/dim]",
            "[dim]Waiting...[/dim]"
        )
    elif status == "done" and results:
        table.add_row(
            "Loading",
            "[green]✓[/green]",
            "[green]Configuration loaded[/green]"
        )
        table.add_row(
            "Testing",
            "[green]✓[/green]",
            f"[green]Tested {results['total']} questions[/green]"
        )
        table.add_row(
            "Results",
            "[green]✓[/green]",
            f"[green]Recall: {results['recall']:.1%} | Pass Rate: {results['pass_rate']:.1%}[/green]"
        )
    
    return table

def create_results_table(results_data):
    """Create a detailed results table"""
    table = Table(show_header=True, box=None, expand=True)
    table.add_column("Question ID", style="bold")
    table.add_column("Recall", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Found/Required", justify="center")
    
    for result in results_data:
        status = "[green]✓[/green]" if result['passed'] else "[red]✗[/red]"
        recall_style = "green" if result['recall'] >= 1.0 else "yellow" if result['recall'] >= 0.5 else "red"
        table.add_row(
            result['id'],
            f"[{recall_style}]{result['recall']:.1%}[/{recall_style}]",
            status,
            f"{result['found']}/{result['required']}"
        )
    
    return table

# Main execution
console.print("\n[bold blue]🔬 LongProbe - Quick Check[/bold blue]\n")

# Simulate the longprobe check command with live updates
with Live(console=console, refresh_per_second=10) as live:
    # Phase 1: Loading
    status = Group(
        Text("Initializing LongProbe...\n", style="cyan"),
        create_progress_display(status="loading")
    )
    live.update(Panel(status, border_style="cyan", title="Quick Check"))
    time.sleep(1.2)
    
    # Phase 2: Testing
    status = Group(
        Text("Configuration loaded\n", style="green"),
        create_progress_display(status="testing")
    )
    live.update(Panel(status, border_style="cyan", title="Quick Check"))
    time.sleep(2.0)
    
    # Simulate results (in real scenario, this would parse actual output)
    results = {
        'total': 3,
        'recall': 0.667,
        'pass_rate': 0.667
    }
    
    # Phase 3: Complete
    status = Group(
        Text("Tests complete!\n", style="green"),
        create_progress_display(status="done", results=results)
    )
    live.update(Panel(status, border_style="green", title="Quick Check"))
    time.sleep(0.8)

# Show detailed results
console.print()

results_data = [
    {'id': 'q_dummy', 'recall': 0.0, 'passed': False, 'found': 0, 'required': 1},
    {'id': 'q_what_is_the_refund_policy', 'recall': 1.0, 'passed': True, 'found': 4, 'required': 4},
    {'id': 'q_how_is_data_encrypted', 'recall': 1.0, 'passed': True, 'found': 4, 'required': 4},
]

results_table = create_results_table(results_data)
console.print(Panel(results_table, title="📊 Test Results", border_style="cyan"))

# Summary
summary = Text()
summary.append("Overall Recall: ", style="bold")
summary.append("66.7%\n", style="green")
summary.append("Pass Rate: ", style="bold")
summary.append("66.7%\n", style="green")
summary.append("Status: ", style="bold")
summary.append("2/3 tests passed", style="yellow")

console.print()
console.print(Panel(summary, title="📈 Summary", border_style="yellow"))
console.print()
