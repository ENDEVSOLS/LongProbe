#!/usr/bin/env python3
"""Baseline Tracking - Detect RAG Regressions Over Time (TUI Version)"""

import time
from longprobe import LongProbe
from longprobe.adapters import create_adapter
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table

console = Console()

def create_progress_display(step1_status="pending", step2_status="pending", recall1=None, recall2=None):
    """Create a live progress display"""
    table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    table.add_column("Step", style="bold", width=10)
    table.add_column("Status", width=5)
    table.add_column("Details", style="")
    
    # Step 1
    if step1_status == "done":
        table.add_row(
            "STEP 1",
            "[green]✓[/green]",
            f"[green]Baseline established - Recall: {recall1:.1%}[/green]"
        )
    elif step1_status == "running":
        table.add_row(
            "STEP 1",
            "[cyan]⚙️[/cyan]",
            "[cyan]Establishing baseline...[/cyan]"
        )
    else:
        table.add_row(
            "STEP 1",
            "[dim]⏳[/dim]",
            "[dim]Pending...[/dim]"
        )
    
    # Step 2
    if step2_status == "done":
        table.add_row(
            "STEP 2",
            "[green]✓[/green]",
            f"[green]Comparison complete - Recall: {recall2:.1%}[/green]"
        )
    elif step2_status == "running":
        table.add_row(
            "STEP 2",
            "[cyan]⚙️[/cyan]",
            "[cyan]Running comparison...[/cyan]"
        )
    else:
        table.add_row(
            "STEP 2",
            "[dim]⏳[/dim]",
            "[dim]Waiting...[/dim]"
        )
    
    return table

def create_results_panel(diff):
    """Create the comparison results panel"""
    content = Text()
    
    if diff.get("regressions"):
        content.append("⚠️  REGRESSIONS DETECTED\n\n", style="bold red")
        for reg in diff["regressions"]:
            content.append(f"  ✗ {reg['question_id']}\n", style="red")
            content.append(f"    Recall: {reg['baseline_recall']:.1%} → {reg['current_recall']:.1%}\n")
            content.append(f"    Lost: {len(reg['lost_chunks'])} chunks\n\n")
    else:
        content.append("✓ No regressions detected\n\n", style="green")
    
    if diff.get("improvements"):
        content.append("📈 IMPROVEMENTS\n\n", style="bold green")
        for imp in diff["improvements"]:
            content.append(f"  ✓ {imp['question_id']}\n", style="green")
            content.append(f"    Recall: {imp['baseline_recall']:.1%} → {imp['current_recall']:.1%}\n\n")
    
    unchanged = diff.get("unchanged", [])
    content.append(f"➡️ Unchanged: {len(unchanged)} questions", style="cyan")
    
    return Panel(content, title="📈 BASELINE COMPARISON", border_style="cyan")

# Main execution
console.print("\n[bold blue]📊 Baseline Tracking - Detect RAG Regressions[/bold blue]\n")

with Live(console=console, refresh_per_second=10) as live:
    # Setup phase
    status = Panel("[cyan]⚙️  Setting up ChromaDB connection...[/cyan]", border_style="cyan")
    live.update(status)
    time.sleep(0.8)
    
    adapter = create_adapter(
        "chroma",
        collection_name="default",
        persist_directory="./chroma_db"
    )
    probe = LongProbe(
        adapter=adapter,
        goldens_path="goldens.yaml",
        config_path="longprobe.yaml"
    )
    
    status = Group(
        Text("✓ Setup complete\n", style="green"),
        create_progress_display()
    )
    live.update(Panel(status, border_style="cyan"))
    time.sleep(0.5)
    
    # Step 1: Establish baseline
    status = Group(
        Text("✓ Setup complete\n", style="green"),
        Text("STEP 1: Establish Baseline", style="bold magenta"),
        Text("⚙️  Running tests...\n", style="cyan"),
        create_progress_display(step1_status="running")
    )
    live.update(Panel(status, border_style="cyan"))
    
    report = probe.run()
    time.sleep(0.5)
    
    status = Group(
        Text("✓ Setup complete\n", style="green"),
        Text("STEP 1: Establish Baseline", style="bold magenta"),
        Text(f"✓ Tests complete - Recall: {report.overall_recall:.1%}", style="green"),
        Text("⚙️  Saving baseline 'production'...\n", style="cyan"),
        create_progress_display(step1_status="running")
    )
    live.update(Panel(status, border_style="cyan"))
    
    probe.save_baseline(label="production")
    time.sleep(0.5)
    
    status = Group(
        Text("✓ Setup complete\n", style="green"),
        create_progress_display(step1_status="done", recall1=report.overall_recall)
    )
    live.update(Panel(status, border_style="green"))
    time.sleep(0.8)
    
    # Step 2: Compare against baseline
    status = Group(
        Text("✓ Setup complete\n", style="green"),
        create_progress_display(step1_status="done", recall1=report.overall_recall),
        Text("\nSTEP 2: Compare Against Baseline", style="bold magenta"),
        Text("⚙️  Running tests again...", style="cyan")
    )
    live.update(Panel(status, border_style="cyan"))
    
    report2 = probe.run()
    time.sleep(0.5)
    
    status = Group(
        Text("✓ Setup complete\n", style="green"),
        create_progress_display(step1_status="done", step2_status="running", recall1=report.overall_recall),
        Text("\nSTEP 2: Compare Against Baseline", style="bold magenta"),
        Text(f"✓ Tests complete - Recall: {report2.overall_recall:.1%}", style="green"),
        Text("⚙️  Comparing with baseline 'production'...", style="cyan")
    )
    live.update(Panel(status, border_style="cyan"))
    
    diff = probe.diff(baseline_label="production")
    time.sleep(0.5)
    
    # Final progress
    status = Group(
        Text("✓ Setup complete\n", style="green"),
        create_progress_display(step1_status="done", step2_status="done", recall1=report.overall_recall, recall2=report2.overall_recall)
    )
    live.update(Panel(status, border_style="green"))
    time.sleep(0.5)

# Show final results outside of Live context
console.print()
console.print(create_results_panel(diff))

# Final verdict
if not diff.get("regressions"):
    console.print("\n[bold green]✅ Safe to deploy - no regressions detected![/bold green]\n")
else:
    console.print("\n[bold red]❌ DO NOT DEPLOY - fix regressions first![/bold red]\n")
