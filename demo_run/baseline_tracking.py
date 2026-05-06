#!/usr/bin/env python3
"""Baseline Tracking - Detect RAG Regressions Over Time"""

from longprobe import LongProbe
from longprobe.adapters import create_adapter
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

console.print("\n[bold blue]📊 Baseline Tracking - Detect RAG Regressions[/bold blue]\n")

# Setup
console.print("[cyan]⚙️  Setting up...[/cyan]")
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
console.print("[green]   ✓ Ready[/green]\n")

# Step 1: Run initial test and save baseline
console.print("[bold magenta]STEP 1: Establish Baseline[/bold magenta]")
console.print("[cyan]Running tests...[/cyan]")
report = probe.run()
console.print(f"[green]✓ Tests complete - Recall: {report.overall_recall:.1%}[/green]")

console.print("[cyan]Saving as baseline 'production'...[/cyan]")
probe.save_baseline(label="production")
console.print("[green]✓ Baseline saved[/green]\n")

# Step 2: Run again and compare
console.print("[bold magenta]STEP 2: Compare Against Baseline[/bold magenta]")
console.print("[cyan]Running tests again...[/cyan]")
report2 = probe.run()
console.print(f"[green]✓ Tests complete - Recall: {report2.overall_recall:.1%}[/green]")

console.print("[cyan]Comparing with baseline 'production'...[/cyan]")
diff = probe.diff(baseline_label="production")

# Show comparison results in a nice panel
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
content.append(f"➡️ Unchanged: {len(unchanged)} questions ", style="cyan")

panel = Panel(content, title="📈 BASELINE COMPARISON", border_style="cyan", expand=False)
console.print(panel)

# Final verdict
if not diff.get("regressions"):
    console.print("\n[bold green]✅ Safe to deploy - no regressions detected![/bold green]\n")
else:
    console.print("\n[bold red]❌ DO NOT DEPLOY - fix regressions first![/bold red]\n")
