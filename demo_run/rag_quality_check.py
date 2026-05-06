#!/usr/bin/env python3
"""RAG Quality Check - LongProbe Python API Demo"""

from longprobe import LongProbe
from longprobe.adapters import create_adapter
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

console.print("\n[bold blue]🔬 RAG Quality Check with LongProbe[/bold blue]\n")

# Create ChromaDB adapter
console.print("[cyan]📦 Connecting to ChromaDB...[/cyan]")
adapter = create_adapter(
    "chroma",
    collection_name="default",
    persist_directory="./chroma_db"
)
console.print("[green]   ✓ Connected to vector store[/green]\n")

# Initialize LongProbe
console.print("[cyan]⚙️  Loading golden set...[/cyan]")
probe = LongProbe(
    adapter=adapter,
    goldens_path="goldens.yaml",
    config_path="longprobe.yaml"
)
console.print(f"[green]   ✓ Loaded {len(probe.golden_set.questions)} golden questions[/green]")

# Show what we're testing
console.print(f"\n[bold]📋 Golden Questions:[/bold]")
for i, q in enumerate(probe.golden_set.questions, 1):
    console.print(f"   {i}. {q.question}")
    console.print(f"      Required chunks: {len(q.required_chunks)}")

# Run the test
console.print(f"\n[cyan]🚀 Running RAG regression tests...[/cyan]\n")
report = probe.run()

# Print detailed results in a panel
results_content = Text()
for result in report.results:
    status_text = "✓ PASS" if result.passed else "✗ FAIL"
    status_style = "green" if result.passed else "red"
    
    results_content.append(f"{result.question_id}: ", style="bold")
    results_content.append(f"{status_text}\n", style=f"bold {status_style}")
    results_content.append(f"  Question: {result.question}\n")
    results_content.append(f"  Recall:   {result.recall_score:.1%}\n", style="green")
    found = len(result.found_chunks)
    required = len(result.required_chunks)
    results_content.append(f"  Found:    {found}/{required} chunks\n")
    if result.missing_chunks:
        missing_preview = result.missing_chunks[0][:35] + "..." if len(result.missing_chunks[0]) > 35 else result.missing_chunks[0]
        results_content.append(f"  Missing:  {missing_preview}\n", style="red")
    results_content.append("\n")

results_panel = Panel(results_content, title="📊 TEST RESULTS", border_style="cyan", expand=False)
console.print(results_panel)

# Summary panel
summary_content = Text()
summary_content.append(f"Overall Recall: {report.overall_recall:.1%}\n", style="green")
summary_content.append(f"Pass Rate:      {report.pass_rate:.1%}\n", style="green")
summary_content.append("\n")

passed = sum(1 for r in report.results if r.passed)
failed = len(report.results) - passed
summary_content.append(f"✓ Passed:       {passed}/{len(report.results)}\n", style="green")
if failed > 0:
    summary_content.append(f"✗ Failed:       {failed}/{len(report.results)}", style="red")

summary_style = "green" if failed == 0 else "yellow"
summary_panel = Panel(summary_content, title="📈 SUMMARY", border_style=summary_style, expand=False)
console.print(summary_panel)

if passed == len(report.results):
    console.print("\n[bold green]✅ All tests passed! RAG quality is good.[/bold green]\n")
else:
    console.print(f"\n[yellow]⚠️  {failed} test(s) failed - review missing chunks[/yellow]\n")
