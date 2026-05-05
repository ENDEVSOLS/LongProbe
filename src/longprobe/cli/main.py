"""
LongProbe CLI — RAG retrieval regression testing harness.

Detect lost chunks before your users notice.

Usage:
    longprobe init              Create config files
    longprobe check             Run probes against golden set
    longprobe baseline save     Save a baseline snapshot
    longprobe baseline list     List saved baselines
    longprobe baseline delete   Delete a baseline snapshot
    longprobe diff              Compare current results against a baseline
    longprobe watch             Watch golden file and re-run on change
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from longprobe.adapters import (
    ChromaAdapter,
    HttpAdapter,
    PineconeAdapter,
    QdrantAdapter,
)
from longprobe.config import GeneratorConfig, ProbeConfig
from longprobe.core.baseline import BaselineStore
from longprobe.core.diff import DiffReporter
from longprobe.core.docparser import DocumentParser
from longprobe.core.generator import QuestionGenerator
from longprobe.core.golden import GoldenQuestion, GoldenSet, generate_question_id
from longprobe.core.scorer import ProbeReport, RecallScorer

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="longprobe",
    help="\U0001f52c RAG retrieval regression testing — detect lost chunks before your users notice.",
    no_args_is_help=True,
)

baseline_app = typer.Typer(
    name="baseline",
    help="Manage baseline snapshots for regression tracking.",
    no_args_is_help=True,
)

app.add_typer(baseline_app, name="baseline")

console = Console()

# ---------------------------------------------------------------------------
# File-content templates
# ---------------------------------------------------------------------------

GOLDENS_TEMPLATE = """\
# LongProbe Golden Questions
# --------------------------
# Define the questions your RAG pipeline MUST answer correctly.
# Each entry specifies a query plus the documents/chunks that should appear
# in the top-k results.
#
# match_mode:
#   id       — exact chunk ID must appear in results
#   text     — literal substring must appear in any returned chunk
#   semantic — cosine similarity between returned chunks and expected text

name: "my-rag-golden-set"
version: "1.0"

questions:
  - id: "q1"
    question: "What is the return policy for electronics?"
    match_mode: "id"
    required_chunks:
      - "doc-faq-returns-policy"
    top_k: 5
    tags: ["faq", "returns"]

  - id: "q2"
    question: "How long does standard shipping take?"
    match_mode: "text"
    required_chunks:
      - "Standard shipping takes 5-7 business days"
    top_k: 5
    tags: ["faq", "shipping"]

  - id: "q3"
    question: "What are the enterprise pricing tiers?"
    match_mode: "semantic"
    semantic_threshold: 0.75
    required_chunks:
      - "Enterprise plans start at $499/month and include dedicated support"
    top_k: 10
    tags: ["pricing"]
"""

CONFIG_TEMPLATE = """\
# LongProbe Configuration
# -----------------------

retriever:
  # Supported adapters: chroma, pinecone, qdrant, http
  # For langchain / llamaindex, use the Python API instead of the CLI.
  type: chroma

  # -- Chroma settings --
  collection: default
  persist_directory: ./chroma_db

  # -- Pinecone settings (uncomment if type: pinecone) --
  # index_name: my-index
  # api_key: ${PINECONE_API_KEY}
  # namespace: ""

  # -- Qdrant settings (uncomment if type: qdrant) --
  # collection: default
  # host: localhost
  # port: 6333
  # api_key: ""

  # -- HTTP adapter settings (uncomment if type: http) --
  # Use this to test any HTTP-based RAG API (LongTrainer, LangServe, etc.)
  # http:
  #   url: "http://localhost:8000/api/retrieve"
  #   method: POST
  #   body_template: '{"query": "{question}", "top_k": {top_k}}'
  #   headers:
  #     Authorization: "Bearer ${API_KEY}"
  #   response_mapping:
  #     results_path: "data.chunks"
  #     id_field: "chunk_id"
  #     text_field: "content"
  #     score_field: "similarity"
  #   timeout: 30

embedder:
  # Embedding provider: openai | huggingface | local
  provider: local
  model: text-embedding-3-small
  # For HuggingFace:
  # provider: huggingface
  # model: sentence-transformers/all-MiniLM-L6-v2

scoring:
  # Minimum recall fraction (0-1) for a question to PASS.
  recall_threshold: 0.8
  # Exit with code 1 when any question fails or regression detected.
  fail_on_regression: true

baseline:
  # Path to the SQLite database used for baseline storage.
  db_path: .longprobe/baselines.db
  # When true, `longprobe check` automatically diffs against "latest".
  auto_compare: true
"""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_golden_set(path: Path) -> GoldenSet | None:
    """Load and validate a golden-set YAML file.

    Returns ``None`` (after printing an error) when the file cannot be read
    or parsed so that callers can decide whether to abort gracefully.
    """
    if not path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Golden-set file not found: {path}",
        )
        console.print(
            "Run [bold cyan]longprobe init[/bold cyan] to create starter files.",
        )
        return None

    try:
        golden_set = GoldenSet.from_yaml(str(path))
    except Exception as exc:
        console.print(
            f"[bold red]Error:[/bold red] Failed to parse golden set: {exc}",
        )
        return None

    if not golden_set.questions:
        console.print(
            "[bold yellow]Warning:[/bold yellow] Golden set contains no questions.",
        )
        return None

    return golden_set


def _load_config(path: Path) -> ProbeConfig:
    """Load configuration from YAML, falling back to defaults.

    A warning is printed when the file does not exist so the user knows they
    are running with built-in defaults.
    """
    if not path.exists():
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Config file not found: {path}",
        )
        console.print("Using default configuration values.\n")
        return ProbeConfig.defaults()

    try:
        return ProbeConfig.from_yaml(str(path))
    except Exception as exc:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Failed to parse config ({exc}). "
            "Using defaults.\n",
        )
        return ProbeConfig.defaults()


def _create_adapter_from_config(config: ProbeConfig):
    """Instantiate the correct retriever adapter based on config.

    For ``langchain`` and ``llamaindex`` types an informative error is printed
    because they require a live retriever object that cannot be created from
    a static config alone.
    """
    rtype = getattr(config.retriever, "type", "chroma") or "chroma"

    if rtype in ("langchain", "llamaindex"):
        console.print(
            f"[bold red]Error:[/bold red] Retriever type '{rtype}' requires "
            "programmatic usage via the Python API.\n"
            "Use the LongProbe Python API directly:\n\n"
            "  from longprobe import LongProbe, LangChainRetrieverAdapter\n"
            "  probe = LongProbe(adapter=LangChainRetrieverAdapter(my_retriever), ...)\n"
            "  report = probe.run()\n",
        )
        raise typer.Exit(1)

    if rtype == "chroma":
        kwargs: dict[str, Any] = {
            "collection_name": getattr(config.retriever, "collection", "default"),
        }
        persist_dir = getattr(config.retriever, "persist_directory", None)
        if persist_dir:
            kwargs["persist_directory"] = persist_dir
        return ChromaAdapter(**kwargs)

    if rtype == "pinecone":
        return PineconeAdapter(
            index_name=getattr(config.retriever, "index_name", ""),
            api_key=getattr(config.retriever, "api_key", ""),
            namespace=getattr(config.retriever, "namespace", ""),
        )

    if rtype == "qdrant":
        return QdrantAdapter(
            collection_name=getattr(config.retriever, "collection", "default"),
            host=getattr(config.retriever, "host", "localhost"),
            port=int(getattr(config.retriever, "port", 6333)),
            api_key=getattr(config.retriever, "api_key", ""),
        )

    if rtype == "http":
        return HttpAdapter(config=config.retriever.http)

    console.print(f"[bold red]Error:[/bold red] Unknown retriever type: '{rtype}'")
    console.print("Supported types: chroma, pinecone, qdrant, http")
    raise typer.Exit(1)


def _display_results(
    report: ProbeReport,
    output_format: str,
    diff_result: dict[str, Any] | None = None,
) -> None:
    """Render a :class:`ProbeReport` in the requested output format."""
    if output_format == "json":
        _display_json(report, diff_result)
    elif output_format == "github":
        _display_github(report, diff_result)
    else:
        _display_table(report, diff_result)


def _display_json(
    report: ProbeReport,
    diff_result: dict[str, Any] | None = None,
) -> None:
    """Dump the report (and optional diff) as pretty-printed JSON."""
    payload: dict[str, Any] = {
        "golden_set": report.golden_set_name,
        "golden_set_version": report.golden_set_version,
        "timestamp": report.timestamp,
        "overall_recall": report.overall_recall,
        "pass_rate": report.pass_rate,
        "total_questions": len(report.results),
        "passed_questions": sum(1 for r in report.results if r.passed),
        "failed_questions": sum(1 for r in report.results if not r.passed),
        "regression_detected": report.regression_detected,
        "results": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "recall_score": r.recall_score,
                "required_chunks": r.required_chunks,
                "found_chunks": r.found_chunks,
                "missing_chunks": r.missing_chunks,
                "passed": r.passed,
                "latency_ms": r.latency_ms,
            }
            for r in report.results
        ],
    }
    if report.baseline_recall is not None:
        payload["baseline_recall"] = report.baseline_recall
        payload["recall_delta"] = report.recall_delta
    if diff_result is not None:
        payload["diff"] = diff_result

    console.print_json(json.dumps(payload, indent=2))


def _display_github(
    report: ProbeReport,
    diff_result: dict[str, Any] | None = None,
) -> None:
    """Print GitHub Actions workflow-command annotations."""
    for r in report.results:
        if r.passed:
            msg = (
                f"recall={r.recall_score:.0%} "
                f"({len(r.found_chunks)}/{len(r.required_chunks)} expected chunks found)"
            )
            console.print(f"::notice file=longprobe,title=Retrieval OK::[{r.question_id}] {msg}")
        else:
            missing_summary = ", ".join(r.missing_chunks) if r.missing_chunks else "\u2014"
            msg = (
                f"recall={r.recall_score:.0%} "
                f"({len(r.found_chunks)}/{len(r.required_chunks)}) "
                f"\u2014 missing: {missing_summary}"
            )
            console.print(f"::error file=longprobe,title=Retrieval Fail::[{r.question_id}] {msg}")

    if diff_result and diff_result.get("regressions"):
        for reg in diff_result["regressions"]:
            console.print(
                f"::error file=longprobe,title=Retrieval Regression::"
                f"Question {reg['question_id']}: recall dropped from "
                f"{reg['baseline_recall']:.2f} to {reg['current_recall']:.2f}. "
                f"Lost chunks: {reg['newly_lost_chunks']}"
            )

    if diff_result and diff_result.get("improvements"):
        for imp in diff_result["improvements"]:
            console.print(
                f"::notice file=longprobe,title=Retrieval Improvement::"
                f"Question {imp['question_id']}: recall improved from "
                f"{imp['baseline_recall']:.2f} to {imp['current_recall']:.2f}."
            )

    console.print(
        f"::group::Summary\n"
        f"Overall recall: {report.overall_recall:.2f}\n"
        f"Pass rate: {report.pass_rate:.2f} "
        f"({sum(1 for r in report.results if r.passed)}/{len(report.results)})\n"
        f"::endgroup::",
    )


def _display_table(
    report: ProbeReport,
    diff_result: dict[str, Any] | None = None,
) -> None:
    """Render results as a Rich table with colour-coded rows and a summary."""
    table = Table(
        title="LongProbe Results",
        show_lines=True,
        expand=True,
        header_style="bold cyan",
    )
    table.add_column("ID", style="dim", max_width=20, no_wrap=True)
    table.add_column("Question", max_width=50, no_wrap=False)
    table.add_column("Recall", justify="right", width=8)
    table.add_column("Required", justify="center", width=8)
    table.add_column("Found", justify="center", width=6)
    table.add_column("Missing", width=40, no_wrap=False)
    table.add_column("Status", justify="center", width=6)

    regression_ids: set[str] = set()
    if diff_result and diff_result.get("regressions"):
        regression_ids = {r["question_id"] for r in diff_result["regressions"]}

    for r in report.results:
        is_pass = r.passed
        is_regression = r.question_id in regression_ids

        if is_regression:
            style = "bold red"
            status = "\u26a0 \u2717"
        elif is_pass:
            style = "green"
            status = "\u2713"
        else:
            style = "red"
            status = "\u2717"

        missing_text = ", ".join(r.missing_chunks) if r.missing_chunks else "\u2014"
        question_text = r.question if len(r.question) <= 47 else r.question[:44] + "..."

        table.add_row(
            r.question_id,
            question_text,
            f"{r.recall_score:.0%}",
            str(len(r.required_chunks)),
            str(len(r.found_chunks)),
            missing_text,
            f"[{style}]{status}[/{style}]",
            style=style,
        )

    console.print()
    console.print(table)

    # --- Summary panel ---
    passed_count = sum(1 for r in report.results if r.passed)
    total_count = len(report.results)
    summary_lines: list[str] = [
        f"[bold]Overall Recall:[/bold]  {report.overall_recall:.2f}",
        f"[bold]Pass Rate:[/bold]       {report.pass_rate:.2f}  "
        f"({passed_count}/{total_count})",
    ]

    if diff_result is not None:
        reg_count = len(diff_result.get("regressions", []))
        imp_count = len(diff_result.get("improvements", []))
        if reg_count:
            summary_lines.append(
                f"[bold red]Regressions:[/bold red]     {reg_count} question(s) degraded",
            )
        else:
            summary_lines.append(
                "[bold green]Regressions:[/bold green]     None detected",
            )
        if imp_count:
            summary_lines.append(
                f"[bold green]Improvements:[/bold green]   {imp_count} question(s) improved",
            )

    if report.baseline_recall is not None and report.recall_delta is not None:
        delta_str = f"{report.recall_delta:+.2f}"
        delta_style = "red" if report.recall_delta < 0 else "green"
        summary_lines.append(
            f"[bold]vs Baseline:[/bold]       [{delta_style}]{delta_str}[/{delta_style}]"
        )

    border_style = "green" if report.pass_rate >= 1.0 else "red"
    console.print(
        Panel(
            "\n".join(summary_lines),
            title="Summary",
            border_style=border_style,
        ),
    )


def _run_probe(
    goldens_path: Path,
    config_path: Path,
    top_k_override: int | None = None,
    threshold_override: float | None = None,
    tags: list[str] | None = None,
) -> tuple[ProbeConfig, Any, ProbeReport]:
    """Shared logic used by ``check``, ``baseline save``, ``diff``, and ``watch``.

    Returns ``(config, adapter, report)``.
    """
    # 1. Load golden set
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading golden set...", total=None)
        golden_set = _load_golden_set(goldens_path)
        if golden_set is None:
            raise typer.Exit(1)

        if tags:
            golden_set = golden_set.filter_by_tags(tags)
            if not golden_set.questions:
                console.print(f"[bold yellow]Warning:[/bold yellow] No questions match the provided tags: {tags}")
                raise typer.Exit(0)

        # 2. Load config
        progress.add_task("Loading config...", total=None)
        config = _load_config(config_path)

        # 3. Create adapter
        progress.add_task("Initialising retriever adapter...", total=None)
        adapter = _create_adapter_from_config(config)

    # 4. Create scorer
    effective_threshold = (
        threshold_override
        if threshold_override is not None
        else config.scoring.recall_threshold
    )
    scorer = RecallScorer(recall_threshold=effective_threshold)

    # 5. Run probes
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Running probes...", total=None)
        report = scorer.score_all(
            golden_set,
            adapter.retrieve,
            top_k_override=top_k_override,
        )

    # 6. Auto-compare against baseline
    if config.baseline.auto_compare:
        db_path = config.baseline.db_path
        store = BaselineStore(db_path=db_path)
        baseline = store.load("latest")
        if baseline is not None:
            report.baseline_recall = baseline.overall_recall
            report.recall_delta = report.overall_recall - baseline.overall_recall
            diff_result = store.diff(report, baseline)
            report.regression_detected = len(diff_result["regressions"]) > 0

    return config, adapter, report


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def init(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing files without prompting.",
    ),
) -> None:
    """Create starter configuration files in the current directory.

    Creates ``goldens.yaml`` with example questions and ``longprobe.yaml``
    with sensible defaults.
    """
    goldens_path = Path("goldens.yaml")
    config_path = Path("longprobe.yaml")
    longprobe_dir = Path(".longprobe")

    # Create .longprobe directory
    if longprobe_dir.exists():
        console.print(
            f"[dim]Directory [bold]{longprobe_dir}[/bold] already exists \u2014 "
            "skipping.[/dim]",
        )
    else:
        longprobe_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]Created [bold]{longprobe_dir}/[/bold][/dim]")

    # Write goldens.yaml
    if goldens_path.exists() and not force:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] {goldens_path} already exists. "
            "Use --force to overwrite.",
        )
    else:
        goldens_path.write_text(GOLDENS_TEMPLATE, encoding="utf-8")
        console.print(f"[green]Created [bold]{goldens_path}[/bold][/green]")

    # Write longprobe.yaml
    if config_path.exists() and not force:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] {config_path} already exists. "
            "Use --force to overwrite.",
        )
    else:
        config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        console.print(f"[green]Created [bold]{config_path}[/bold][/green]")

    console.print()
    console.print(
        Panel(
            "[bold green]LongProbe initialised successfully![/bold green]\n\n"
            "Next steps:\n"
            "  1. Edit [cyan]longprobe.yaml[/cyan] with your retriever settings\n"
            "  2. Add your golden questions to [cyan]goldens.yaml[/cyan]\n"
            "  3. Run [cyan]longprobe check[/cyan] to test retrieval quality",
            title="Ready to go",
            border_style="green",
        ),
    )


@app.command()
def check(
    goldens: Path = typer.Option(
        "goldens.yaml",
        "--goldens",
        "-g",
        help="Path to golden questions YAML.",
    ),
    config: Path = typer.Option(
        "longprobe.yaml",
        "--config",
        "-c",
        help="Path to config YAML.",
    ),
    output: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format: table, json, github",
    ),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        "-k",
        help="Override top_k for all questions.",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Override recall threshold (0.0\u20131.0).",
    ),
    tag: list[str] = typer.Option(
        [],
        "--tag",
        help="Only run questions that have this tag (can be specified multiple times).",
    ),
) -> None:
    """Run probes against the golden set and report retrieval quality."""
    cfg, _adapter, report = _run_probe(
        goldens_path=goldens,
        config_path=config,
        top_k_override=top_k,
        threshold_override=threshold,
        tags=tag,
    )

    # Build diff_result for display if regression was detected
    diff_result = None
    if report.regression_detected and cfg.baseline.auto_compare:
        store = BaselineStore(db_path=cfg.baseline.db_path)
        baseline = store.load("latest")
        if baseline is not None:
            diff_result = store.diff(report, baseline)

    _display_results(report, output, diff_result)

    # Determine exit code
    fail_on_regression = cfg.scoring.fail_on_regression
    regression_detected = report.regression_detected
    if fail_on_regression and (report.pass_rate < 1.0 or regression_detected):
        sys.exit(1)


# ---------------------------------------------------------------------------
# Baseline sub-commands
# ---------------------------------------------------------------------------


@baseline_app.command("save")
def baseline_save(
    label: str = typer.Option(
        "latest",
        "--label",
        "-l",
        help="Baseline label (e.g. v1.2, commit-sha).",
    ),
    goldens: Path = typer.Option(
        "goldens.yaml",
        "--goldens",
        "-g",
        help="Path to golden questions YAML.",
    ),
    config: Path = typer.Option(
        "longprobe.yaml",
        "--config",
        "-c",
        help="Path to config YAML.",
    ),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        "-k",
        help="Override top_k for all questions.",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Override recall threshold (0.0\u20131.0).",
    ),
    tag: list[str] = typer.Option(
        [],
        "--tag",
        help="Only run questions that have this tag (can be specified multiple times).",
    ),
) -> None:
    """Run probes and persist the report as a named baseline snapshot."""
    cfg, _adapter, report = _run_probe(
        goldens_path=goldens,
        config_path=config,
        top_k_override=top_k,
        threshold_override=threshold,
        tags=tag,
    )

    store = BaselineStore(db_path=cfg.baseline.db_path)
    store.save(report=report, label=label)

    console.print(
        Panel(
            f"Baseline [bold cyan]{label}[/bold cyan] saved successfully.\n"
            f"  Questions: {len(report.results)}\n"
            f"  Overall recall: {report.overall_recall:.2f}\n"
            f"  Pass rate: {report.pass_rate:.2f}\n"
            f"  Location: {cfg.baseline.db_path}",
            title="Baseline Saved",
            border_style="green",
        ),
    )


@baseline_app.command("list")
def baseline_list(
    db_path: Path = typer.Option(
        ".longprobe/baselines.db",
        "--db-path",
        help="Path to the baseline database.",
    ),
) -> None:
    """List all saved baseline snapshots."""
    store = BaselineStore(db_path=str(db_path))
    baselines = store.list_labels()

    if not baselines:
        console.print(
            "[bold yellow]No baselines found.[/bold yellow]\n"
            "Run [cyan]longprobe baseline save[/cyan] to create one.",
        )
        return

    table = Table(
        title="Saved Baselines",
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("Label", style="bold cyan", no_wrap=True)
    table.add_column("Golden Set", max_width=30)
    table.add_column("Version", width=10)
    table.add_column("Timestamp", no_wrap=True)
    table.add_column("Created", no_wrap=True)

    for bl in baselines:
        table.add_row(
            bl.get("label", "\u2014"),
            bl.get("golden_set_name", "\u2014"),
            bl.get("golden_set_version", "\u2014"),
            bl.get("timestamp", "\u2014"),
            bl.get("created_at", "\u2014"),
        )

    console.print()
    console.print(table)


@baseline_app.command("delete")
def baseline_delete(
    label: str = typer.Option(
        ...,
        "--label",
        "-l",
        help="Baseline label to delete.",
    ),
    db_path: Path = typer.Option(
        ".longprobe/baselines.db",
        "--db-path",
        help="Path to the baseline database.",
    ),
) -> None:
    """Delete a saved baseline snapshot."""
    store = BaselineStore(db_path=str(db_path))

    # Check if baseline exists by trying to load it
    baseline = store.load(label)
    if baseline is None:
        console.print(f"[bold red]Error:[/bold red] Baseline '{label}' not found.")
        raise typer.Exit(1)

    confirm = typer.confirm(f"Delete baseline '{label}'?")
    if not confirm:
        console.print("[dim]Cancelled.[/dim]")
        raise typer.Exit(0)

    deleted = store.delete(label)
    if deleted:
        console.print(f"[green]Deleted baseline [bold]{label}[/bold].[/green]")
    else:
        console.print(f"[bold red]Error:[/bold red] Could not delete baseline '{label}'.")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Diff command
# ---------------------------------------------------------------------------


@app.command()
def diff(
    baseline_label: str = typer.Option(
        "latest",
        "--baseline",
        "-b",
        help="Baseline label to compare against.",
    ),
    goldens: Path = typer.Option(
        "goldens.yaml",
        "--goldens",
        "-g",
        help="Path to golden questions YAML.",
    ),
    config: Path = typer.Option(
        "longprobe.yaml",
        "--config",
        "-c",
        help="Path to config YAML.",
    ),
    output: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format: table, json, github.",
    ),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        "-k",
        help="Override top_k for all questions.",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Override recall threshold (0.0\u20131.0).",
    ),
    tag: list[str] = typer.Option(
        [],
        "--tag",
        help="Only run questions that have this tag (can be specified multiple times).",
    ),
) -> None:
    """Compare current probe results against a saved baseline."""
    cfg, _adapter, report = _run_probe(
        goldens_path=goldens,
        config_path=config,
        top_k_override=top_k,
        threshold_override=threshold,
        tags=tag,
    )

    store = BaselineStore(db_path=cfg.baseline.db_path)
    baseline = store.load(baseline_label)

    if baseline is None:
        console.print(
            f"[bold red]Error:[/bold red] Baseline '{baseline_label}' not found.\n"
            f"Run [cyan]longprobe baseline save --label {baseline_label}[/cyan] "
            "to create it.",
        )
        raise typer.Exit(1)

    diff_reporter = DiffReporter()
    diff_result = diff_reporter.diff(current=report, baseline=baseline)
    diff_dict = asdict(diff_result)

    _display_results(report, output, diff_dict)

    # Exit with error code if regressions found
    if diff_result.regressions:
        console.print(
            f"\n[bold red]{len(diff_result.regressions)} regression(s) detected against "
            f"baseline '{baseline_label}'.[/bold red]",
        )
        sys.exit(1)
    else:
        console.print(
            f"\n[bold green]No regressions detected against "
            f"baseline '{baseline_label}'.[/bold green]",
        )


# ---------------------------------------------------------------------------
# Watch command
# ---------------------------------------------------------------------------


@app.command()
def watch(
    goldens: Path = typer.Option(
        "goldens.yaml",
        "--goldens",
        "-g",
        help="Path to golden questions YAML.",
    ),
    config: Path = typer.Option(
        "longprobe.yaml",
        "--config",
        "-c",
        help="Path to config YAML.",
    ),
    interval: float = typer.Option(
        2.0,
        "--interval",
        "-i",
        help="Poll interval in seconds.",
    ),
    tag: list[str] = typer.Option(
        [],
        "--tag",
        help="Only run questions that have this tag (can be specified multiple times).",
    ),
) -> None:
    """Watch the golden-set file and re-run probes on every change.

    Press Ctrl+C to stop.
    """
    goldens_path = Path(goldens)
    config_path = Path(config)

    if not goldens_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Golden-set file not found: {goldens_path}",
        )
        raise typer.Exit(1)

    last_mtime: float | None = goldens_path.stat().st_mtime
    run_count = 0

    console.print(
        Panel(
            f"Watching [bold cyan]{goldens_path}[/bold cyan] for changes "
            f"(interval: {interval}s)\n"
            "Press [bold]Ctrl+C[/bold] to stop.",
            title="LongProbe Watch",
            border_style="cyan",
        ),
    )

    try:
        while True:
            current_mtime = goldens_path.stat().st_mtime
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                run_count += 1

                console.clear()
                console.print(
                    f"[bold cyan]Run #{run_count}[/bold cyan] "
                    f"\u2014 {datetime.now().strftime('%H:%M:%S')}\n",
                )

                try:
                    cfg, _adapter, report = _run_probe(
                        goldens_path=goldens_path,
                        config_path=config_path,
                        tags=tag,
                    )

                    diff_result = None
                    if report.regression_detected:
                        store = BaselineStore(db_path=cfg.baseline.db_path)
                        baseline = store.load("latest")
                        if baseline is not None:
                            diff_reporter = DiffReporter()
                            diff_obj = diff_reporter.diff(current=report, baseline=baseline)
                            diff_result = asdict(diff_obj)

                    _display_results(report, "table", diff_result)
                except typer.Exit:
                    # Errors already printed by _run_probe
                    pass
                except Exception as exc:
                    console.print(
                        f"[bold red]Error during run:[/bold red] {exc}",
                    )

            time.sleep(interval)
    except KeyboardInterrupt:
        console.print(
            f"\n[bold cyan]LongProbe watch stopped.[/bold cyan] "
            f"(Ran {run_count} time(s))",
        )


# ---------------------------------------------------------------------------
# Capture command
# ---------------------------------------------------------------------------


@app.command()
def capture(
    question: list[str] = typer.Option(
        [],
        "--question",
        "-q",
        help="A specific question to capture (can be specified multiple times).",
    ),
    questions_file: Path | None = typer.Option(
        None,
        "--questions-file",
        "-Q",
        help="Path to a text file with one question per line.",
    ),
    auto: bool = typer.Option(
        False,
        "--auto",
        help="Trust the retriever and automatically save all results without prompting.",
    ),
    match_mode: str = typer.Option(
        "text",
        "--match-mode",
        "-m",
        help="Match mode for generated questions (id, text, semantic).",
    ),
    tag: list[str] = typer.Option(
        [],
        "--tag",
        help="Tag to apply to the captured questions (can be specified multiple times).",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of chunks to retrieve per question.",
    ),
    goldens: Path = typer.Option(
        "goldens.yaml",
        "--goldens",
        "-g",
        help="Path to golden questions YAML (will append if exists).",
    ),
    config: Path = typer.Option(
        "longprobe.yaml",
        "--config",
        "-c",
        help="Path to config YAML.",
    ),
    id_prefix: str = typer.Option(
        "q",
        "--id-prefix",
        help="Prefix to use for auto-generated question IDs.",
    ),
) -> None:
    """Capture retriever results and build a golden set."""
    # 1. Collect questions
    questions_to_ask = list(question)
    if questions_file and questions_file.exists():
        text = questions_file.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        questions_to_ask.extend(lines)

    if not questions_to_ask:
        console.print("[bold red]Error:[/bold red] No questions provided. Use --question or --questions-file.")
        raise typer.Exit(1)

    # 2. Load existing golden set or create empty
    if goldens.exists():
        golden_set = _load_golden_set(goldens)
        if golden_set is None:
            raise typer.Exit(1)
        console.print(f"[dim]Loaded existing golden set '{goldens}'[/dim]")
    else:
        name = typer.prompt("Name for this new golden set", default="my-golden-set")
        version = typer.prompt("Version", default="1.0")
        golden_set = GoldenSet(name=name, version=version, questions=[])

    # 3. Load config and adapter
    cfg = _load_config(config)
    adapter = _create_adapter_from_config(cfg)

    existing_ids = {q.id for q in golden_set.questions}
    new_questions: list[GoldenQuestion] = []

    # 4. Iterate over questions
    for q_text in questions_to_ask:
        console.print(f"\n[bold cyan]🔍 Querying retriever for:[/bold cyan] [italic]\"{q_text}\"[/italic]")
        start = time.perf_counter()
        results = adapter.retrieve(q_text, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000

        console.print(f"   [dim]Retrieved {len(results)} chunks in {elapsed:.0f}ms[/dim]\n")

        if not results:
            console.print("   [yellow]No results found. Skipping...[/yellow]")
            continue

        approved_chunks: list[str] = []

        if auto:
            # Auto mode: accept all chunks
            for r in results:
                if match_mode == "id":
                    approved_chunks.append(r.get("id", ""))
                else:
                    approved_chunks.append(r.get("text", ""))
        else:
            # Interactive mode
            skip_question = False
            for idx, r in enumerate(results, 1):
                chunk_id = r.get("id", "unknown")
                chunk_text = r.get("text", "")

                # Truncate text for display
                display_text = chunk_text
                if len(display_text) > 200:
                    display_text = display_text[:200] + "..."

                content = f"[bold]Chunk {idx}: {chunk_id}[/bold]\n\n[dim]{display_text}[/dim]"
                console.print(Panel(content, border_style="blue"))

                ans = Prompt.ask(
                    "   ✅ Include this chunk?",
                    choices=["y", "n", "s", "q"],
                    default="y",
                    show_choices=True
                )

                if ans == "y":
                    if match_mode == "id":
                        approved_chunks.append(chunk_id)
                    else:
                        approved_chunks.append(chunk_text)
                elif ans == "s":
                    skip_question = True
                    break
                elif ans == "q":
                    console.print("[dim]Aborting capture process...[/dim]")
                    # Save what we have so far
                    if new_questions:
                        added = golden_set.merge(new_questions)
                        golden_set.to_yaml(str(goldens))
                        console.print(f"\n[green]Saved {added} question(s) to {goldens}[/green]")
                    raise typer.Exit(0)

            if skip_question:
                console.print("   [yellow]Skipping question...[/yellow]")
                continue

        if not approved_chunks:
            console.print("   [yellow]No chunks approved. Skipping question...[/yellow]")
            continue

        # Generate ID and save
        q_id = generate_question_id(q_text, prefix=id_prefix, existing_ids=existing_ids)
        existing_ids.add(q_id)

        golden_q = GoldenQuestion(
            id=q_id,
            question=q_text,
            match_mode=match_mode,
            required_chunks=approved_chunks,
            tags=list(tag),
            top_k=top_k,
        )
        new_questions.append(golden_q)
        console.print(f"   [green]✅ Saved question '{q_id}' with {len(approved_chunks)} required chunks.[/green]")

    # 5. Merge and save
    if new_questions:
        added = golden_set.merge(new_questions)
        golden_set.to_yaml(str(goldens))
        console.print(f"\n[bold green]Successfully added {added} new question(s) to {goldens}[/bold green]")
    else:
        console.print("\n[dim]No new questions were captured.[/dim]")


# ---------------------------------------------------------------------------
# Generate command
# ---------------------------------------------------------------------------


@app.command()
def generate(
    path: Path = typer.Argument(
        ...,
        help="File or directory containing documents to generate questions from.",
    ),
    num_questions: int = typer.Option(
        0,
        "--num-questions",
        "-n",
        help="Number of questions to generate (0 = use config default).",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output YAML file path. Prints to stdout if not set.",
    ),
    provider: str = typer.Option(
        "",
        "--provider",
        help="LLM provider (overrides config). E.g. openai, anthropic, gemini, ollama.",
    ),
    model: str = typer.Option(
        "",
        "--model",
        help="LLM model (overrides config). E.g. gpt-4o-mini, claude-3-haiku-20240307.",
    ),
    config: Path = typer.Option(
        "longprobe.yaml",
        "--config",
        "-c",
        help="Path to config YAML.",
    ),
    auto_capture: bool = typer.Option(
        False,
        "--auto-capture",
        help="After generating questions, send each through the retriever and save as a golden set.",
    ),
    match_mode: str = typer.Option(
        "text",
        "--match-mode",
        "-m",
        help="Match mode for auto-captured chunks: id, text, semantic.",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Top-k for auto-capture retriever queries.",
    ),
    goldens: Path = typer.Option(
        "goldens.yaml",
        "--goldens",
        "-g",
        help="Golden set file path for auto-capture.",
    ),
    tag: list[str] = typer.Option(
        [],
        "--tag",
        help="Tag to apply to auto-captured questions (can be specified multiple times).",
    ),
    id_prefix: str = typer.Option(
        "q",
        "--id-prefix",
        help="Prefix for auto-generated question IDs.",
    ),
) -> None:
    """Generate golden questions from documents using an LLM."""
    # 1. Load config
    cfg = _load_config(config)

    # Apply CLI overrides.
    gen_config = cfg.generator
    if provider:
        gen_config = GeneratorConfig(
            provider=provider,
            model=model or gen_config.model,
            api_key=gen_config.api_key,
            base_url=gen_config.base_url,
            num_questions=gen_config.num_questions,
            temperature=gen_config.temperature,
            max_tokens=gen_config.max_tokens,
        )
    elif model:
        gen_config = GeneratorConfig(
            provider=gen_config.provider,
            model=model,
            api_key=gen_config.api_key,
            base_url=gen_config.base_url,
            num_questions=gen_config.num_questions,
            temperature=gen_config.temperature,
            max_tokens=gen_config.max_tokens,
        )

    effective_n = num_questions if num_questions > 0 else gen_config.num_questions

    # 2. Check API key early.
    if not gen_config.api_key:
        # Try common env vars based on provider.
        import os
        env_hints = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        env_var = env_hints.get(gen_config.provider, f"{gen_config.provider.upper()}_API_KEY")
        if not os.environ.get(env_var):
            console.print(
                f"[bold red]Error:[/bold red] No API key configured for "
                f"{gen_config.provider}.\n"
                f"Set the environment variable [cyan]{env_var}[/cyan] or add "
                f"[cyan]generator.api_key[/cyan] to [bold]{config}[/bold]."
            )
            raise typer.Exit(1)

    # 3. Parse documents.
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Reading documents...", total=None)
        parser = DocumentParser()
        documents = parser.parse_path(str(path))

    if not documents:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] No extractable text found in "
            f"'{path}'. Ensure the path contains supported file types."
        )
        raise typer.Exit(0)

    console.print(
        f"[dim]Loaded {len(documents)} document(s) with "
        f"{sum(len(t) for _, t in documents):,} characters.[/dim]"
    )

    # 4. Generate questions.
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(
                f"Generating {effective_n} questions via {gen_config.provider}/{gen_config.model}...",
                total=None,
            )
            generator = QuestionGenerator(gen_config)
            questions = generator.generate(documents, num_questions=effective_n)
    except ImportError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1)
    except RuntimeError as exc:
        console.print(f"[bold red]LLM Error:[/bold red] {exc}")
        raise typer.Exit(1)

    if not questions:
        console.print(
            "[bold yellow]Warning:[/bold yellow] LLM did not generate any questions."
        )
        raise typer.Exit(0)

    # 5. Output or auto-capture.
    if auto_capture:
        # --- Auto-capture mode ---
        # 5a. Create adapter from config.
        rtype = getattr(cfg.retriever, "type", "") or ""
        if not rtype or rtype in ("langchain", "llamaindex"):
            console.print(
                "[bold red]Error:[/bold red] Auto-capture requires a retriever "
                "configured in longprobe.yaml.\n"
                "Set [cyan]retriever.type[/cyan] to http, chroma, pinecone, or qdrant."
            )
            raise typer.Exit(1)

        try:
            adapter = _create_adapter_from_config(cfg)
        except Exception as exc:
            console.print(
                f"[bold red]Error:[/bold red] Failed to create retriever adapter: {exc}"
            )
            raise typer.Exit(1)

        # 5b. Load existing golden set or create empty.
        if goldens.exists():
            try:
                golden_set = GoldenSet.from_yaml(str(goldens))
                console.print(f"[dim]Loaded existing golden set '{goldens}'[/dim]")
            except Exception:
                golden_set = GoldenSet(
                    name="auto-generated",
                    version="1.0",
                    questions=[],
                )
                console.print(
                    f"[dim]Could not parse existing '{goldens}', creating new golden set.[/dim]"
                )
        else:
            golden_set = GoldenSet(
                name="auto-generated",
                version="1.0",
                questions=[],
            )

        existing_ids = {q.id for q in golden_set.questions}
        new_questions: list[GoldenQuestion] = []
        skipped = 0

        # 5c. Query retriever for each question.
        console.print(
            f"\n[bold cyan]Auto-capturing {len(questions)} questions via "
            f"{rtype} adapter...[/bold cyan]"
        )

        for q_text in questions:
            console.print(
                f"  [dim]\U0001f50d[/dim] [italic]\"{q_text[:60]}{'...' if len(q_text) > 60 else ''}\"[/italic]"
            )
            try:
                start = time.perf_counter()
                results = adapter.retrieve(q_text, top_k=top_k)
                elapsed = (time.perf_counter() - start) * 1000
            except Exception as exc:
                console.print(f"    [yellow]Retriever error: {exc}. Skipping.[/yellow]")
                skipped += 1
                continue

            console.print(
                f"    [dim]Retrieved {len(results)} chunks in {elapsed:.0f}ms[/dim]"
            )

            if not results:
                console.print("    [yellow]No results. Skipping.[/yellow]")
                skipped += 1
                continue

            # Build required_chunks list.
            approved_chunks: list[str] = []
            for r in results:
                if match_mode == "id":
                    approved_chunks.append(r.get("id", ""))
                else:
                    approved_chunks.append(r.get("text", ""))

            if not approved_chunks:
                skipped += 1
                continue

            q_id = generate_question_id(q_text, prefix=id_prefix, existing_ids=existing_ids)
            existing_ids.add(q_id)

            golden_q = GoldenQuestion(
                id=q_id,
                question=q_text,
                match_mode=match_mode,
                required_chunks=approved_chunks,
                tags=list(tag),
                top_k=top_k,
            )
            new_questions.append(golden_q)

        # 5d. Merge and save.
        if new_questions:
            _added = golden_set.merge(new_questions)
            try:
                golden_set.to_yaml(str(goldens))
            except OSError as exc:
                console.print(
                    f"[bold red]Error:[/bold red] Failed to write golden set: {exc}"
                )
                raise typer.Exit(1)

        captured_count = len(new_questions)
        console.print(
            Panel(
                f"Generated:   [bold cyan]{len(questions)}[/bold cyan] questions\n"
                f"Captured:    [bold green]{captured_count}[/bold green] questions\n"
                f"Skipped:     [bold yellow]{skipped}[/bold yellow] (no results / errors)\n"
                f"Saved to:    [bold]{goldens}[/bold]",
                title="Auto-Capture Complete",
                border_style="green" if captured_count > 0 else "yellow",
            )
        )
    elif output is not None:
        # --- Write questions to YAML file ---
        import yaml as _yaml

        data = {
            "questions": questions,
            "source": str(path),
            "count": len(questions),
            "generator": {
                "provider": gen_config.provider,
                "model": gen_config.model,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            _yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        console.print(
            Panel(
                f"Generated [bold cyan]{len(questions)}[/bold cyan] questions.\n"
                f"Saved to [bold]{output}[/bold]",
                title="Generation Complete",
                border_style="green",
            )
        )
    else:
        # Print to stdout, one per line.
        for q in questions:
            console.print(q)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
