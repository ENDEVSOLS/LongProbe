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

import sys
import time
from dataclasses import asdict
from datetime import datetime
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from longprobe._files import atomic_write_text, check_destination
from longprobe.adapters import create_adapter_from_config
from longprobe.cli.output import (
    github_annotations,
    report_exit_code,
    report_payload,
    validate_output,
    write_json,
)
from longprobe.config import ConfigurationError, GeneratorConfig, ProbeConfig
from longprobe.core.baseline import BaselineStore
from longprobe.core.diff import DiffReporter
from longprobe.core.docparser import DocumentParser
from longprobe.core.evaluation import BaselineNotFoundError, evaluate, select_questions
from longprobe.core.explain import explain_failure
from longprobe.core.generator import QuestionGenerator
from longprobe.core.golden import GoldenQuestion, GoldenSet, generate_question_id
from longprobe.core.policy import BaselineQualityError, assess_baseline
from longprobe.core.reporting import ReportError, read_report
from longprobe.core.scorer import EmptyEvaluationError, ProbeReport

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
diagnostics = Console(stderr=True)

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
  # Optional explicit per-question thresholds take precedence:
  # question_thresholds:
  #   q1: 1.0
  # Exit with code 1 when any question fails or regression detected.
  fail_on_regression: true

baseline:
  # Path to the SQLite database used for baseline storage.
  db_path: .longprobe/baselines.db
  # When true, `longprobe check` automatically diffs against "latest".
  auto_compare: true
  # Default promotion requires every evaluated approved question to pass.
  quality_gate: all_questions
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
    except (OSError, ValueError):
        console.print(
            f"Cannot load golden set {path}; repair the file or select a new path.", markup=False
        )
        return None

    return golden_set


def _load_config(path: Path) -> ProbeConfig:
    """Require the selected CLI configuration; never silently substitute defaults."""
    return ProbeConfig.from_yaml(str(path))


def _command_errors(fn):
    """Return one safe error object for parsed JSON-mode invocations."""

    @wraps(fn)
    def wrapped(*args, **kwargs):
        output = signature(fn).bind_partial(*args, **kwargs).arguments.get("output", "table")
        try:
            return fn(*args, **kwargs)
        except (typer.Exit, SystemExit):
            raise
        except Exception as exc:
            if isinstance(exc, BaselineQualityError):
                code, kind, message = 1, "quality_failure", str(exc)
            elif isinstance(
                exc, (ConfigurationError, EmptyEvaluationError, BaselineNotFoundError, ReportError)
            ):
                code, kind, message = 2, "invalid_input", str(exc)
            elif isinstance(exc, FileNotFoundError):
                code, kind, message = (
                    2,
                    "missing_input",
                    "Selected input was not found; check paths or run longprobe init.",
                )
            elif isinstance(exc, ValueError):
                code, kind, message = (
                    2,
                    "invalid_input",
                    "Invalid input; check golden questions and command options.",
                )
            elif isinstance(exc, ImportError):
                code, kind, message = (
                    2,
                    "missing_dependency",
                    "Required optional dependency is missing; install the selected adapter extra.",
                )
            else:
                code, kind, message = (
                    3,
                    "execution_error",
                    "Operation failed; check file permissions and retrieval availability.",
                )
            if output == "json":
                write_json(
                    {
                        "schema_version": 2,
                        "status": "error",
                        "error": {"code": kind, "message": message},
                    }
                )
            diagnostics.print(message, markup=False)
            raise typer.Exit(code) from None

    return wrapped


def _create_adapter_from_config(config: ProbeConfig):
    """Compatibility seam for CLI callers and tests; construction is shared."""
    return create_adapter_from_config(config)


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


def _display_json(report: ProbeReport, diff_result: dict[str, Any] | None = None) -> None:
    write_json(report_payload(report))


def _display_github(report: ProbeReport, diff_result: dict[str, Any] | None = None) -> None:
    sys.stdout.write(github_annotations(report) + "\n")


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
        f"[bold]Pass Rate:[/bold]       {report.pass_rate:.2f}  ({passed_count}/{total_count})",
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

    identity = report.baseline_identity
    summary_lines.append(
        f"Baseline: {identity.get('resolved_label') or identity.get('requested_label') or 'none'}; comparison: {identity.get('status', 'not_requested')}"
    )
    if identity.get("timestamp"):
        summary_lines.append(f"Baseline timestamp: {identity['timestamp']}")
    if diff_result:
        if not diff_result.get("compatible", False):
            summary_lines.append(
                "Incompatible: " + "; ".join(diff_result.get("compatibility_reasons", []))
            )
        for label, key in (
            ("Added questions", "added_questions"),
            ("Removed questions", "removed_questions"),
        ):
            if diff_result.get(key):
                summary_lines.append(label + ": " + ", ".join(diff_result[key]))
        for change in diff_result.get("changed_questions", []):
            summary_lines.append(f"Changed {change['question_id']}: {', '.join(change['fields'])}")

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
    include_drafts: bool = False,
    *,
    baseline_label: str | None = None,
    compare: bool = True,
) -> tuple[ProbeConfig, Any, ProbeReport]:
    """Adapt CLI input to the shared evaluation module."""
    golden_set = GoldenSet.from_yaml(str(goldens_path))
    select_questions(golden_set, ProbeConfig.defaults(), tags=tags, include_drafts=include_drafts)
    if not include_drafts:
        drafts = sum(q.status == "draft" for q in golden_set.questions)
        if drafts:
            diagnostics.print(
                f"Notice: {drafts} draft question(s) excluded. Run longprobe edit to review and approve."
            )
    config = _load_config(config_path)
    if threshold_override is not None:
        config.scoring.recall_threshold = threshold_override
    config.validate_for("evaluate")
    select_questions(golden_set, config, tags=tags, include_drafts=include_drafts)
    if top_k_override is not None and (type(top_k_override) is not int or top_k_override <= 0):
        raise ConfigurationError("top_k must be a positive integer.")
    adapter = _create_adapter_from_config(config)
    store = BaselineStore(config.baseline.db_path)
    report = evaluate(
        golden_set,
        adapter.retrieve,
        config,
        store=store,
        tags=tags,
        include_drafts=include_drafts,
        top_k_override=top_k_override,
        baseline_label=baseline_label,
        compare=compare,
    )
    return config, adapter, report


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
@_command_errors
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
            f"[dim]Directory [bold]{longprobe_dir}[/bold] already exists \u2014 skipping.[/dim]",
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
        atomic_write_text(goldens_path, GOLDENS_TEMPLATE, overwrite=force)
        console.print(f"[green]Created [bold]{goldens_path}[/bold][/green]")

    # Write longprobe.yaml
    if config_path.exists() and not force:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] {config_path} already exists. "
            "Use --force to overwrite.",
        )
    else:
        atomic_write_text(config_path, CONFIG_TEMPLATE, overwrite=force)
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
@_command_errors
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
        help="Override recall threshold (0.0-1.0).",
    ),
    tag: list[str] = typer.Option(
        [],
        "--tag",
        help="Only run questions that have this tag (can be specified multiple times).",
    ),
    include_drafts: bool = typer.Option(
        False,
        "--include-drafts",
        help="Include draft questions in the evaluation run.",
    ),
    baseline_label: Annotated[
        str | None, typer.Option("--baseline", "-b", help="Explicit baseline label; must exist.")
    ] = None,
    compare: Annotated[
        bool, typer.Option("--compare/--no-compare", help="Enable configured baseline comparison.")
    ] = True,
) -> None:
    """Run probes against the golden set and report retrieval quality."""
    validate_output(output)
    if baseline_label is not None and not compare:
        raise ConfigurationError("--baseline and --no-compare cannot be combined.")
    cfg, _adapter, report = _run_probe(
        goldens_path=goldens,
        config_path=config,
        top_k_override=top_k,
        threshold_override=threshold,
        tags=tag,
        include_drafts=include_drafts,
        baseline_label=baseline_label,
        compare=compare,
    )
    _display_results(report, output, report.comparison)
    code = report_exit_code(report, blocking=cfg.scoring.fail_on_regression)
    if code:
        sys.exit(code)


# ---------------------------------------------------------------------------
# Baseline sub-commands
# ---------------------------------------------------------------------------


@baseline_app.command("save")
@_command_errors
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
        help="Override recall threshold (0.0-1.0).",
    ),
    tag: list[str] = typer.Option(
        [],
        "--tag",
        help="Only run questions that have this tag (can be specified multiple times).",
    ),
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output format: table, json, github.")
    ] = "table",
) -> None:
    """Run probes strictly on approved questions and persist the report as a named baseline snapshot."""
    validate_output(output)
    if not label.strip():
        raise ConfigurationError("Baseline label must be nonempty.")
    cfg, _adapter, report = _run_probe(
        goldens_path=goldens,
        config_path=config,
        top_k_override=top_k,
        threshold_override=threshold,
        tags=tag,
        include_drafts=False,
        compare=False,
    )
    decision = assess_baseline(report)
    report.quality = asdict(decision)
    if decision.passed:
        BaselineStore(cfg.baseline.db_path).save(report, label)
        report.quality["saved_label"] = label
        diagnostics.print(f"Baseline '{label}' saved explicitly.", markup=False)
    else:
        diagnostics.print("Baseline was NOT saved: " + "; ".join(decision.reasons), markup=False)
    _display_results(report, output)
    if not decision.passed:
        raise typer.Exit(1)


@baseline_app.command("list")
@_command_errors
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
@_command_errors
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
@_command_errors
def diff(
    baseline_a: str | None = typer.Argument(
        None,
        help="First baseline label (or target baseline if comparing two stored snapshots).",
    ),
    baseline_b: str | None = typer.Argument(
        None,
        help="Second baseline label to compare against first baseline.",
    ),
    baseline_label: str = typer.Option(
        "latest",
        "--baseline",
        "-b",
        help="Baseline label to compare against live probe run.",
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
        help="Override recall threshold (0.0-1.0).",
    ),
    tag: list[str] = typer.Option(
        [],
        "--tag",
        help="Only run questions that have this tag (can be specified multiple times).",
    ),
    include_drafts: bool = typer.Option(
        False,
        "--include-drafts",
        help="Include draft questions (live run mode only).",
    ),
) -> None:
    """Compare probe results against a saved baseline, or compare two historical snapshots."""
    validate_output(output)
    cfg = _load_config(config)
    cfg.validate_for("baseline")
    if threshold is not None:
        cfg.scoring.recall_threshold = threshold
        cfg.validate_for("baseline")
    store = BaselineStore(db_path=cfg.baseline.db_path)
    if baseline_a is not None and baseline_b is not None:
        b1, b2 = store.load(baseline_a), store.load(baseline_b)
        if b1 is None or b2 is None:
            raise BaselineNotFoundError("Requested baseline was not found; check explicit labels.")
        comparison = DiffReporter().diff(b2, b1)
        report = b2
        report.comparison = asdict(comparison)
        report.baseline_identity = {
            "status": "compared" if comparison.compatible else "incompatible",
            "requested_label": baseline_a,
            "resolved_label": baseline_a,
            "timestamp": b1.timestamp,
        }
        report.baseline_recall = b1.overall_recall
        report.recall_delta = comparison.overall_delta
        report.regression_detected = bool(comparison.regressions)
        failed = [r.question_id for r in report.results if not r.passed]
        reasons = []
        if failed:
            reasons.append("One or more target questions failed their recorded thresholds.")
        if comparison.regressions:
            reasons.append("Required evidence regressed.")
        if not comparison.compatible:
            reasons.append("Baseline comparison is incompatible or unverified.")
        if threshold is not None and report.overall_recall < threshold:
            reasons.append("Target overall recall is below the explicit threshold.")
        report.quality = {
            "passed": not reasons,
            "mode": "evaluation",
            "failed_questions": failed,
            "reasons": reasons,
        }
    else:
        target = baseline_a if baseline_a is not None else baseline_label
        _, _, report = _run_probe(
            goldens, config, top_k, threshold, tag, include_drafts, baseline_label=target
        )
    _display_results(report, output, report.comparison)
    code = report_exit_code(report)
    if code:
        sys.exit(code)


@app.command("report")
@_command_errors
def render_report(
    report_path: Annotated[Path, typer.Argument(help="Saved LongProbe JSON report.")],
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Render a saved report without loading configuration or running retrieval."""
    validate_output(output)
    report = read_report(report_path)
    _display_results(report, output, report.comparison)


# ---------------------------------------------------------------------------
# Watch command
# ---------------------------------------------------------------------------


@app.command()
@_command_errors
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
                    _cfg, _adapter, report = _run_probe(
                        goldens_path=goldens_path,
                        config_path=config_path,
                        tags=tag,
                    )

                    diff_result = report.comparison
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
            f"\n[bold cyan]LongProbe watch stopped.[/bold cyan] (Ran {run_count} time(s))",
        )


# ---------------------------------------------------------------------------
# Capture command
# ---------------------------------------------------------------------------


@app.command()
@_command_errors
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
        console.print(
            "[bold red]Error:[/bold red] No questions provided. Use --question or --questions-file."
        )
        raise typer.Exit(1)

    check_destination(goldens, overwrite=True)
    if top_k <= 0 or match_mode not in {"id", "text", "semantic"}:
        raise ConfigurationError(
            "top_k must be positive; match_mode must be id, text, or semantic."
        )

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
    cfg.validate_for("capture")
    adapter = _create_adapter_from_config(cfg)

    existing_ids = {q.id for q in golden_set.questions}
    new_questions: list[GoldenQuestion] = []

    # 4. Iterate over questions
    for q_text in questions_to_ask:
        console.print(
            f'\n[bold cyan]🔍 Querying retriever for:[/bold cyan] [italic]"{q_text}"[/italic]'
        )
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
                    approved_chunks.extend(
                        [r["id"]] if isinstance(r.get("id"), str) and r["id"].strip() else []
                    )
                else:
                    approved_chunks.extend(
                        [r["text"]] if isinstance(r.get("text"), str) and r["text"].strip() else []
                    )
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
                    show_choices=True,
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

        status_val = "draft" if auto else "approved"
        golden_q = GoldenQuestion(
            id=q_id,
            question=q_text,
            match_mode=match_mode,
            required_chunks=approved_chunks,
            tags=list(tag),
            status=status_val,
            top_k=top_k,
        )
        new_questions.append(golden_q)
        console.print(
            f"   [green]✅ Saved question '{q_id}' ({status_val}) with {len(approved_chunks)} required chunks.[/green]"
        )

    # 5. Merge and save
    if new_questions:
        added = golden_set.merge(new_questions)
        golden_set.to_yaml(str(goldens))
        console.print(
            f"\n[bold green]Successfully added {added} new question(s) to {goldens}[/bold green]"
        )
        if any(q.status == "draft" for q in new_questions):
            console.print(
                "[dim]💡 Hint: Draft questions are excluded from CI checks by default. Run [cyan]longprobe edit[/cyan] to approve, or [cyan]longprobe check --include-drafts[/cyan] to test.[/dim]"
            )
    else:
        console.print("\n[dim]No new questions were captured.[/dim]")


# ---------------------------------------------------------------------------
# Generate command
# ---------------------------------------------------------------------------


@app.command()
@_command_errors
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
        help="LLM model (overrides config). E.g. gpt-5.5-instant, claude-haiku-4-5.",
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
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace an existing question-list --output file; not valid with --auto-capture.",
        ),
    ] = False,
) -> None:
    """Generate golden questions from documents using an LLM."""
    if (auto_capture and output is not None) or (overwrite and (output is None or auto_capture)):
        raise ConfigurationError(
            "Use --overwrite only with --output; --output and --auto-capture are separate modes."
        )

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

    cfg.generator = gen_config
    cfg.validate_for("auto_capture" if auto_capture else "generate")
    gen_config = cfg.generator
    if auto_capture and match_mode not in {"id", "text", "semantic"}:
        raise ConfigurationError("match_mode must be id, text, or semantic.")
    if num_questions < 0 or top_k <= 0:
        raise ConfigurationError("num_questions must be nonnegative and top_k must be positive.")

    golden_set = None
    if auto_capture:
        check_destination(goldens, overwrite=True)
        try:
            golden_set = GoldenSet.from_yaml(str(goldens))
        except FileNotFoundError:
            golden_set = GoldenSet(name="auto-generated", version="1.0", questions=[])
    elif output is not None:
        try:
            check_destination(output, overwrite=overwrite)
        except FileExistsError:
            raise ConfigurationError(
                "Output exists; use --overwrite to replace it or choose a new --output path."
            ) from None

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
    except RuntimeError:
        console.print("LLM generation failed; check provider credentials, model, and connection.")
        raise typer.Exit(1)

    if not questions:
        console.print("[bold yellow]Warning:[/bold yellow] LLM did not generate any questions.")
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
            console.print(f"[bold red]Error:[/bold red] Failed to create retriever adapter: {exc}")
            raise typer.Exit(1)

        # The destination has already passed preflight before generation.
        assert golden_set is not None

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
                f'  [dim]\U0001f50d[/dim] [italic]"{q_text[:60]}{"..." if len(q_text) > 60 else ""}"[/italic]'
            )
            try:
                start = time.perf_counter()
                results = adapter.retrieve(q_text, top_k=top_k)
                elapsed = (time.perf_counter() - start) * 1000
            except Exception:
                console.print(
                    "    Retriever request failed; check connection and credentials. Skipping."
                )
                skipped += 1
                continue

            console.print(f"    [dim]Retrieved {len(results)} chunks in {elapsed:.0f}ms[/dim]")

            if not results:
                console.print("    [yellow]No results. Skipping.[/yellow]")
                skipped += 1
                continue

            # Build required_chunks list.
            approved_chunks: list[str] = []
            for r in results:
                if match_mode == "id":
                    approved_chunks.extend(
                        [r["id"]] if isinstance(r.get("id"), str) and r["id"].strip() else []
                    )
                else:
                    approved_chunks.extend(
                        [r["text"]] if isinstance(r.get("text"), str) and r["text"].strip() else []
                    )

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
                status="draft",
                top_k=top_k,
            )
            new_questions.append(golden_q)

        # 5d. Merge and save.
        if new_questions:
            _added = golden_set.merge(new_questions)
            try:
                golden_set.to_yaml(str(goldens))
            except OSError as exc:
                console.print(f"[bold red]Error:[/bold red] Failed to write golden set: {exc}")
                raise typer.Exit(1)

        captured_count = len(new_questions)
        if not captured_count:
            console.print(
                f"No valid new questions captured; existing goldens were unchanged. Skipped: {skipped}"
            )
            return
        console.print(
            Panel(
                f"Generated:   [bold cyan]{len(questions)}[/bold cyan] questions\n"
                f"Captured:    [bold green]{captured_count}[/bold green] questions\n"
                f"Skipped:     [bold yellow]{skipped}[/bold yellow] (no results / errors)\n"
                f"Status:      [bold yellow]draft[/bold yellow] (unapproved)\n"
                f"Saved to:    [bold]{goldens}[/bold]\n\n"
                f"[dim]💡 Hint: Run [cyan]longprobe edit[/cyan] to approve, or [cyan]longprobe check --include-drafts[/cyan] to test.[/dim]",
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
        content = _yaml.safe_dump(
            data, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
        atomic_write_text(output, content, overwrite=overwrite)
        console.print(
            Panel(
                f"Generated [bold cyan]{len(questions)}[/bold cyan] questions.\n"
                f"Saved to [bold]{output}[/bold]\n\n"
                f"[dim]This is a question list, not an evaluable golden set; capture and verify expected evidence next.[/dim]",
                title="Generation Complete",
                border_style="green",
            )
        )
    else:
        # Print to stdout, one per line.
        for q in questions:
            console.print(q)


# ---------------------------------------------------------------------------
# Edit command & helpers
# ---------------------------------------------------------------------------


def edit_approve_all_drafts(golden_set: GoldenSet) -> int:
    """Promote all questions with status 'draft' to 'approved'. Returns count updated."""
    updated = 0
    for q in golden_set.questions:
        if q.status.lower() == "draft":
            q.status = "approved"
            updated += 1
    return updated


def edit_set_status(golden_set: GoldenSet, question_id: str, new_status: str) -> bool:
    """Set status for a question by ID. Returns True if question found and updated."""
    for q in golden_set.questions:
        if q.id == question_id:
            q.status = new_status.lower()
            return True
    return False


def edit_update_question(
    golden_set: GoldenSet,
    question_id: str,
    question_text: str | None = None,
    required_chunks: list[str] | None = None,
    match_mode: str | None = None,
    top_k: int | None = None,
    status: str | None = None,
) -> bool:
    """Update fields on a question by ID. Returns True if updated."""
    for q in golden_set.questions:
        if q.id == question_id:
            if question_text is not None:
                q.question = question_text
            if required_chunks is not None:
                q.required_chunks = required_chunks
            if match_mode is not None:
                q.match_mode = match_mode.lower()
            if top_k is not None:
                q.top_k = top_k
            if status is not None:
                q.status = status.lower()
            return True
    return False


def edit_delete_question(golden_set: GoldenSet, question_id: str) -> bool:
    """Delete a question from golden set by ID. Returns True if deleted."""
    original_len = len(golden_set.questions)
    golden_set.questions = [q for q in golden_set.questions if q.id != question_id]
    return len(golden_set.questions) < original_len


@app.command()
@_command_errors
def edit(
    goldens: Path = typer.Option(
        "goldens.yaml",
        "--goldens",
        "-g",
        help="Path to golden questions YAML file to edit.",
    ),
) -> None:
    """Interactive TUI editor to browse, edit, and approve golden questions."""
    if not goldens.exists():
        console.print(f"[bold red]Error:[/bold red] Golden set file not found: {goldens}")
        raise typer.Exit(1)

    golden_set = GoldenSet.from_yaml(str(goldens))
    modified = False

    while True:
        draft_count = sum(1 for q in golden_set.questions if q.status.lower() == "draft")
        approved_count = sum(1 for q in golden_set.questions if q.status.lower() == "approved")

        console.print()
        table = Table(
            title=f"Golden Questions: {golden_set.name} (v{golden_set.version})",
            header_style="bold cyan",
            show_lines=True,
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("ID", style="bold cyan", max_width=25)
        table.add_column("Status", width=12)
        table.add_column("Match", width=10)
        table.add_column("Top-K", width=7)
        table.add_column("Question", max_width=50)

        for idx, q in enumerate(golden_set.questions, 1):
            status_badge = (
                "[bold green]APPROVED[/bold green]"
                if q.status.lower() == "approved"
                else "[bold yellow]DRAFT[/bold yellow]"
            )
            table.add_row(
                str(idx),
                q.id,
                status_badge,
                q.match_mode,
                str(q.top_k),
                q.question[:47] + "..." if len(q.question) > 50 else q.question,
            )

        console.print(table)
        console.print(
            f"[dim]Total: {len(golden_set.questions)} | Approved: {approved_count} | Drafts: {draft_count}[/dim]\n"
        )

        console.print("[bold]Options:[/bold]")
        console.print("  [cyan]a[/cyan] - Approve all draft questions")
        console.print("  [cyan]e <#>[/cyan] - Edit question by index (e.g. e 1)")
        console.print("  [cyan]d <#>[/cyan] - Delete question by index (e.g. d 1)")
        console.print("  [cyan]s[/cyan] - Save and exit")
        console.print("  [cyan]q[/cyan] - Quit without saving")

        action = Prompt.ask("\nAction", default="s").strip().lower()

        if action == "a":
            count = edit_approve_all_drafts(golden_set)
            if count > 0:
                modified = True
                console.print(f"[bold green]Approved {count} draft question(s).[/bold green]")
            else:
                console.print("[yellow]No draft questions to approve.[/yellow]")
        elif action.startswith("e "):
            try:
                idx = int(action.split()[1]) - 1
                if 0 <= idx < len(golden_set.questions):
                    target_q = golden_set.questions[idx]
                    console.print(f"\n[bold cyan]Editing question '{target_q.id}'[/bold cyan]")
                    new_q_text = Prompt.ask("Question text", default=target_q.question)
                    new_match_mode = Prompt.ask(
                        "Match mode (id, text, semantic)",
                        default=target_q.match_mode,
                        choices=["id", "text", "semantic"],
                    )
                    new_status = Prompt.ask(
                        "Status (approved, draft)",
                        default=target_q.status,
                        choices=["approved", "draft"],
                    )

                    edit_update_question(
                        golden_set,
                        target_q.id,
                        question_text=new_q_text,
                        match_mode=new_match_mode,
                        status=new_status,
                    )
                    modified = True
                    console.print(f"[green]Updated question '{target_q.id}'.[/green]")
                else:
                    console.print("[red]Invalid index.[/red]")
            except (ValueError, IndexError):
                console.print("[red]Invalid edit command format. Use 'e <index>'.[/red]")
        elif action.startswith("d "):
            try:
                idx = int(action.split()[1]) - 1
                if 0 <= idx < len(golden_set.questions):
                    target_q = golden_set.questions[idx]
                    if edit_delete_question(golden_set, target_q.id):
                        modified = True
                        console.print(f"[red]Deleted question '{target_q.id}'.[/red]")
                else:
                    console.print("[red]Invalid index.[/red]")
            except (ValueError, IndexError):
                console.print("[red]Invalid delete command format. Use 'd <index>'.[/red]")
        elif action == "s":
            golden_set.to_yaml(str(goldens))
            console.print(f"[bold green]Saved golden set to {goldens}[/bold green]")
            break
        elif action == "q":
            if modified:
                if Confirm.ask("Quit without saving unsaved changes?"):
                    console.print("[dim]Exited without saving.[/dim]")
                    break
            else:
                break


@app.command()
@_command_errors
def explain(
    question_id: str = typer.Argument(
        ...,
        help="ID of the question to explain failure for.",
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
    include_drafts: bool = typer.Option(
        False,
        "--include-drafts",
        help="Include draft questions in evaluation.",
    ),
) -> None:
    """Print a diagnostic explanation for a failed golden question."""
    _cfg, _adapter, report = _run_probe(
        goldens_path=goldens,
        config_path=config,
        include_drafts=include_drafts,
    )

    target_res = next((r for r in report.results if r.question_id == question_id), None)
    if target_res is None:
        console.print(
            f"[bold red]Error:[/bold red] Question ID '{question_id}' not found in evaluated report."
        )
        raise typer.Exit(1)

    explanation = explain_failure(target_res)
    console.print(
        Panel(
            explanation,
            title=f"Diagnostic Explanation: {question_id}",
            border_style="yellow" if not target_res.passed else "green",
        )
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
