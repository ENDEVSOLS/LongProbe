"""
Diff reporter for LongProbe RAG regression testing.

Compares two :class:`~longprobe.core.scorer.ProbeReport` instances (a *current*
run against a *baseline*) and produces a structured :class:`RegressionDiff`
that can be rendered as a rich terminal table, raw JSON, or GitHub Actions
workflow-command annotations.

Typical usage::

    reporter = DiffReporter()
    diff = reporter.diff(current_report, baseline_report)
    print(reporter.format_table(diff))        # rich terminal output
    print(reporter.format_json(diff))          # machine-readable JSON
    print(reporter.format_github(diff))        # CI annotations
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from .scorer import ProbeReport, QuestionResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ChunkRegression:
    """Describes a single question whose recall has *decreased* compared to
    the baseline.

    Attributes:
        question_id: Identifier of the affected question.
        question: The natural-language question text.
        newly_lost_chunks: Chunk IDs that were found in the baseline but are
            missing in the current run, sorted alphabetically.
        baseline_recall: Recall score observed in the baseline run.
        current_recall: Recall score observed in the current run.
    """

    question_id: str
    question: str
    newly_lost_chunks: list[str] = field(default_factory=list)
    baseline_recall: float = 0.0
    current_recall: float = 0.0


@dataclass
class ChunkImprovement:
    """Describes a single question whose recall has *increased* compared to
    the baseline.

    Attributes:
        question_id: Identifier of the affected question.
        question: The natural-language question text.
        newly_found_chunks: Chunk IDs that are newly found in the current run
            but were missing in the baseline, sorted alphabetically.
        baseline_recall: Recall score observed in the baseline run.
        current_recall: Recall score observed in the current run.
    """

    question_id: str
    question: str
    newly_found_chunks: list[str] = field(default_factory=list)
    baseline_recall: float = 0.0
    current_recall: float = 0.0


@dataclass
class RegressionDiff:
    """Complete diff between a current evaluation and a baseline.

    Attributes:
        overall_delta: Difference between current and baseline overall recall.
            Negative values indicate a regression.
        regressions: List of questions that regressed.
        improvements: List of questions that improved.
        unchanged: Question IDs whose recall did not change (within epsilon).
    """

    overall_delta: float = 0.0
    regressions: list[ChunkRegression] = field(default_factory=list)
    improvements: list[ChunkImprovement] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DiffReporter
# ---------------------------------------------------------------------------

_EPSILON: float = 1e-9


class DiffReporter:
    """Computes and formats regression diffs between two probe reports.

    Usage::

        reporter = DiffReporter()
        diff = reporter.diff(current_report, baseline_report)
        print(reporter.format_table(diff))
    """

    def __init__(self) -> None:
        """Initialise the reporter.  No persistent state is required."""

    # ------------------------------------------------------------------
    # Core diff logic
    # ------------------------------------------------------------------

    def diff(self, current: ProbeReport, baseline: ProbeReport) -> RegressionDiff:
        """Compare *current* against *baseline* and return a structured diff.

        For every question present in both reports the per-question recall
        scores are compared.  A small epsilon (``1e-9``) is used to decide
        equality so that floating-point noise does not produce spurious
        diffs.

        Args:
            current: The probe report from the run under test.
            baseline: The reference (previous) probe report.

        Returns:
            A :class:`RegressionDiff` with regressions, improvements, and
            unchanged question IDs.
        """
        # Build a lookup from question_id -> QuestionResult for the baseline.
        baseline_map: dict[str, QuestionResult] = {
            r.question_id: r for r in baseline.results
        }

        regressions: list[ChunkRegression] = []
        improvements: list[ChunkImprovement] = []
        unchanged: list[str] = []

        for cur_result in current.results:
            base_result = baseline_map.get(cur_result.question_id)
            if base_result is None:
                # New question that did not exist in the baseline — skip.
                continue

            baseline_found_set = set(base_result.found_chunks)
            current_found_set = set(cur_result.found_chunks)

            if cur_result.recall_score < base_result.recall_score - _EPSILON:
                # Regression: recall dropped.
                lost = sorted(
                    [c for c in base_result.found_chunks if c not in current_found_set]
                )
                regressions.append(
                    ChunkRegression(
                        question_id=cur_result.question_id,
                        question=cur_result.question,
                        newly_lost_chunks=lost,
                        baseline_recall=base_result.recall_score,
                        current_recall=cur_result.recall_score,
                    )
                )
            elif cur_result.recall_score > base_result.recall_score + _EPSILON:
                # Improvement: recall increased.
                found = sorted(
                    [c for c in cur_result.found_chunks if c not in baseline_found_set]
                )
                improvements.append(
                    ChunkImprovement(
                        question_id=cur_result.question_id,
                        question=cur_result.question,
                        newly_found_chunks=found,
                        baseline_recall=base_result.recall_score,
                        current_recall=cur_result.recall_score,
                    )
                )
            else:
                unchanged.append(cur_result.question_id)

        overall_delta = current.overall_recall - baseline.overall_recall

        return RegressionDiff(
            overall_delta=overall_delta,
            regressions=regressions,
            improvements=improvements,
            unchanged=unchanged,
        )

    # ------------------------------------------------------------------
    # Formatters
    # ------------------------------------------------------------------

    def format_table(self, diff: RegressionDiff) -> str:
        """Render the diff as a rich terminal table.

        Uses :mod:`rich` to produce a coloured, human-readable report
        suitable for CLI output.

        Args:
            diff: The structured regression diff to render.

        Returns:
            The rendered table as a plain-text string (ANSI escape codes
            stripped via ``Console(file=StringIO)``).
        """
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, no_color=False)

        # --- Header panel ------------------------------------------------
        delta_sign = "+" if diff.overall_delta >= 0 else ""
        header_text = Text.assemble(
            ("LongProbe Regression Report\n", "bold"),
            (
                f"Overall recall delta: {delta_sign}{diff.overall_delta:.4f}  |  "
                f"Regressions: {len(diff.regressions)}  |  "
                f"Improvements: {len(diff.improvements)}  |  "
                f"Unchanged: {len(diff.unchanged)}",
                "cyan",
            ),
        )
        console.print(Panel(header_text, border_style="bold blue"))

        # --- Regressions table -------------------------------------------
        if diff.regressions:
            reg_table = Table(
                title="⚠  Regressions",
                title_style="bold red",
                border_style="red",
                show_lines=True,
                expand=True,
            )
            reg_table.add_column("Question ID", style="bold red", max_width=24)
            reg_table.add_column("Question", max_width=50)
            reg_table.add_column("Lost Chunks", max_width=40)
            reg_table.add_column("Baseline", justify="right", max_width=10)
            reg_table.add_column("Current", justify="right", max_width=10)

            for r in diff.regressions:
                question_display = (
                    r.question[:47] + "..." if len(r.question) > 50 else r.question
                )
                chunks_display = ", ".join(r.newly_lost_chunks)
                if len(chunks_display) > 38:
                    chunks_display = chunks_display[:35] + "..."
                reg_table.add_row(
                    r.question_id,
                    question_display,
                    Text(chunks_display, style="red"),
                    f"{r.baseline_recall:.4f}",
                    f"{r.current_recall:.4f}",
                )

            console.print()
            console.print(reg_table)

        # --- Improvements table ------------------------------------------
        if diff.improvements:
            imp_table = Table(
                title="✓  Improvements",
                title_style="bold green",
                border_style="green",
                show_lines=True,
                expand=True,
            )
            imp_table.add_column("Question ID", style="bold green", max_width=24)
            imp_table.add_column("Question", max_width=50)
            imp_table.add_column("Found Chunks", max_width=40)
            imp_table.add_column("Baseline", justify="right", max_width=10)
            imp_table.add_column("Current", justify="right", max_width=10)

            for i in diff.improvements:
                question_display = (
                    i.question[:47] + "..." if len(i.question) > 50 else i.question
                )
                chunks_display = ", ".join(i.newly_found_chunks)
                if len(chunks_display) > 38:
                    chunks_display = chunks_display[:35] + "..."
                imp_table.add_row(
                    i.question_id,
                    question_display,
                    Text(chunks_display, style="green"),
                    f"{i.baseline_recall:.4f}",
                    f"{i.current_recall:.4f}",
                )

            console.print()
            console.print(imp_table)

        # --- No changes --------------------------------------------------
        if not diff.regressions and not diff.improvements:
            console.print()
            console.print(
                Text(
                    "No recall changes detected between current and baseline.",
                    style="bold dim",
                )
            )

        return buf.getvalue()

    def format_json(self, diff: RegressionDiff) -> str:
        """Serialise the diff to a JSON string.

        All dataclass fields are recursively converted via
        :func:`dataclasses.asdict`.

        Args:
            diff: The structured regression diff to serialise.

        Returns:
            A pretty-printed JSON string (2-space indentation).
        """
        return json.dumps(asdict(diff), indent=2)

    def format_github(self, diff: RegressionDiff) -> str:
        """Generate GitHub Actions workflow-command annotations.

        * Regressions are emitted as ``::error`` annotations so that they
          fail the CI step.
        * Improvements are emitted as ``::notice`` annotations.
        * A summary is wrapped in a ``::group::`` / ``::endgroup::`` block.

        Args:
            diff: The structured regression diff to render.

        Returns:
            The complete annotation string ready for ``echo``-ing in a
            GitHub Actions workflow.
        """
        lines: list[str] = []

        for r in diff.regressions:
            chunks_str = ", ".join(r.newly_lost_chunks) if r.newly_lost_chunks else "none"
            lines.append(
                f"::error file=longprobe,title=Retrieval Regression::"
                f"Question {r.question_id}: recall dropped from "
                f"{r.baseline_recall:.2f} to {r.current_recall:.2f}. "
                f"Lost chunks: {chunks_str}"
            )

        for i in diff.improvements:
            lines.append(
                f"::notice file=longprobe,title=Retrieval Improvement::"
                f"Question {i.question_id}: recall improved from "
                f"{i.baseline_recall:.2f} to {i.current_recall:.2f}."
            )

        # Summary group
        lines.append("::group::LongProbe Regression Summary")
        delta_sign = "+" if diff.overall_delta >= 0 else ""
        lines.append(
            f"Overall recall delta: {delta_sign}{diff.overall_delta:.4f}"
        )
        lines.append(f"Regressions: {len(diff.regressions)}")
        lines.append(f"Improvements: {len(diff.improvements)}")
        lines.append(f"Unchanged: {len(diff.unchanged)}")
        lines.append("::endgroup::")

        return "\n".join(lines)
