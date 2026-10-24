"""Reusable-workflow helpers, also executable locally without GitHub access.

Artifact download and PR publication remain GitHub workflow responsibilities.
This module validates downloaded data, evaluates once, and derives feedback.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path

from .cli.output import escape_annotation, github_annotations
from .config import ConfigurationError, ProbeConfig
from .core.diff import DiffReporter
from .core.evaluation import evaluate
from .core.reporting import ReportError, read_report, report_from_dict


def validate_baseline_inputs(artifact: str, run_id: str, label: str) -> None:
    if bool(artifact) != bool(run_id):
        raise ConfigurationError(
            "Baseline artifact name and source run ID must be supplied together."
        )
    if artifact and (not re.fullmatch(r"[1-9][0-9]*", run_id) or not label.strip()):
        raise ConfigurationError(
            "Baseline source run must be a positive integer and label must be nonempty."
        )


def restore_baseline(
    artifact_dir: Path, destination: Path, config_path: Path, goldens_path: Path, label: str
) -> None:
    """Validate an existing artifact read-only before replacing any destination."""
    from .core.golden import GoldenSet

    config = ProbeConfig.from_yaml(str(config_path))
    config.validate_for("baseline")
    if destination.resolve() != Path(config.baseline.db_path).resolve():
        raise ConfigurationError(
            "Restored baseline path must match baseline.db_path in the selected configuration."
        )
    candidates = list(artifact_dir.rglob("*.db"))
    if len(candidates) != 1 or candidates[0].is_symlink() or not candidates[0].is_file():
        raise ConfigurationError(
            "Baseline artifact must contain exactly one regular SQLite .db file."
        )
    source = candidates[0]
    if not source.resolve().is_relative_to(artifact_dir.resolve()) or destination.is_symlink():
        raise ConfigurationError("Baseline artifact and destination must use regular local paths.")
    if any(
        Path(str(path) + suffix).exists()
        for path in (source, destination)
        for suffix in ("-wal", "-shm", "-journal")
    ):
        raise ConfigurationError(
            "Use a closed standalone baseline database without journal sidecars."
        )
    try:
        with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ConfigurationError("Baseline artifact database failed integrity validation.")
            row = conn.execute(
                "SELECT report_json FROM baselines WHERE label = ?", (label,)
            ).fetchone()
        if row is None:
            raise ConfigurationError(
                "Requested baseline label is absent from the restored artifact."
            )
        baseline = report_from_dict(json.loads(row[0]))
    except (sqlite3.Error, json.JSONDecodeError):
        raise ConfigurationError(
            "Baseline artifact does not contain a valid LongProbe database."
        ) from None
    # Build a contract preview using a local empty-result callable, never a service.
    preview = evaluate(
        GoldenSet.from_yaml(str(goldens_path)), lambda query, k: [], config, compare=False
    )
    comparison = DiffReporter().diff(preview, baseline)
    if not comparison.compatible:
        raise ConfigurationError(
            "Restored baseline is incompatible: " + "; ".join(comparison.compatibility_reasons)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".baseline-")
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def determine_exit(report_path: Path, process_exit: int, blocking: bool) -> int:
    """Quality relaxation cannot hide invalid input, invalid JSON, or crashes."""
    if process_exit not in {0, 1}:
        return process_exit if process_exit in {2, 3} else 3
    try:
        report = read_report(report_path)
    except (ValueError, OSError):
        return 3
    if report.schema_version != 2 or not report.quality:
        return 3
    if report.comparison is not None and not report.comparison["compatible"]:
        return 2
    if process_exit == 1 and report.quality["passed"]:
        return 3  # Contradictory process/report state is not a quality relaxation.
    return 1 if blocking and not report.quality["passed"] else 0


def _markdown(value) -> str:
    text = html.escape(str(value)).replace("\r", " ").replace("\n", " ")
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", text)


def markdown_report(report) -> str:
    """Human feedback uses the recorded verdict rather than guessed defaults."""
    identity = report.baseline_identity
    comparison = report.comparison or {}
    lines = [
        "<!-- longprobe-report -->",
        "## LongProbe RAG Regression Report",
        "",
        f"Quality: **{'PASS' if report.quality.get('passed') else 'FAIL'}**",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Recall | {report.overall_recall:.1%} |",
        f"| Passing questions | {sum(r.passed for r in report.results)}/{len(report.results)} |",
        f"| Baseline | {_markdown(identity.get('resolved_label') or identity.get('requested_label') or 'none')} |",
        f"| Comparison | {_markdown(identity.get('status', 'not_requested'))} |",
        "",
    ]
    for reason in report.quality.get("reasons", []):
        lines.append("- " + _markdown(reason))
    for label, key in (
        ("Added questions", "added_questions"),
        ("Removed questions", "removed_questions"),
    ):
        if comparison.get(key):
            lines.append(f"- {label}: " + ", ".join(_markdown(q) for q in comparison[key]))
    for change in comparison.get("changed_questions", []):
        lines.append(
            f"- Changed {_markdown(change['question_id'])}: "
            + ", ".join(_markdown(f) for f in change["fields"])
        )
    for regression in comparison.get("regressions", []):
        lines.append(
            f"- Regression {_markdown(regression['question_id'])}: lost "
            + ", ".join(_markdown(c) for c in regression["newly_lost_chunks"])
        )
    for result in report.results:
        if result.missing_chunks:
            lines.append(
                f"- {_markdown(result.question_id)} missing: "
                + ", ".join(_markdown(c) for c in result.missing_chunks)
            )
    text = "\n".join(lines) + "\n"
    return text[:59000] + (
        "\n\nReport truncated; inspect the JSON artifact.\n" if len(text) > 59000 else ""
    )


def run_evaluation(
    config: Path,
    goldens: Path,
    *,
    label: str | None = None,
    blocking: bool = True,
    directory: Path = Path("."),
) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "longprobe-result.json"
    error_path = directory / "longprobe-diagnostics.txt"
    command = [
        sys.executable,
        "-m",
        "longprobe.cli.main",
        "check",
        "--config",
        str(config),
        "--goldens",
        str(goldens),
        "--output",
        "json",
    ]
    if label is not None:
        command += ["--baseline", label]
    else:
        command += ["--no-compare"]
    with (
        report_path.open("w", encoding="utf-8") as stdout,
        error_path.open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    code = determine_exit(report_path, result.returncode, blocking)
    has_result = False
    try:
        report = read_report(report_path)
        has_result = (
            report.schema_version == 2 and bool(report.quality) and result.returncode in {0, 1}
        )
    except (ValueError, OSError):
        report = None
    if has_result:
        print(github_annotations(report))
        (directory / "longprobe-comment.md").write_text(markdown_report(report), encoding="utf-8")
    else:
        print(
            "::error title=LongProbe Execution::Evaluation did not produce a valid report; inspect diagnostics."
        )
    if output_path := os.environ.get("GITHUB_OUTPUT"):
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"has-result={'true' if has_result else 'false'}\nexit-code={code}\n")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("validate-inputs")
    inputs.add_argument("--artifact", default="")
    inputs.add_argument("--run-id", default="")
    inputs.add_argument("--label", default="latest")
    restore = commands.add_parser("restore")
    restore.add_argument("--artifact-dir", required=True, type=Path)
    restore.add_argument("--destination", required=True, type=Path)
    restore.add_argument("--config", required=True, type=Path)
    restore.add_argument("--goldens", required=True, type=Path)
    restore.add_argument("--label", default="latest")
    run = commands.add_parser("run")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--goldens", required=True, type=Path)
    run.add_argument("--baseline")
    run.add_argument("--nonblocking", action="store_true")
    run.add_argument("--directory", type=Path, default=Path("."))
    args = parser.parse_args()
    try:
        if args.command == "validate-inputs":
            validate_baseline_inputs(args.artifact, args.run_id, args.label)
        elif args.command == "restore":
            restore_baseline(
                args.artifact_dir, args.destination, args.config, args.goldens, args.label
            )
        else:
            code = run_evaluation(
                args.config,
                args.goldens,
                label=args.baseline,
                blocking=not args.nonblocking,
                directory=args.directory,
            )
            # The workflow performs its final gate after uploads/comments.
            return 0 if os.environ.get("GITHUB_OUTPUT") else code
    except FileNotFoundError:
        print(
            "::error title=LongProbe Setup::Selected input was not found; check configuration and artifact paths.",
            file=sys.stderr,
        )
        return 2
    except (ConfigurationError, ReportError, ValueError) as exc:
        print("::error title=LongProbe Setup::" + escape_annotation(str(exc)), file=sys.stderr)
        return 2
    except (OSError, sqlite3.Error):
        print(
            "::error title=LongProbe Execution::File or database operation failed.", file=sys.stderr
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
