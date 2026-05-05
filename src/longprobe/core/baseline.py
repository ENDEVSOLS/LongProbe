"""
SQLite-backed baseline store for LongProbe RAG regression testing.

Persists :class:`ProbeReport` snapshots to a local SQLite database so that
subsequent evaluation runs can be compared against historical baselines.
Each baseline is identified by a user-defined *label* (default ``"latest"``).

Typical usage::

    store = BaselineStore()
    store.save(report, label="v1.0")
    baseline = store.load("v1.0")
    delta = store.diff(report, baseline)
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone

from .scorer import ProbeReport, QuestionResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dict_to_question_result(data: dict) -> QuestionResult:
    """Reconstruct a :class:`QuestionResult` from a plain dictionary.

    Handles the deserialisation of every field declared on the dataclass,
    including list-of-strings fields that arrive as native JSON lists.

    Args:
        data: Dictionary with keys matching the dataclass fields.

    Returns:
        A fully populated :class:`QuestionResult` instance.
    """
    return QuestionResult(
        question_id=data["question_id"],
        question=data["question"],
        recall_score=float(data["recall_score"]),
        retrieved_chunk_ids=list(data["retrieved_chunk_ids"]),
        required_chunks=list(data["required_chunks"]),
        missing_chunks=list(data["missing_chunks"]),
        found_chunks=list(data["found_chunks"]),
        passed=bool(data["passed"]),
        latency_ms=float(data["latency_ms"]),
    )


def _dict_to_report(data: dict) -> ProbeReport:
    """Reconstruct a :class:`ProbeReport` from a plain dictionary.

    Nested :class:`QuestionResult` objects inside the ``results`` list are
    rebuilt via :func:`_dict_to_question_result`.

    Args:
        data: Dictionary with keys matching the :class:`ProbeReport` fields.

    Returns:
        A fully populated :class:`ProbeReport` instance.
    """
    results = [
        _dict_to_question_result(r) for r in data.get("results", [])
    ]

    return ProbeReport(
        golden_set_name=data["golden_set_name"],
        golden_set_version=data["golden_set_version"],
        timestamp=data["timestamp"],
        overall_recall=float(data["overall_recall"]),
        pass_rate=float(data["pass_rate"]),
        results=results,
        baseline_recall=float(data["baseline_recall"]) if data.get("baseline_recall") is not None else None,
        recall_delta=float(data["recall_delta"]) if data.get("recall_delta") is not None else None,
        regression_detected=bool(data.get("regression_detected", False)),
    )


# ---------------------------------------------------------------------------
# BaselineStore
# ---------------------------------------------------------------------------


class BaselineStore:
    """SQLite-backed store for :class:`ProbeReport` snapshots.

    Each stored baseline is keyed by a *label* string.  Saving a report
    under an existing label replaces the previous snapshot (upsert
    semantics).

    The database is a single file located at *db_path*.  The parent
    directory is created automatically if it does not exist.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = ".longprobe/baselines.db") -> None:
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Ensure the ``baselines`` table and supporting index exist."""
        with closing(self._get_connection()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    golden_set_name TEXT NOT NULL,
                    golden_set_version TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_baselines_label
                ON baselines(label)
                """
            )
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Return a new SQLite connection.

        A fresh connection is created on every call so that the store is
        safe to use from multiple threads.  The caller is responsible for
        closing the connection; ``closing()`` is the intended pattern.

        Returns:
            An open :class:`sqlite3.Connection` with ``Row`` factory and
            declarative type parsing enabled.
        """
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def save(self, report: ProbeReport, label: str = "latest") -> None:
        """Persist a :class:`ProbeReport` under *label*.

        If a row with the same label already exists it is replaced;
        otherwise a new row is inserted (upsert).

        Args:
            report: The probe report to store.
            label: Unique identifier for the baseline snapshot.
        """
        report_json = json.dumps(asdict(report), ensure_ascii=False)
        timestamp = (
            datetime.now(timezone.utc).isoformat()
            if not report.timestamp
            else report.timestamp
        )

        with closing(self._get_connection()) as conn:
            # Check whether a row with this label already exists.
            row = conn.execute(
                "SELECT id FROM baselines WHERE label = ?",
                (label,),
            ).fetchone()

            if row is not None:
                conn.execute(
                    """
                    UPDATE baselines
                    SET golden_set_name  = ?,
                        golden_set_version = ?,
                        timestamp          = ?,
                        report_json        = ?
                    WHERE label = ?
                    """,
                    (
                        report.golden_set_name,
                        report.golden_set_version,
                        timestamp,
                        report_json,
                        label,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO baselines (
                        label, golden_set_name, golden_set_version,
                        timestamp, report_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        label,
                        report.golden_set_name,
                        report.golden_set_version,
                        timestamp,
                        report_json,
                    ),
                )

            conn.commit()

    def load(self, label: str = "latest") -> ProbeReport | None:
        """Load the most recent baseline for *label*.

        Args:
            label: The baseline label to look up.

        Returns:
            A reconstructed :class:`ProbeReport`, or ``None`` if no
            baseline exists for the given label.
        """
        with closing(self._get_connection()) as conn:
            row = conn.execute(
                "SELECT report_json FROM baselines WHERE label = ?",
                (label,),
            ).fetchone()

        if row is None:
            return None

        data = json.loads(row["report_json"])
        return _dict_to_report(data)

    def list_labels(self) -> list[dict]:
        """Return metadata for every stored baseline.

        Returns:
            A list of dictionaries, each containing ``label``,
            ``golden_set_name``, ``golden_set_version``, ``timestamp``,
            and ``created_at``, ordered by most recent first.
        """
        with closing(self._get_connection()) as conn:
            rows = conn.execute(
                """
                SELECT
                    label,
                    golden_set_name,
                    golden_set_version,
                    timestamp,
                    created_at
                FROM baselines
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [
            {
                "label": row["label"],
                "golden_set_name": row["golden_set_name"],
                "golden_set_version": row["golden_set_version"],
                "timestamp": row["timestamp"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete(self, label: str) -> bool:
        """Delete the baseline identified by *label*.

        Args:
            label: The baseline label to delete.

        Returns:
            ``True`` if a row was deleted, ``False`` if the label did not
            exist.
        """
        with closing(self._get_connection()) as conn:
            cursor = conn.execute(
                "DELETE FROM baselines WHERE label = ?",
                (label,),
            )
            conn.commit()

        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Diff / regression analysis
    # ------------------------------------------------------------------

    def diff(self, current: ProbeReport, baseline: ProbeReport) -> dict:
        """Compare a current run against a stored baseline.

        For every question present in *current*, the method determines
        whether recall has regressed, improved, or stayed the same
        relative to the baseline and collects the specific chunks that
        changed.

        Args:
            current: The freshly produced probe report.
            baseline: The previously stored baseline report.

        Returns:
            A dictionary with the following keys:

            * ``overall_delta`` – difference in overall recall (negative
              indicates regression).
            * ``regressions`` – list of question-level regressions with
              newly lost chunk identifiers.
            * ``improvements`` – list of question-level improvements with
              newly found chunk identifiers.
            * ``unchanged`` – list of question IDs whose recall did not
              change.
        """
        # Build a lookup from the baseline results keyed by question_id.
        baseline_by_id: dict[str, QuestionResult] = {
            r.question_id: r for r in baseline.results
        }

        regressions: list[dict] = []
        improvements: list[dict] = []
        unchanged: list[str] = []

        for cur_result in current.results:
            base_result = baseline_by_id.get(cur_result.question_id)

            # New question not present in baseline – skip classification.
            if base_result is None:
                continue

            cur_recall = cur_result.recall_score
            base_recall = base_result.recall_score

            if cur_recall < base_recall:
                # Regression: recall went down.
                baseline_found = set(base_result.found_chunks)
                current_found = set(cur_result.found_chunks)
                newly_lost = sorted(
                    c for c in base_result.found_chunks if c not in current_found
                )
                regressions.append(
                    {
                        "question_id": cur_result.question_id,
                        "question": cur_result.question,
                        "newly_lost_chunks": newly_lost,
                        "baseline_recall": base_recall,
                        "current_recall": cur_recall,
                    }
                )
            elif cur_recall > base_recall:
                # Improvement: recall went up.
                newly_found = sorted(
                    c for c in cur_result.found_chunks
                    if c not in set(base_result.found_chunks)
                )
                improvements.append(
                    {
                        "question_id": cur_result.question_id,
                        "question": cur_result.question,
                        "newly_found_chunks": newly_found,
                        "baseline_recall": base_recall,
                        "current_recall": cur_recall,
                    }
                )
            else:
                # Unchanged.
                unchanged.append(cur_result.question_id)

        return {
            "overall_delta": current.overall_recall - baseline.overall_recall,
            "regressions": regressions,
            "improvements": improvements,
            "unchanged": unchanged,
        }
