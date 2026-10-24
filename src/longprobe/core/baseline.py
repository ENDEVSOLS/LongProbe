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
from datetime import UTC, datetime

from .reporting import report_from_dict, report_to_dict
from .scorer import ProbeReport

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_dict_to_report = report_from_dict


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
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Baseline label must be a nonempty string.")
        report_json = json.dumps(report_to_dict(report), ensure_ascii=False, allow_nan=False)
        timestamp = datetime.now(UTC).isoformat() if not report.timestamp else report.timestamp

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
        from .diff import DiffReporter

        return asdict(DiffReporter().diff(current, baseline))
