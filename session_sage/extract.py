"""Extract sessions and turns from the Copilot CLI SQLite session store."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_DB = Path.home() / ".copilot" / "session-store.db"


@dataclass
class Turn:
    session_id: str
    turn_index: int
    user_message: str
    assistant_response: str
    timestamp: str
    session_summary: Optional[str] = None
    session_cwd: Optional[str] = None
    session_repository: Optional[str] = None


@dataclass
class Checkpoint:
    session_id: str
    title: Optional[str]
    overview: Optional[str]
    work_done: Optional[str]
    technical_details: Optional[str]
    created_at: str


@dataclass
class SessionMeta:
    id: str
    summary: Optional[str]
    cwd: Optional[str]
    repository: Optional[str]
    branch: Optional[str]
    created_at: str
    turns: list[Turn] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return column names for a table; empty set if table is absent."""
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row[1] for row in rows}
    except sqlite3.Error:
        return set()


def load_all(db_path: Path = DEFAULT_DB, since: Optional[str] = None) -> list[SessionMeta]:
    """Load all sessions with their turns, checkpoints, and file touches.

    Args:
        db_path: Path to the SQLite session store.
        since: Optional ISO-format date string (e.g. '2026-06-01') to filter sessions.
    """
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return _load(conn, since)


def _load(conn: sqlite3.Connection, since: Optional[str]) -> list[SessionMeta]:
    sessions: dict[str, SessionMeta] = {}

    # Use parameterized query to prevent injection via --since argument
    if since:
        rows = conn.execute(
            "SELECT id, summary, cwd, repository, branch, created_at "
            "FROM sessions WHERE created_at >= ? ORDER BY created_at",
            (since,),
        )
    else:
        rows = conn.execute(
            "SELECT id, summary, cwd, repository, branch, created_at FROM sessions ORDER BY created_at"
        )
    for row in rows:
        sessions[row["id"]] = SessionMeta(
            id=row["id"],
            summary=row["summary"],
            cwd=row["cwd"],
            repository=row["repository"],
            branch=row["branch"],
            created_at=row["created_at"],
        )

    if not sessions:
        return []

    sids_sql = "(" + ",".join(f"'{sid}'" for sid in sessions) + ")"

    # --- turns ---
    if _table_columns(conn, "turns"):
        for row in conn.execute(
            f"""
            SELECT session_id, turn_index, user_message, assistant_response, timestamp
            FROM turns
            WHERE session_id IN {sids_sql}
              AND user_message IS NOT NULL AND trim(user_message) != ''
            ORDER BY timestamp
            """
        ):
            sid = row["session_id"]
            if sid not in sessions:
                continue
            s = sessions[sid]
            s.turns.append(
                Turn(
                    session_id=sid,
                    turn_index=row["turn_index"],
                    user_message=row["user_message"] or "",
                    assistant_response=row["assistant_response"] or "",
                    timestamp=row["timestamp"],
                    session_summary=s.summary,
                    session_cwd=s.cwd,
                    session_repository=s.repository,
                )
            )

    # --- checkpoints ---
    cp_cols = _table_columns(conn, "checkpoints")
    if cp_cols:
        select_cols = ", ".join(
            [c for c in ["session_id", "title", "overview", "work_done", "technical_details", "created_at"]
             if c in cp_cols]
        )
        for row in conn.execute(
            f"SELECT {select_cols} FROM checkpoints "
            f"WHERE session_id IN {sids_sql} ORDER BY created_at"
        ):
            sid = row["session_id"]
            if sid not in sessions:
                continue
            sessions[sid].checkpoints.append(
                Checkpoint(
                    session_id=sid,
                    title=row["title"] if "title" in cp_cols else None,
                    overview=row["overview"] if "overview" in cp_cols else None,
                    work_done=row["work_done"] if "work_done" in cp_cols else None,
                    technical_details=row["technical_details"] if "technical_details" in cp_cols else None,
                    created_at=row["created_at"] if "created_at" in cp_cols else "",
                )
            )

    # --- files touched (deduplicated) ---
    if _table_columns(conn, "session_files"):
        seen: dict[str, set[str]] = {sid: set() for sid in sessions}
        for row in conn.execute(
            f"SELECT session_id, file_path FROM session_files "
            f"WHERE session_id IN {sids_sql} ORDER BY first_seen_at"
        ):
            sid = row["session_id"]
            if sid in sessions and row["file_path"] not in seen[sid]:
                seen[sid].add(row["file_path"])
                sessions[sid].files_touched.append(row["file_path"])

    return list(sessions.values())
