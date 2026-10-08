"""Persist local media and classification results for the History page."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

HISTORY_ROOT = Path(os.environ.get("MGENRE_HISTORY_DIR", "data/history"))
PAGE_SIZE = 10
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov"}


def _connect() -> sqlite3.Connection:
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(HISTORY_ROOT / "history.sqlite3", timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source_type TEXT NOT NULL,
            filename TEXT NOT NULL,
            media_file TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            result_json TEXT NOT NULL
        )"""
    )
    return connection


@contextmanager
def _database():
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def save_analysis(
    source_path: Path,
    *,
    filename: str,
    source_type: str,
    suffix: str,
    result: dict,
) -> int:
    """Copy the source media and save its result; return the new history ID."""
    if source_type not in {"upload", "link"}:
        raise ValueError("Unknown media source type.")
    media_dir = HISTORY_ROOT / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    safe_suffix = suffix.lower() if suffix.lower() in {".mp3", ".wav", ".flac", *VIDEO_SUFFIXES} else ""
    media_file = f"{uuid4().hex}{safe_suffix}"
    saved_path = media_dir / media_file
    try:
        shutil.copyfile(source_path, saved_path)
        with _database() as connection:
            cursor = connection.execute(
                """INSERT INTO analyses
                   (created_at, source_type, filename, media_file, size_bytes, result_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().astimezone().isoformat(timespec="seconds"),
                    source_type,
                    filename,
                    media_file,
                    saved_path.stat().st_size,
                    json.dumps(result),
                ),
            )
            return int(cursor.lastrowid)
    except Exception:
        saved_path.unlink(missing_ok=True)
        raise


def _entry(row: sqlite3.Row) -> dict:
    created_at = datetime.fromisoformat(row["created_at"])
    suffix = Path(row["media_file"]).suffix.lower()
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "created_at_display": created_at.strftime("%d %b %Y · %H:%M"),
        "source_type": row["source_type"],
        "filename": row["filename"],
        "media_file": row["media_file"],
        "media_kind": "video" if suffix in VIDEO_SUFFIXES else "audio" if suffix else "download",
        "media_exists": (HISTORY_ROOT / "media" / row["media_file"]).is_file(),
        "size_mb": round(row["size_bytes"] / (1024 * 1024), 1),
        "result": json.loads(row["result_json"]),
    }


def list_analyses(*, page: int = 1) -> tuple[list[dict], int]:
    """Return one newest-first page and the total number of saved analyses."""
    with _database() as connection:
        total = int(connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0])
        rows = connection.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ? OFFSET ?",
            (PAGE_SIZE, (page - 1) * PAGE_SIZE),
        ).fetchall()
    return [_entry(row) for row in rows], total


def media_path(analysis_id: int) -> Path | None:
    """Resolve a stored media file by ID without accepting filesystem paths."""
    with _database() as connection:
        row = connection.execute(
            "SELECT media_file FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    if row is None:
        return None
    media_dir = (HISTORY_ROOT / "media").resolve()
    path = (media_dir / row["media_file"]).resolve()
    return path if path.is_relative_to(media_dir) and path.is_file() else None
