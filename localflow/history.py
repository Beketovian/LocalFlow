"""Dictation history and statistics (SQLite, fully local)."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class HistoryEntry:
    id: int
    timestamp: float
    raw_text: str
    formatted_text: str
    app: str
    language: str
    duration: float  # seconds of audio
    mode: str  # dictation | hands-free | command

    @property
    def words(self) -> int:
        return len(self.formatted_text.split())


@dataclass
class Stats:
    total_entries: int = 0
    total_words: int = 0
    total_audio_seconds: float = 0.0
    average_wpm: float = 0.0
    streak_days: int = 0
    words_today: int = 0


class History:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                raw_text TEXT NOT NULL,
                formatted_text TEXT NOT NULL,
                app TEXT DEFAULT '',
                language TEXT DEFAULT '',
                duration REAL DEFAULT 0,
                mode TEXT DEFAULT 'dictation'
            )"""
        )
        self._db.commit()

    def add(
        self,
        raw_text: str,
        formatted_text: str,
        app: str = "",
        language: str = "",
        duration: float = 0.0,
        mode: str = "dictation",
        timestamp: Optional[float] = None,
    ) -> int:
        cur = self._db.execute(
            "INSERT INTO history (timestamp, raw_text, formatted_text, app, language, duration, mode)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (timestamp or time.time(), raw_text, formatted_text, app, language, duration, mode),
        )
        self._db.commit()
        return int(cur.lastrowid)

    def recent(self, limit: int = 50) -> List[HistoryEntry]:
        rows = self._db.execute(
            "SELECT id, timestamp, raw_text, formatted_text, app, language, duration, mode"
            " FROM history ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [HistoryEntry(*row) for row in rows]

    def search(self, query: str, limit: int = 50) -> List[HistoryEntry]:
        rows = self._db.execute(
            "SELECT id, timestamp, raw_text, formatted_text, app, language, duration, mode"
            " FROM history WHERE formatted_text LIKE ? OR raw_text LIKE ?"
            " ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [HistoryEntry(*row) for row in rows]

    def delete(self, entry_id: int) -> bool:
        cur = self._db.execute("DELETE FROM history WHERE id = ?", (entry_id,))
        self._db.commit()
        return cur.rowcount > 0

    def clear(self) -> None:
        self._db.execute("DELETE FROM history")
        self._db.commit()

    def stats(self, now: Optional[float] = None) -> Stats:
        now = now or time.time()
        rows = self._db.execute(
            "SELECT timestamp, formatted_text, duration FROM history"
        ).fetchall()
        if not rows:
            return Stats()
        total_words = sum(len(r[1].split()) for r in rows)
        total_audio = sum(r[2] for r in rows)
        wpm = (total_words / (total_audio / 60.0)) if total_audio > 0 else 0.0

        day = 86400
        days_with_activity = {int((r[0]) // day) for r in rows}
        today = int(now // day)
        words_today = sum(len(r[1].split()) for r in rows if int(r[0] // day) == today)
        streak = 0
        d = today
        # streak counts today if active, otherwise starts from yesterday
        if d not in days_with_activity:
            d -= 1
        while d in days_with_activity:
            streak += 1
            d -= 1
        return Stats(
            total_entries=len(rows),
            total_words=total_words,
            total_audio_seconds=total_audio,
            average_wpm=round(wpm, 1),
            streak_days=streak,
            words_today=words_today,
        )

    def close(self) -> None:
        self._db.close()
