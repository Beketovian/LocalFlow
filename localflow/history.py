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
                mode TEXT DEFAULT 'dictation',
                words INTEGER
            )"""
        )
        # Word counts used to be recomputed from every row's text on each
        # stats() call - O(everything) 6x/minute from the dashboard. They are
        # stored per row now; older databases get the column backfilled once.
        cols = {row[1] for row in self._db.execute("PRAGMA table_info(history)")}
        if "words" not in cols:
            self._db.execute("ALTER TABLE history ADD COLUMN words INTEGER")
        rows = self._db.execute(
            "SELECT id, formatted_text FROM history WHERE words IS NULL"
        ).fetchall()
        if rows:
            self._db.executemany(
                "UPDATE history SET words = ? WHERE id = ?",
                [(len(text.split()), rid) for rid, text in rows],
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
            "INSERT INTO history (timestamp, raw_text, formatted_text, app, language, duration, mode, words)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp or time.time(), raw_text, formatted_text, app, language,
             duration, mode, len(formatted_text.split())),
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
        total_entries, total_words, total_audio = self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(words), 0), COALESCE(SUM(duration), 0)"
            " FROM history"
        ).fetchone()
        if not total_entries:
            return Stats()
        wpm = (total_words / (total_audio / 60.0)) if total_audio > 0 else 0.0

        day = 86400
        today = int(now // day)
        days_with_activity = {
            row[0] for row in self._db.execute(
                "SELECT DISTINCT CAST(timestamp / 86400 AS INTEGER) FROM history"
            )
        }
        words_today = self._db.execute(
            "SELECT COALESCE(SUM(words), 0) FROM history"
            " WHERE timestamp >= ? AND timestamp < ?",
            (today * day, (today + 1) * day),
        ).fetchone()[0]
        streak = 0
        d = today
        # streak counts today if active, otherwise starts from yesterday
        if d not in days_with_activity:
            d -= 1
        while d in days_with_activity:
            streak += 1
            d -= 1
        return Stats(
            total_entries=total_entries,
            total_words=total_words,
            total_audio_seconds=total_audio,
            average_wpm=round(wpm, 1),
            streak_days=streak,
            words_today=words_today,
        )

    def close(self) -> None:
        self._db.close()
