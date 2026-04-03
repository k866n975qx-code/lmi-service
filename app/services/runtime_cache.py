from __future__ import annotations

import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock


def db_namespace(conn: sqlite3.Connection) -> str:
    """Return a stable namespace for a SQLite connection."""
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except Exception:
        return f"memory:{id(conn)}"

    paths: list[str] = []
    for row in rows:
        path = row[2] if len(row) > 2 else ""
        if path:
            paths.append(str(Path(path).resolve()))
    if paths:
        return "|".join(sorted(paths))
    return f"memory:{id(conn)}"


class TTLCacheStore:
    """Small in-process TTL cache for local runtime artifacts."""

    def __init__(self, maxsize: int = 256):
        self.maxsize = max(1, int(maxsize))
        self._entries: OrderedDict[tuple, tuple[float, object]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: tuple):
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return None, False
            expires_at, value = entry
            if expires_at <= now:
                self._entries.pop(key, None)
                return None, False
            self._entries.move_to_end(key)
            return value, True

    def set(self, key: tuple, value, ttl_seconds: float):
        expires_at = time.monotonic() + max(1.0, float(ttl_seconds))
        with self._lock:
            self._entries[key] = (expires_at, value)
            self._entries.move_to_end(key)
            self._evict_locked(now=time.monotonic())

    def get_or_set(self, key: tuple, producer, ttl_seconds: float):
        cached, hit = self.get(key)
        if hit:
            return cached
        value = producer()
        self.set(key, value, ttl_seconds)
        return value

    def clear(self):
        with self._lock:
            self._entries.clear()

    def _evict_locked(self, *, now: float):
        expired = [key for key, (expires_at, _) in self._entries.items() if expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
