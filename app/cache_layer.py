import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timezone

def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _hash_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

class CacheLayer:
    """
    Append-friendly TTL cache for provider responses.
    - Index in SQLite (cache_index)
    - Payloads stored as JSON files on disk
    """
    def __init__(self, root_dir: str = ".cache", db_path: str = "./data/cache.sqlite3", default_ttl_hours: int = 24):
        self.root_dir = Path(root_dir)
        self.db_path = db_path
        self.default_ttl_hours = int(default_ttl_hours)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS cache_index(
            cache_key TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            first_seen_utc TEXT NOT NULL,
            last_updated_utc TEXT NOT NULL,
            ttl_hours INTEGER NOT NULL
        )
        """)
        conn.close()

    def _conn(self):
        return sqlite3.connect(self.db_path, isolation_level=None)

    def make_key(self, provider: str, endpoint: str, symbol: str, start: str = "", end: str = "", params: dict | None = None):
        params_hash = _hash_obj(params or {})
        return f"{provider}|{endpoint}|{symbol}|{start}|{end}|{params_hash}"

    def _path_for(self, cache_key: str) -> Path:
        # shard by sha prefix
        h = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        sub = self.root_dir / h[:2] / h[2:4]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{h}.json"

    def get(self, cache_key: str):
        conn = self._conn()
        row = conn.execute("SELECT path, last_updated_utc, ttl_hours FROM cache_index WHERE cache_key=?", (cache_key,)).fetchone()
        conn.close()
        if not row:
            return None, None
        path, last_updated_utc, ttl_hours = row
        path = Path(path)
        try:
            from datetime import datetime, timezone
            last_dt = datetime.fromisoformat(last_updated_utc.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600.0
            if age > float(ttl_hours):
                return None, age
            data = json.loads(path.read_text(encoding="utf-8"))
            return data, age
        except Exception:
            return None, None

    def set(self, cache_key: str, payload: dict, ttl_hours: int | None = None):
        ttl = int(ttl_hours or self.default_ttl_hours)
        path = self._path_for(cache_key)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        conn = self._conn()
        conn.execute("""
        INSERT INTO cache_index(cache_key, path, first_seen_utc, last_updated_utc, ttl_hours)
        VALUES(?,?,?,?,?)
        ON CONFLICT(cache_key) DO UPDATE SET path=excluded.path, last_updated_utc=excluded.last_updated_utc, ttl_hours=excluded.ttl_hours
        """, (cache_key, str(path), _utc_now(), _utc_now(), ttl))
        conn.close()

    def fetch(self, cache_key: str, fetch_fn, merge_fn=None, ttl_hours: int | None = None):
        data, age = self.get(cache_key)
        if data is not None:
            return data, True, age
        fresh = fetch_fn()
        if fresh is None:
            return None, False, age
        if merge_fn and data:
            fresh = merge_fn(data, fresh)
        self.set(cache_key, fresh, ttl_hours)
        return fresh, False, 0.0

    def invalidate_all(self):
        conn = self._conn()
        for (path,) in conn.execute("SELECT path FROM cache_index").fetchall():
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
        conn.execute("DELETE FROM cache_index")
        conn.close()
