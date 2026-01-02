import sqlite3
from datetime import datetime, timezone, timedelta

def acquire_lock(
    conn: sqlite3.Connection,
    name: str,
    owner: str,
    ttl_seconds: int = 7200,
    stale_after_seconds: int | None = None,
) -> bool:
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS locks(
      name TEXT PRIMARY KEY,
      owner TEXT NOT NULL,
      acquired_at_utc TEXT NOT NULL,
      expires_at_utc TEXT NOT NULL
    )
    """)
    now = datetime.now(timezone.utc)
    exp = now + timedelta(seconds=ttl_seconds)
    row = cur.execute("SELECT owner, expires_at_utc FROM locks WHERE name=?", (name,)).fetchone()
    if not row:
        cur.execute(
            "INSERT INTO locks(name, owner, acquired_at_utc, expires_at_utc) VALUES(?,?,?,?)",
            (name, owner, now.isoformat(), exp.isoformat()),
        )
        return True
    existing_owner, expires_at_utc = row
    expires = datetime.fromisoformat(expires_at_utc)

    def _owner_is_stale():
        if not existing_owner:
            return True
        try:
            run_row = cur.execute(
                "SELECT status, started_at_utc, finished_at_utc FROM runs WHERE run_id=?",
                (existing_owner,),
            ).fetchone()
        except sqlite3.Error:
            return False
        if not run_row:
            return True
        status, started_at_utc, finished_at_utc = run_row
        if finished_at_utc:
            return True
        if status and str(status).lower() != "running":
            return True
        if stale_after_seconds:
            try:
                started_at = datetime.fromisoformat(started_at_utc)
            except Exception:
                return True
            age_seconds = (now - started_at).total_seconds()
            if age_seconds > stale_after_seconds:
                return True
        return False

    if expires < now or _owner_is_stale():
        cur.execute(
            "UPDATE locks SET owner=?, acquired_at_utc=?, expires_at_utc=? WHERE name=?",
            (owner, now.isoformat(), exp.isoformat(), name),
        )
        return True
    return False

def release_lock(conn: sqlite3.Connection, name: str, owner: str):
    cur = conn.cursor()
    cur.execute("DELETE FROM locks WHERE name=? AND owner=?", (name, owner))
