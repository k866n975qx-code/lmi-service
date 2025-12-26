import sqlite3
from datetime import datetime, timezone, timedelta

def acquire_lock(conn: sqlite3.Connection, name: str, owner: str, ttl_seconds: int = 7200) -> bool:
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
    expires = datetime.fromisoformat(row[1])
    if expires < now:
        cur.execute(
            "UPDATE locks SET owner=?, acquired_at_utc=?, expires_at_utc=? WHERE name=?",
            (owner, now.isoformat(), exp.isoformat(), name),
        )
        return True
    return False

def release_lock(conn: sqlite3.Connection, name: str, owner: str):
    cur = conn.cursor()
    cur.execute("DELETE FROM locks WHERE name=? AND owner=?", (name, owner))
