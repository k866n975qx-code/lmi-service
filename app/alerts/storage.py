from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None

DDL_ALERTS = [
    """
    CREATE TABLE IF NOT EXISTS alert_messages (
      id TEXT PRIMARY KEY,
      fingerprint TEXT NOT NULL,
      category TEXT NOT NULL,
      severity INTEGER NOT NULL,
      title TEXT NOT NULL,
      body_html TEXT NOT NULL,
      details_json TEXT,
      status TEXT NOT NULL DEFAULT 'open', -- 'open'|'acked'|'closed'
      period_type TEXT,
      as_of_date_local TEXT NOT NULL,
      created_at_utc TEXT NOT NULL,
      updated_at_utc TEXT NOT NULL,
      first_triggered_at_utc TEXT NOT NULL,
      last_triggered_at_utc TEXT NOT NULL,
      last_notified_at_utc TEXT,
      notification_count INTEGER NOT NULL DEFAULT 0,
      reminder_count INTEGER NOT NULL DEFAULT 0,
      next_reminder_at_utc TEXT,
      sent_at_utc TEXT,
      acked_at_utc TEXT,
      acked_by TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS alert_notifications (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      alert_id TEXT,
      channel TEXT NOT NULL,
      sent_at_utc TEXT NOT NULL,
      success INTEGER NOT NULL,
      error TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS alert_settings (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
    """,
]

def migrate_alerts(conn: sqlite3.Connection):
    cur = conn.cursor()
    for stmt in DDL_ALERTS:
        cur.execute(stmt)
    cols = {r[1] for r in cur.execute("PRAGMA table_info(alert_messages)").fetchall()}
    for col, ddl in [
        ("fingerprint", "ALTER TABLE alert_messages ADD COLUMN fingerprint TEXT NOT NULL DEFAULT ''"),
        ("details_json", "ALTER TABLE alert_messages ADD COLUMN details_json TEXT"),
        ("updated_at_utc", "ALTER TABLE alert_messages ADD COLUMN updated_at_utc TEXT NOT NULL DEFAULT ''"),
        ("first_triggered_at_utc", "ALTER TABLE alert_messages ADD COLUMN first_triggered_at_utc TEXT NOT NULL DEFAULT ''"),
        ("last_triggered_at_utc", "ALTER TABLE alert_messages ADD COLUMN last_triggered_at_utc TEXT NOT NULL DEFAULT ''"),
        ("last_notified_at_utc", "ALTER TABLE alert_messages ADD COLUMN last_notified_at_utc TEXT"),
        ("notification_count", "ALTER TABLE alert_messages ADD COLUMN notification_count INTEGER NOT NULL DEFAULT 0"),
        ("reminder_count", "ALTER TABLE alert_messages ADD COLUMN reminder_count INTEGER NOT NULL DEFAULT 0"),
        ("next_reminder_at_utc", "ALTER TABLE alert_messages ADD COLUMN next_reminder_at_utc TEXT"),
        ("sent_at_utc", "ALTER TABLE alert_messages ADD COLUMN sent_at_utc TEXT"),
        ("acked_by", "ALTER TABLE alert_messages ADD COLUMN acked_by TEXT"),
        ("acked_at_utc", "ALTER TABLE alert_messages ADD COLUMN acked_at_utc TEXT"),
        ("period_type", "ALTER TABLE alert_messages ADD COLUMN period_type TEXT"),
        ("status", "ALTER TABLE alert_messages ADD COLUMN status TEXT NOT NULL DEFAULT 'open'"),
        ("as_of_date_local", "ALTER TABLE alert_messages ADD COLUMN as_of_date_local TEXT NOT NULL DEFAULT ''"),
    ]:
        if col not in cols:
            cur.execute(ddl)
    # Indices (after ensuring columns exist)
    cur.execute("CREATE INDEX IF NOT EXISTS ix_alerts_status_date ON alert_messages(status, as_of_date_local);")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_alerts_fp ON alert_messages(fingerprint);")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_alerts_next_reminder ON alert_messages(next_reminder_at_utc);")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_alert_notifications_time ON alert_notifications(sent_at_utc);")
    conn.commit()

def get_setting(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    cur = conn.cursor()
    row = cur.execute("SELECT value FROM alert_settings WHERE key=?", (key,)).fetchone()
    return row[0] if row else default

def set_setting(conn: sqlite3.Connection, key: str, value: str):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO alert_settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()

def get_open_alert_by_fingerprint(conn: sqlite3.Connection, fingerprint: str):
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT id, severity, last_notified_at_utc, notification_count, next_reminder_at_utc
        FROM alert_messages
        WHERE fingerprint=? AND status='open'
        ORDER BY created_at_utc DESC LIMIT 1
        """,
        (fingerprint,),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "severity": row[1],
        "last_notified_at_utc": row[2],
        "notification_count": row[3],
        "next_reminder_at_utc": row[4],
    }

def get_alert_by_id(conn: sqlite3.Connection, alert_id: str):
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT id, status, severity, last_notified_at_utc
        FROM alert_messages
        WHERE id=?
        """,
        (alert_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "status": row[1],
        "severity": row[2],
        "last_notified_at_utc": row[3],
    }

def create_alert(conn: sqlite3.Connection, alert: dict, now_utc: str):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO alert_messages
          (id, fingerprint, category, severity, title, body_html, details_json, status, period_type,
           as_of_date_local, created_at_utc, updated_at_utc, first_triggered_at_utc, last_triggered_at_utc,
           last_notified_at_utc, notification_count, reminder_count, next_reminder_at_utc)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            alert["id"],
            alert["fingerprint"],
            alert["category"],
            int(alert["severity"]),
            alert["title"],
            alert["body_html"],
            json.dumps(alert.get("details")) if alert.get("details") is not None else None,
            "open",
            alert.get("period_type"),
            alert["as_of_date_local"],
            now_utc,
            now_utc,
            now_utc,
            now_utc,
            None,
            0,
            0,
            None,
        ),
    )
    conn.commit()

def update_alert_on_trigger(
    conn: sqlite3.Connection,
    alert_id: str,
    severity: int,
    title: str,
    body_html: str,
    as_of_date_local: str,
    now_utc: str,
    details: dict | None = None,
):
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE alert_messages
        SET severity=?, title=?, body_html=?, details_json=?, as_of_date_local=?,
            updated_at_utc=?, last_triggered_at_utc=?
        WHERE id=?
        """,
        (
            int(severity),
            title,
            body_html,
            json.dumps(details) if details is not None else None,
            as_of_date_local,
            now_utc,
            now_utc,
            alert_id,
        ),
    )
    conn.commit()

def mark_alert_notified(
    conn: sqlite3.Connection,
    alert_id: str | None,
    channel: str,
    success: bool,
    error: str | None,
    now_utc: str,
    next_reminder_at_utc: str | None = None,
    increment_reminder: bool = False,
):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO alert_notifications(alert_id, channel, sent_at_utc, success, error) VALUES (?,?,?,?,?)",
        (alert_id, channel, now_utc, 1 if success else 0, error),
    )
    if alert_id and success:
        cur.execute(
            """
            UPDATE alert_messages
            SET last_notified_at_utc=?, notification_count=notification_count+1,
                reminder_count=reminder_count+?,
                next_reminder_at_utc=?,
                sent_at_utc=?
            WHERE id=?
            """,
            (now_utc, 1 if increment_reminder else 0, next_reminder_at_utc, now_utc, alert_id),
        )
    conn.commit()

def list_open_alerts(conn: sqlite3.Connection, min_severity: int | None = None, max_severity: int | None = None):
    cur = conn.cursor()
    clauses = ["status='open'"]
    params: list = []
    if min_severity is not None:
        clauses.append("severity>=?")
        params.append(int(min_severity))
    if max_severity is not None:
        clauses.append("severity<=?")
        params.append(int(max_severity))
    where = " AND ".join(clauses)
    rows = cur.execute(
        f"""
        SELECT id, category, severity, title, body_html, as_of_date_local, last_notified_at_utc
        FROM alert_messages
        WHERE {where}
        ORDER BY severity DESC, as_of_date_local DESC
        """,
        params,
    ).fetchall()
    return [
        {
            "id": r[0],
            "category": r[1],
            "severity": r[2],
            "title": r[3],
            "body_html": r[4],
            "as_of_date_local": r[5],
            "last_notified_at_utc": r[6],
        }
        for r in rows
    ]

def list_due_reminders(conn: sqlite3.Connection, now_utc: str, min_severity: int):
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT id, category, severity, title, body_html, next_reminder_at_utc, reminder_count, first_triggered_at_utc
        FROM alert_messages
        WHERE status='open' AND severity>=? AND next_reminder_at_utc IS NOT NULL AND next_reminder_at_utc<=?
        ORDER BY next_reminder_at_utc ASC
        """,
        (int(min_severity), now_utc),
    ).fetchall()
    return [
        {
            "id": r[0],
            "category": r[1],
            "severity": r[2],
            "title": r[3],
            "body_html": r[4],
            "next_reminder_at_utc": r[5],
            "reminder_count": r[6],
            "first_triggered_at_utc": r[7],
        }
        for r in rows
    ]

def ack_alert(conn: sqlite3.Connection, alert_id: str, who: str | None = None) -> bool:
    cur = conn.cursor()
    cur.execute(
        "UPDATE alert_messages SET status='acked', acked_at_utc=?, acked_by=? WHERE id=? AND status='open'",
        (now_utc_iso(), who, alert_id),
    )
    conn.commit()
    return cur.rowcount > 0

def close_stale_alerts(conn: sqlite3.Connection, days: int = 7) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.isoformat()
    cur = conn.cursor()
    cur.execute(
        "UPDATE alert_messages SET status='closed', updated_at_utc=? WHERE status='open' AND last_triggered_at_utc < ?",
        (now_utc_iso(), cutoff_iso),
    )
    conn.commit()
    return cur.rowcount

def count_notifications_since(conn: sqlite3.Connection, since_utc: str) -> int:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT COUNT(*) FROM alert_notifications WHERE sent_at_utc >= ? AND success=1",
        (since_utc,),
    ).fetchone()
    return int(row[0]) if row else 0

def alert_trend_data(conn: sqlite3.Connection, days: int = 30, category: str | None = None) -> list[dict]:
    """Get daily alert counts grouped by date and category for trend analysis."""
    cur = conn.cursor()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    if category:
        rows = cur.execute(
            """
            SELECT as_of_date_local, category, COUNT(*) AS cnt, MAX(severity) AS max_sev
            FROM alert_messages
            WHERE created_at_utc >= ? AND category = ?
            GROUP BY as_of_date_local, category
            ORDER BY as_of_date_local
            """,
            (cutoff, category),
        ).fetchall()
    else:
        rows = cur.execute(
            """
            SELECT as_of_date_local, category, COUNT(*) AS cnt, MAX(severity) AS max_sev
            FROM alert_messages
            WHERE created_at_utc >= ?
            GROUP BY as_of_date_local, category
            ORDER BY as_of_date_local
            """,
            (cutoff,),
        ).fetchall()
    return [{"date": r[0], "category": r[1], "count": r[2], "max_severity": r[3]} for r in rows]
