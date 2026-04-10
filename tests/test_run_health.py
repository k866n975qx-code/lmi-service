from app.api import routes
from app.db import get_conn, migrate
from app.pipeline.locking import acquire_lock
from app.pipeline.utils import mark_run_interrupted, start_run


def test_health_prefers_operational_last_run(monkeypatch, tmp_path):
    db_path = tmp_path / "health.db"
    conn = get_conn(str(db_path))
    migrate(conn)
    conn.execute(
        """
        INSERT INTO runs(run_id, started_at_utc, finished_at_utc, status, error_message)
        VALUES(?,?,?,?,?)
        """,
        (
            "run-success",
            "2026-04-09T12:00:00+00:00",
            "2026-04-09T12:01:00+00:00",
            "succeeded",
            None,
        ),
    )
    conn.execute(
        """
        INSERT INTO runs(run_id, started_at_utc, finished_at_utc, status, error_message)
        VALUES(?,?,?,?,?)
        """,
        (
            "run-maintenance",
            "2026-04-09T13:00:00+00:00",
            "2026-04-09T13:01:00+00:00",
            "failed",
            "Local interruption during provider_actions after successful transaction normalization",
        ),
    )
    monkeypatch.setattr(routes.settings, "db_path", str(db_path))

    payload = routes.health()

    assert payload["last_event_run"]["run_id"] == "run-maintenance"
    assert payload["last_event_run"]["error_message"] == (
        "Local interruption during provider_actions after successful transaction normalization"
    )
    assert payload["last_run"]["run_id"] == "run-success"
    assert payload["last_successful_run"]["run_id"] == "run-success"
    assert payload["last_failed_run"] is None
    assert payload["last_maintenance_run"]["run_id"] == "run-maintenance"


def test_mark_run_interrupted_marks_running_run_and_releases_lock(tmp_path):
    db_path = tmp_path / "interrupt.db"
    conn = get_conn(str(db_path))
    migrate(conn)
    start_run(conn, "run-1")
    assert acquire_lock(conn, "sync", "run-1", ttl_seconds=60, stale_after_seconds=60)

    assert mark_run_interrupted(str(db_path), "run-1", "SIGTERM") is True

    verify = get_conn(str(db_path))
    row = verify.execute(
        "SELECT status, finished_at_utc, error_message FROM runs WHERE run_id=?",
        ("run-1",),
    ).fetchone()
    lock_row = verify.execute("SELECT owner FROM locks WHERE name='sync'").fetchone()

    assert row[0] == "failed"
    assert row[1] is not None
    assert row[2] == "maintenance_interruption:signal:SIGTERM"
    assert lock_row is None
