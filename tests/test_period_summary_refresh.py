import sqlite3
from datetime import date, timedelta

from app.pipeline import snapshots as snapshots_mod


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE period_summary (
            period_type TEXT NOT NULL,
            period_start_date TEXT NOT NULL,
            period_end_date TEXT NOT NULL,
            is_rolling INTEGER NOT NULL,
            built_from_run_id TEXT,
            PRIMARY KEY (period_type, period_start_date, period_end_date, is_rolling)
        )
        """
    )
    return conn


def _insert_period_row(
    conn: sqlite3.Connection,
    period_type: str,
    start: str,
    end: str,
    is_rolling: int,
    built_from_run_id: str,
) -> None:
    conn.execute(
        """
        INSERT INTO period_summary (period_type, period_start_date, period_end_date, is_rolling, built_from_run_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (period_type, start, end, is_rolling, built_from_run_id),
    )
    conn.commit()


def _daily_for(as_of_date: str) -> dict:
    return {"timestamps": {"portfolio_data_as_of_local": as_of_date}}


def test_rolling_rebuilds_when_same_date_but_old_run(monkeypatch):
    conn = _make_conn()
    as_of = "2026-01-15"
    run_id = "run-new"
    _insert_period_row(conn, "MONTH", "2026-01-01", as_of, 1, "run-old")

    writes = []

    def fake_period_bounds(period_type: str, on: date):
        if period_type == "MONTH":
            return date(2026, 1, 1), on + timedelta(days=1)
        return on, on

    def fake_build_period_snapshot(_conn, snapshot_type, as_of, mode):
        return {
            "snapshot_type": snapshot_type,
            "snapshot_mode": mode,
            "period": {"start_date": "2026-01-01", "end_date": as_of},
            "period_summary": {},
        }

    monkeypatch.setattr(snapshots_mod, "_period_bounds", fake_period_bounds)
    monkeypatch.setattr(snapshots_mod, "validate_period_snapshot", lambda _snap: (True, []))
    monkeypatch.setattr(snapshots_mod, "_delete_rolling_children", lambda _cur, _ptype, _start: 0)
    monkeypatch.setattr("app.pipeline.periods.build_period_snapshot", fake_build_period_snapshot)
    monkeypatch.setattr(
        "app.pipeline.flat_persist._write_period_flat",
        lambda _conn, _snapshot, write_run_id: writes.append(write_run_id),
    )

    snapshots_mod.persist_rolling_summaries(conn, run_id, daily={}, as_of_date_local=as_of)
    assert writes == [run_id]


def test_rolling_skips_when_same_date_and_same_run(monkeypatch):
    conn = _make_conn()
    as_of = "2026-01-15"
    run_id = "run-same"
    _insert_period_row(conn, "MONTH", "2026-01-01", as_of, 1, run_id)

    writes = []

    def fake_period_bounds(period_type: str, on: date):
        if period_type == "MONTH":
            return date(2026, 1, 1), on + timedelta(days=1)
        return on, on

    def fake_build_period_snapshot(_conn, snapshot_type, as_of, mode):
        return {
            "snapshot_type": snapshot_type,
            "snapshot_mode": mode,
            "period": {"start_date": "2026-01-01", "end_date": as_of},
            "period_summary": {},
        }

    monkeypatch.setattr(snapshots_mod, "_period_bounds", fake_period_bounds)
    monkeypatch.setattr(snapshots_mod, "validate_period_snapshot", lambda _snap: (True, []))
    monkeypatch.setattr(snapshots_mod, "_delete_rolling_children", lambda _cur, _ptype, _start: 0)
    monkeypatch.setattr("app.pipeline.periods.build_period_snapshot", fake_build_period_snapshot)
    monkeypatch.setattr(
        "app.pipeline.flat_persist._write_period_flat",
        lambda _conn, _snapshot, write_run_id: writes.append(write_run_id),
    )

    snapshots_mod.persist_rolling_summaries(conn, run_id, daily={}, as_of_date_local=as_of)
    assert writes == []


def test_final_rebuilds_existing_period_when_daily_rewritten(monkeypatch):
    conn = _make_conn()
    as_of = "2026-01-31"
    run_id = "run-new"
    _insert_period_row(conn, "MONTH", "2026-01-01", as_of, 0, "run-old")

    writes = []

    def fake_period_bounds(period_type: str, on: date):
        if period_type == "MONTH":
            return date(2026, 1, 1), on
        return on, on + timedelta(days=1)

    def fake_build_period_snapshot(_conn, snapshot_type, as_of, mode):
        return {
            "snapshot_type": snapshot_type,
            "snapshot_mode": mode,
            "period": {"start_date": "2026-01-01", "end_date": as_of},
            "period_summary": {},
        }

    monkeypatch.setattr(snapshots_mod, "_period_bounds", fake_period_bounds)
    monkeypatch.setattr(snapshots_mod, "validate_period_snapshot", lambda _snap: (True, []))
    monkeypatch.setattr(snapshots_mod, "_delete_rolling_children", lambda _cur, _ptype, _start: 0)
    monkeypatch.setattr("app.pipeline.periods.build_period_snapshot", fake_build_period_snapshot)
    monkeypatch.setattr(
        "app.pipeline.flat_persist._write_period_flat",
        lambda _conn, _snapshot, write_run_id: writes.append(write_run_id),
    )

    snapshots_mod.maybe_persist_periodic(
        conn,
        run_id,
        _daily_for(as_of),
        daily_was_written=True,
    )
    assert writes == [run_id]


def test_final_skips_existing_period_when_daily_not_rewritten(monkeypatch):
    conn = _make_conn()
    as_of = "2026-01-31"
    run_id = "run-skip"
    _insert_period_row(conn, "MONTH", "2026-01-01", as_of, 0, "run-old")

    writes = []

    def fake_period_bounds(period_type: str, on: date):
        if period_type == "MONTH":
            return date(2026, 1, 1), on
        return on, on + timedelta(days=1)

    def fake_build_period_snapshot(_conn, snapshot_type, as_of, mode):
        return {
            "snapshot_type": snapshot_type,
            "snapshot_mode": mode,
            "period": {"start_date": "2026-01-01", "end_date": as_of},
            "period_summary": {},
        }

    monkeypatch.setattr(snapshots_mod, "_period_bounds", fake_period_bounds)
    monkeypatch.setattr(snapshots_mod, "validate_period_snapshot", lambda _snap: (True, []))
    monkeypatch.setattr(snapshots_mod, "_delete_rolling_children", lambda _cur, _ptype, _start: 0)
    monkeypatch.setattr("app.pipeline.periods.build_period_snapshot", fake_build_period_snapshot)
    monkeypatch.setattr(
        "app.pipeline.flat_persist._write_period_flat",
        lambda _conn, _snapshot, write_run_id: writes.append(write_run_id),
    )

    snapshots_mod.maybe_persist_periodic(
        conn,
        run_id,
        _daily_for(as_of),
        daily_was_written=False,
    )
    assert writes == []


def test_final_creates_missing_period_even_when_daily_not_rewritten(monkeypatch):
    conn = _make_conn()
    as_of = "2026-01-31"
    run_id = "run-create"

    writes = []

    def fake_period_bounds(period_type: str, on: date):
        if period_type == "MONTH":
            return date(2026, 1, 1), on
        return on, on + timedelta(days=1)

    def fake_build_period_snapshot(_conn, snapshot_type, as_of, mode):
        return {
            "snapshot_type": snapshot_type,
            "snapshot_mode": mode,
            "period": {"start_date": "2026-01-01", "end_date": as_of},
            "period_summary": {},
        }

    monkeypatch.setattr(snapshots_mod, "_period_bounds", fake_period_bounds)
    monkeypatch.setattr(snapshots_mod, "validate_period_snapshot", lambda _snap: (True, []))
    monkeypatch.setattr(snapshots_mod, "_delete_rolling_children", lambda _cur, _ptype, _start: 0)
    monkeypatch.setattr("app.pipeline.periods.build_period_snapshot", fake_build_period_snapshot)
    monkeypatch.setattr(
        "app.pipeline.flat_persist._write_period_flat",
        lambda _conn, _snapshot, write_run_id: writes.append(write_run_id),
    )

    snapshots_mod.maybe_persist_periodic(
        conn,
        run_id,
        _daily_for(as_of),
        daily_was_written=False,
    )
    assert writes == [run_id]
