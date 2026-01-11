from collections import deque
import json
from datetime import date
from pathlib import Path
import os
from fastapi import APIRouter, BackgroundTasks, HTTPException
from .schemas import SyncRun, StatusResponse, SyncWindowRequest
from ..pipeline.orchestrator import trigger_sync, trigger_sync_window, get_status, get_snapshot
from ..pipeline.periods import build_period_snapshot, _period_bounds
from ..pipeline.diff_daily import diff_daily_from_db
from ..pipeline.diff_periods import diff_periods_from_db
from ..pipeline.snapshot_views import slim_snapshot
from ..pipeline.null_reasons import replace_nulls_with_reasons
from ..config import settings
from ..db import get_conn
from ..cache_layer import CacheLayer

router = APIRouter()

_PERIOD_MAP = {
    "weekly": "WEEK",
    "monthly": "MONTH",
    "quarterly": "QUARTER",
    "yearly": "YEAR",
}

@router.get(
    '/health',
    summary="Health check",
    description="Returns service and DB connectivity plus last run metadata.",
    tags=["Health"],
)
def health():
    try:
        conn = get_conn(settings.db_path)
        cur = conn.cursor()
        row = cur.execute(
            "SELECT run_id, status, started_at_utc, finished_at_utc FROM runs ORDER BY started_at_utc DESC LIMIT 1"
        ).fetchone()
        last = None
        if row:
            last = {'run_id': row[0], 'status': row[1], 'started_at_utc': row[2], 'finished_at_utc': row[3]}
        return {'ok': True, 'db': 'ok', 'last_run': last}
    except Exception as e:
        raise HTTPException(503, f'db_error: {e}')

@router.post(
    '/cache/{action}',
    summary="Cache admin",
    description="Invalidate or backfill the local cache.",
    tags=["Admin"],
)
def cache_admin(action: str):
    if action not in ('invalidate', 'backfill'):
        raise HTTPException(400, 'action must be invalidate|backfill')
    cache = CacheLayer(settings.cache_dir, settings.cache_db_path, settings.cache_ttl_hours)
    if action == 'invalidate':
        cache.invalidate_all()
        return {'ok': True, 'cleared': True}
    return {'ok': True, 'backfill': 'noop'}

@router.post(
    '/sync-all',
    response_model=SyncRun,
    status_code=202,
    summary="Trigger sync",
    description="Starts the sync pipeline in the background and returns the run_id.",
    tags=["Sync"],
)
def sync_all(background: BackgroundTasks):
    run_id = trigger_sync(background)
    return SyncRun(run_id=run_id)

@router.post(
    '/sync-window',
    response_model=SyncRun,
    status_code=202,
    summary="Trigger sync for LM window",
    description="Pulls LM transactions for a date window (append-only) and runs the pipeline.",
    tags=["Sync"],
)
def sync_window(req: SyncWindowRequest, background: BackgroundTasks):
    try:
        start = date.fromisoformat(req.start_date)
    except ValueError:
        raise HTTPException(400, 'start_date must be YYYY-MM-DD')
    try:
        end = date.fromisoformat(req.end_date)
    except ValueError:
        raise HTTPException(400, 'end_date must be YYYY-MM-DD')
    if start > end:
        raise HTTPException(400, 'start_date must be <= end_date')
    run_id = trigger_sync_window(background, req.start_date, req.end_date)
    return SyncRun(run_id=run_id)

@router.get(
    '/status/{run_id}',
    response_model=StatusResponse,
    summary="Get sync status",
    description="Return status for a given run_id.",
    tags=["Sync"],
)
def status(run_id: str):
    st = get_status(run_id)
    if not st:
        raise HTTPException(404, 'run not found')
    return st

@router.get(
    '/summaries/available',
    summary="List available summaries",
    description=(
        "Lists stored daily dates and period summaries. "
        "Optional snapshot_type filter: daily|weekly|monthly|quarterly|yearly."
    ),
    tags=["Snapshots"],
)
def list_available_period_summaries(snapshot_type: str | None = None):
    conn = get_conn(settings.db_path)
    cur = conn.cursor()
    snapshot_type = snapshot_type.lower() if snapshot_type else None
    if snapshot_type and snapshot_type not in {"daily", "weekly", "monthly", "quarterly", "yearly"}:
        raise HTTPException(400, 'snapshot_type must be daily|weekly|monthly|quarterly|yearly')
    daily_rows = cur.execute(
        "SELECT as_of_date_local FROM snapshot_daily_current ORDER BY as_of_date_local DESC"
    ).fetchall()
    period_rows = cur.execute(
        """
        SELECT period_type, period_start_date, period_end_date, snapshot_id, created_at_utc
        FROM snapshots
        ORDER BY period_end_date DESC
        """
    ).fetchall()
    if snapshot_type == "daily":
        period_rows = []
    elif snapshot_type:
        target = _PERIOD_MAP[snapshot_type]
        period_rows = [row for row in period_rows if row[0] == target]
        daily_rows = []
    return {
        "daily": [row[0] for row in daily_rows],
        "period": [
            {
                "period_type": row[0],
                "start_date": row[1],
                "end_date": row[2],
                "snapshot_id": row[3],
                "created_at_utc": row[4],
            }
            for row in period_rows
        ],
    }

@router.get(
    '/portfolio/daily/latest',
    summary="Get latest daily portfolio",
    description=(
        "Returns the most recent daily portfolio snapshot. "
        "Schema: samples/daily.json. "
        "Use slim=false for full payload."
    ),
    tags=["Snapshots"],
)
def get_latest_daily_portfolio(slim: bool = True):
    conn = get_conn(settings.db_path)
    cur = conn.cursor()
    row = cur.execute(
        "SELECT payload_json FROM snapshot_daily_current ORDER BY as_of_date_local DESC LIMIT 1"
    ).fetchone()
    if not row:
        raise HTTPException(404, 'daily snapshot not found')
    payload = json.loads(row[0])
    payload = slim_snapshot(payload) if slim else payload
    return replace_nulls_with_reasons(payload, kind="daily", conn=conn)

@router.get(
    '/portfolio/daily/{as_of}',
    summary="Get daily portfolio by date",
    description=(
        "Returns a daily portfolio snapshot for the given date. "
        "Schema: samples/daily.json. "
        "Use slim=false for full payload."
    ),
    tags=["Snapshots"],
)
def get_daily_portfolio(as_of: str, slim: bool = True):
    conn = get_conn(settings.db_path)
    cur = conn.cursor()
    row = cur.execute(
        "SELECT payload_json FROM snapshot_daily_current WHERE as_of_date_local=?",
        (as_of,),
    ).fetchone()
    if not row:
        raise HTTPException(404, 'daily snapshot not found')
    payload = json.loads(row[0])
    payload = slim_snapshot(payload) if slim else payload
    return replace_nulls_with_reasons(payload, kind="daily", conn=conn)

@router.get(
    '/period-summary/{kind}/{as_of}',
    summary="Get stored period summary by as-of date",
    description=(
        "Returns a persisted period summary. "
        "Schema: samples/period.json. "
        "Use slim=false for full payload."
    ),
    tags=["Periods"],
)
def get_period_summary(kind: str, as_of: str, slim: bool = True):
    kind = kind.lower()
    if kind not in _PERIOD_MAP:
        raise HTTPException(400, 'kind must be weekly|monthly|quarterly|yearly')
    try:
        as_of_date = date.fromisoformat(as_of)
    except ValueError:
        raise HTTPException(400, 'as_of must be YYYY-MM-DD')
    start, end = _period_bounds(kind, as_of_date)
    snap = get_snapshot(_PERIOD_MAP[kind], start.isoformat(), end.isoformat())
    if not snap:
        raise HTTPException(404, 'snapshot not found')
    snap = slim_snapshot(snap) if slim else snap
    conn = get_conn(settings.db_path)
    return replace_nulls_with_reasons(snap, kind="period", conn=conn)

@router.get(
    '/period-summary/{kind}/{as_of}/{mode}',
    summary="Get period summary (final or to-date)",
    description=(
        "Returns a period summary in final or to_date mode. "
        "Schema: samples/period.json. "
        "Use slim=false for full payload."
    ),
    tags=["Periods"],
)
def get_period_summary_mode(kind: str, as_of: str, mode: str, slim: bool = True):
    kind = kind.lower()
    mode = mode.lower()
    if kind not in ('weekly', 'monthly', 'quarterly', 'yearly'):
        raise HTTPException(400, 'kind must be weekly|monthly|quarterly|yearly')
    if mode not in ('to_date', 'final'):
        raise HTTPException(400, 'mode must be to_date|final')
    conn = get_conn(settings.db_path)
    try:
        if mode == "final":
            try:
                as_of_date = date.fromisoformat(as_of)
            except ValueError:
                raise HTTPException(400, 'as_of must be YYYY-MM-DD')
            start, end = _period_bounds(kind, as_of_date)
            snap = get_snapshot(_PERIOD_MAP[kind], start.isoformat(), end.isoformat())
            if snap:
                snap = slim_snapshot(snap) if slim else snap
                return replace_nulls_with_reasons(snap, kind="period", conn=conn)
        snap = build_period_snapshot(conn, snapshot_type=kind, as_of=as_of, mode=mode)
        snap = slim_snapshot(snap) if slim else snap
        return replace_nulls_with_reasons(snap, kind="period", conn=conn)
    except ValueError as e:
        raise HTTPException(404, str(e))

@router.get(
    '/compare/daily/{left_date}/{right_date}',
    summary="Compare two daily portfolios",
    description="Returns a daily comparison. Schema: samples/diff_daily.json.",
    tags=["Diffs"],
)
def compare_daily_portfolios(left_date: str, right_date: str):
    conn = get_conn(settings.db_path)
    try:
        diff = diff_daily_from_db(conn, left_date, right_date)
        return replace_nulls_with_reasons(diff, kind="diff", conn=conn)
    except ValueError as e:
        raise HTTPException(404, str(e))

@router.get(
    '/compare/period/{kind}/{left_as_of}/{right_as_of}',
    summary="Compare two period summaries",
    description="Returns a period comparison. Schema: samples/diff_period.json.",
    tags=["Diffs"],
)
def compare_period_summaries(kind: str, left_as_of: str, right_as_of: str):
    kind = kind.lower()
    conn = get_conn(settings.db_path)
    try:
        diff = diff_periods_from_db(conn, kind, left_as_of, right_as_of)
        return replace_nulls_with_reasons(diff, kind="diff", conn=conn)
    except ValueError as e:
        raise HTTPException(404, str(e))

@router.get(
    '/logs',
    summary="Read error logs",
    description="Returns the last N lines from the error log file.",
    tags=["Admin"],
)
def read_logs(lines: int = 200):
    if lines < 1:
        raise HTTPException(400, 'lines must be >= 1')
    if lines > 2000:
        lines = 2000
    log_path = os.getenv("LOG_ERROR_FILE", "./data/logs/error.log")
    path = Path(log_path)
    if not path.exists():
        raise HTTPException(404, 'log file not found')
    tail = deque(maxlen=lines)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tail.append(line.rstrip("\n"))
    return {"path": str(path), "lines": list(tail)}
