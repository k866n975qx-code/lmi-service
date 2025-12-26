from datetime import date
from fastapi import APIRouter, BackgroundTasks, HTTPException
from .schemas import SyncRun, DiffRequest, StatusResponse
from ..pipeline.orchestrator import trigger_sync, get_status, get_snapshot, diff_snapshots
from ..pipeline.periods import build_period_snapshot, _period_bounds
from ..pipeline.diff_daily import diff_daily_from_db
from ..pipeline.diff_periods import diff_periods_from_db
from ..pipeline.snapshot_views import slim_snapshot
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

@router.get('/health')
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

@router.post('/cache/{action}')
def cache_admin(action: str):
    if action not in ('invalidate', 'backfill'):
        raise HTTPException(400, 'action must be invalidate|backfill')
    cache = CacheLayer(settings.cache_dir, settings.cache_db_path, settings.cache_ttl_hours)
    if action == 'invalidate':
        cache.invalidate_all()
        return {'ok': True, 'cleared': True}
    return {'ok': True, 'backfill': 'noop'}

@router.post('/sync-all', response_model=SyncRun, status_code=202)
def sync_all(background: BackgroundTasks):
    run_id = trigger_sync(background)
    return SyncRun(run_id=run_id)

@router.get('/status/{run_id}', response_model=StatusResponse)
def status(run_id: str):
    st = get_status(run_id)
    if not st:
        raise HTTPException(404, 'run not found')
    return st

@router.get('/snapshots/{period}/{start}/{end}')
def snapshots(period: str, start: str, end: str, slim: bool = False):
    snap = get_snapshot(period.upper(), start, end)
    if not snap:
        raise HTTPException(404, 'snapshot not found')
    return slim_snapshot(snap) if slim else snap

@router.get('/snapshots/available')
def snapshots_available():
    conn = get_conn(settings.db_path)
    cur = conn.cursor()
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

@router.get('/period/{snapshot_type}/{as_of}')
def period_snapshot_stored(snapshot_type: str, as_of: str, slim: bool = False):
    snapshot_type = snapshot_type.lower()
    if snapshot_type not in _PERIOD_MAP:
        raise HTTPException(400, 'snapshot_type must be weekly|monthly|quarterly|yearly')
    try:
        as_of_date = date.fromisoformat(as_of)
    except ValueError:
        raise HTTPException(400, 'as_of must be YYYY-MM-DD')
    start, end = _period_bounds(snapshot_type, as_of_date)
    snap = get_snapshot(_PERIOD_MAP[snapshot_type], start.isoformat(), end.isoformat())
    if not snap:
        raise HTTPException(404, 'snapshot not found')
    return slim_snapshot(snap) if slim else snap

@router.get('/period/{snapshot_type}/{as_of}/{mode}')
def period_snapshot(snapshot_type: str, as_of: str, mode: str, slim: bool = False):
    snapshot_type = snapshot_type.lower()
    mode = mode.lower()
    if snapshot_type not in ('weekly', 'monthly', 'quarterly', 'yearly'):
        raise HTTPException(400, 'snapshot_type must be weekly|monthly|quarterly|yearly')
    if mode not in ('to_date', 'final'):
        raise HTTPException(400, 'mode must be to_date|final')
    conn = get_conn(settings.db_path)
    try:
        if mode == "final":
            try:
                as_of_date = date.fromisoformat(as_of)
            except ValueError:
                raise HTTPException(400, 'as_of must be YYYY-MM-DD')
            start, end = _period_bounds(snapshot_type, as_of_date)
            snap = get_snapshot(_PERIOD_MAP[snapshot_type], start.isoformat(), end.isoformat())
            if snap:
                return slim_snapshot(snap) if slim else snap
        snap = build_period_snapshot(conn, snapshot_type=snapshot_type, as_of=as_of, mode=mode)
        return slim_snapshot(snap) if slim else snap
    except ValueError as e:
        raise HTTPException(404, str(e))

@router.post('/diff')
def diff(req: DiffRequest):
    if not req.left_id or not req.right_id:
        raise HTTPException(400, 'left_id and right_id required')
    return diff_snapshots(req)

@router.get('/diff/daily/{left_date}/{right_date}')
def diff_daily(left_date: str, right_date: str):
    conn = get_conn(settings.db_path)
    try:
        return diff_daily_from_db(conn, left_date, right_date)
    except ValueError as e:
        raise HTTPException(404, str(e))

@router.get('/diff/period/{snapshot_type}/{left_as_of}/{right_as_of}')
def diff_period(snapshot_type: str, left_as_of: str, right_as_of: str):
    snapshot_type = snapshot_type.lower()
    conn = get_conn(settings.db_path)
    try:
        return diff_periods_from_db(conn, snapshot_type, left_as_of, right_as_of)
    except ValueError as e:
        raise HTTPException(404, str(e))
