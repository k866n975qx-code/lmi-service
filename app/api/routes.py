from collections import deque
from datetime import date
from pathlib import Path
import os
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from .schemas import SyncRun, StatusResponse, SyncWindowRequest
from ..pipeline.orchestrator import trigger_sync, trigger_sync_window, get_status
from ..pipeline.periods import _period_bounds
from ..pipeline.diff_daily import diff_daily_from_db
from ..pipeline.diff_periods import diff_periods_from_db
from ..pipeline.snapshot_views import assemble_daily_snapshot, assemble_period_snapshot
from ..pipeline.null_reasons import replace_nulls_with_reasons
from ..config import settings
from ..db import get_conn
from ..cache_layer import CacheLayer
from ..services.telegram import TelegramClient, send_goal_tiers_to_telegram

router = APIRouter()

_PERIOD_MAP = {
    "weekly": "WEEK",
    "monthly": "MONTH",
    "quarterly": "QUARTER",
    "yearly": "YEAR",
}

# Rolling summaries use the same period_type (WEEK, MONTH, etc.) with is_rolling=1

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
        "SELECT as_of_date_local FROM daily_portfolio ORDER BY as_of_date_local DESC"
    ).fetchall()
    period_rows = cur.execute(
        """
        SELECT period_type, period_start_date, period_end_date, built_from_run_id, created_at_utc
        FROM period_summary
        WHERE is_rolling = 0
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
        "Use slim=false for full payload. "
        "Use apply_null_reasons=false to return raw nulls instead of placeholder reason strings."
    ),
    tags=["Snapshots"],
)
def get_latest_daily_portfolio(
    slim: bool = True,
    apply_null_reasons: bool = Query(True, description="If false, return raw nulls; no placeholder reason strings."),
):
    conn = get_conn(settings.db_path)
    payload = assemble_daily_snapshot(conn, as_of_date=None, slim=slim)
    if not payload:
        raise HTTPException(404, 'daily snapshot not found')
    if not apply_null_reasons:
        return payload
    return replace_nulls_with_reasons(payload, kind="daily", conn=conn)

@router.get(
    '/portfolio/daily/{as_of}',
    summary="Get daily portfolio by date",
    description=(
        "Returns a daily portfolio snapshot for the given date. "
        "Schema: samples/daily.json. "
        "Use slim=false for full payload. "
        "Use apply_null_reasons=false to return raw nulls instead of placeholder reason strings."
    ),
    tags=["Snapshots"],
)
def get_daily_portfolio(
    as_of: str,
    slim: bool = True,
    apply_null_reasons: bool = Query(True, description="If false, return raw nulls; no placeholder reason strings."),
):
    conn = get_conn(settings.db_path)
    payload = assemble_daily_snapshot(conn, as_of_date=as_of, slim=slim)
    if not payload:
        raise HTTPException(404, 'daily snapshot not found')
    if not apply_null_reasons:
        return payload
    return replace_nulls_with_reasons(payload, kind="daily", conn=conn)

@router.get(
    '/period-summary/{kind}/latest',
    summary="Get latest persisted period summary",
    description=(
        "Returns the most recent completed period summary (final snapshot). "
        "Example: /period-summary/weekly/latest returns the last completed week. "
        "Schema: samples/period.json. "
        "Use slim=false for full payload."
    ),
    tags=["Periods"],
)
def get_latest_period_summary(kind: str):
    kind = kind.lower()
    if kind not in ('weekly', 'monthly', 'quarterly', 'yearly'):
        raise HTTPException(400, 'kind must be weekly|monthly|quarterly|yearly')

    conn = get_conn(settings.db_path)
    cur = conn.cursor()
    period_type = _PERIOD_MAP[kind]
    row = cur.execute(
        """
        SELECT period_start_date, period_end_date
        FROM period_summary
        WHERE period_type = ? AND is_rolling = 0
        ORDER BY period_end_date DESC
        LIMIT 1
        """,
        (period_type,),
    ).fetchone()

    if not row:
        raise HTTPException(404, f'no {kind} snapshots found')

    snap = assemble_period_snapshot(conn, period_type, row[1], period_start_date=row[0], rolling=False)
    if not snap:
        raise HTTPException(404, f'no {kind} snapshots found')
    # Period snapshots are already summary-level; slim_snapshot is for daily V5 shape only
    return replace_nulls_with_reasons(snap, kind="period", conn=conn)

@router.get(
    '/period-summary/{kind}/rolling',
    summary="Get current rolling period summary",
    description=(
        "Returns the current incomplete period summary (rolling snapshot). "
        "Example: /period-summary/monthly/rolling returns month-to-date. "
        "Schema: samples/period.json. "
    ),
    tags=["Periods"],
)
def get_rolling_period_summary(kind: str):
    kind = kind.lower()
    if kind not in ('weekly', 'monthly', 'quarterly', 'yearly'):
        raise HTTPException(400, 'kind must be weekly|monthly|quarterly|yearly')

    conn = get_conn(settings.db_path)
    cur = conn.cursor()
    period_type = _PERIOD_MAP[kind]
    row = cur.execute(
        """
        SELECT period_start_date, period_end_date
        FROM period_summary
        WHERE period_type = ? AND is_rolling = 1
        ORDER BY period_end_date DESC
        LIMIT 1
        """,
        (period_type,),
    ).fetchone()

    if not row:
        raise HTTPException(404, f'no rolling {kind} snapshot found')

    snap = assemble_period_snapshot(conn, period_type, row[1], period_start_date=row[0], rolling=True)
    if not snap:
        raise HTTPException(404, f'no rolling {kind} snapshot found')
    return replace_nulls_with_reasons(snap, kind="period", conn=conn)

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
def get_period_summary(kind: str, as_of: str):
    kind = kind.lower()
    if kind not in _PERIOD_MAP:
        raise HTTPException(400, 'kind must be weekly|monthly|quarterly|yearly')
    try:
        as_of_date = date.fromisoformat(as_of)
    except ValueError:
        raise HTTPException(400, 'as_of must be YYYY-MM-DD')
    start, end = _period_bounds(kind, as_of_date)
    conn = get_conn(settings.db_path)
    snap = assemble_period_snapshot(conn, _PERIOD_MAP[kind], end.isoformat(), period_start_date=start.isoformat(), rolling=False)
    if not snap:
        raise HTTPException(404, 'snapshot not found')
    return replace_nulls_with_reasons(snap, kind="period", conn=conn)

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

@router.get(
    '/goals/tiers',
    summary="Get dividend goal tiers",
    description=(
        "Returns comprehensive dividend goal tracking across 5 tiers from most conservative "
        "(no action) to most optimistic (max contributions, DRIP, price appreciation, LTV maintained)."
    ),
    tags=["Goals"],
)
def get_goal_tiers():
    conn = get_conn(settings.db_path)
    payload = assemble_daily_snapshot(conn, as_of_date=None)
    if not payload:
        raise HTTPException(404, 'daily snapshot not found')
    goal_tiers = payload.get("goals") or payload.get("goal_tiers")
    if not goal_tiers:
        raise HTTPException(404, 'goal tiers not available')
    return goal_tiers

@router.post(
    '/goals/tiers/send-telegram',
    summary="Send goal tiers to Telegram",
    description="Sends formatted goal tier analysis to the configured Telegram chat.",
    tags=["Goals"],
)
def send_goal_tiers_telegram():
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        raise HTTPException(400, 'telegram not configured')

    conn = get_conn(settings.db_path)
    payload = assemble_daily_snapshot(conn, as_of_date=None)
    if not payload:
        raise HTTPException(404, 'daily snapshot not found')
    goal_tiers = payload.get("goals") or payload.get("goal_tiers")
    if not goal_tiers:
        raise HTTPException(404, 'goal tiers not available')

    telegram_client = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    success = send_goal_tiers_to_telegram(goal_tiers, telegram_client)

    if not success:
        raise HTTPException(500, 'failed to send telegram message')

    return {"ok": True, "sent": True}
