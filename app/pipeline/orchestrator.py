import json, uuid, time
import structlog
from ..db import get_conn, migrate
from ..config import settings
from .holdings import reconstruct_holdings
from .market import MarketData
from .snapshots import build_daily_snapshot, persist_daily_snapshot, maybe_persist_periodic, diff_payloads
from .facts import upsert_facts_from_sources
from .utils import append_lm_raw, start_run, finish_run_ok, finish_run_fail, get_run_status, ensure_cusip_map
from .transactions import normalize_investment_transactions
from .corporate_actions import load_provider_actions, upsert_lm_dividend_events, symbols_for_actions, estimate_dividend_schedule
from .validation import validate_daily_snapshot
from .locking import acquire_lock, release_lock

log = structlog.get_logger()

def trigger_sync(background) -> str:
    run_id = str(uuid.uuid4())
    background.add_task(_sync_impl, run_id)
    return run_id

def _sync_impl(run_id: str):
    conn = get_conn(settings.db_path)
    migrate(conn)
    start_run(conn, run_id)
    deadline = time.monotonic() + settings.sync_time_budget_seconds
    lock_acquired = False
    if not acquire_lock(conn, "sync", run_id):
        finish_run_fail(conn, run_id, "lock_held")
        return
    lock_acquired = True

    try:
        def _check_deadline(stage: str):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"time_budget_exceeded_{stage}")

        # 1) Pull Lunch Money (append-only raw)
        _check_deadline("before_lm_pull")
        append_lm_raw(conn, run_id, deadline=deadline)
        ensure_cusip_map(conn)
        normalize_investment_transactions(conn, run_id)
        upsert_lm_dividend_events(conn, run_id)
        _check_deadline("before_provider_actions")
        load_provider_actions(conn, run_id, symbols_for_actions(conn), deadline=deadline)
        estimate_dividend_schedule(conn, run_id, window_days=14, ex_window_days=21)

        # 2) Reconstruct holdings purely from LM first
        _check_deadline("before_holdings")
        holdings, symbols = reconstruct_holdings(conn)

        # 3) Load market data for needed symbols (include benchmarks)
        _check_deadline("before_market_data")
        md = MarketData()
        bench_symbols = [settings.benchmark_primary, settings.benchmark_secondary]
        price_symbols = sorted({*symbols, *[s for s in bench_symbols if s]})
        md.load(price_symbols, deadline=deadline)

        # 4) Build daily snapshot (validates internally)
        _check_deadline("before_snapshot_build")
        daily, sources = build_daily_snapshot(conn, holdings, md)

        ok, reasons = validate_daily_snapshot(daily)
        if not ok:
            log.error("daily_validation_failed", run_id=run_id, reasons=reasons)
            finish_run_fail(conn, run_id, "daily_validation_failed")
            return

        wrote_daily = persist_daily_snapshot(conn, daily, run_id)
        if wrote_daily:
            upsert_facts_from_sources(conn, run_id, daily, sources)

        # 6) Persist periodic snapshots if boundary can be closed
        maybe_persist_periodic(conn, run_id, daily)

        finish_run_ok(conn, run_id)
    except Exception as e:
        log.error("sync_failed", run_id=run_id, err=str(e))
        finish_run_fail(conn, run_id, str(e))
        raise
    finally:
        if lock_acquired:
            release_lock(conn, "sync", run_id)

def get_status(run_id: str):
    conn = get_conn(settings.db_path)
    return get_run_status(conn, run_id)

def get_snapshot(period: str, start: str, end: str):
    # TODO: implement retrieval of persisted snapshots per period
    conn = get_conn(settings.db_path)
    cur = conn.cursor()
    cur.execute("SELECT payload_json FROM snapshots WHERE period_type=? AND period_start_date=? AND period_end_date=?", (period, start, end))
    row = cur.fetchone()
    return json.loads(row[0]) if row else None

def diff_snapshots(req):
    conn = get_conn(settings.db_path)
    cur = conn.cursor()
    left = cur.execute("SELECT payload_json FROM snapshots WHERE snapshot_id=?", (req.left_id,)).fetchone()
    right = cur.execute("SELECT payload_json FROM snapshots WHERE snapshot_id=?", (req.right_id,)).fetchone()
    if not left or not right:
        return {"error": "snapshot not found"}
    return diff_payloads(json.loads(left[0]), json.loads(right[0]))
