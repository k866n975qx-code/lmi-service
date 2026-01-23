import json, uuid, time, os
import structlog
from ..db import get_conn, migrate
from ..config import settings
from .holdings import reconstruct_holdings
from .market import MarketData
from .snapshots import build_daily_snapshot, persist_daily_snapshot, maybe_persist_periodic, persist_rolling_summaries
from ..alerts.evaluator import evaluate_alerts
from ..alerts.storage import migrate_alerts as migrate_alerts_table
from ..alerts.notifier import send_alerts_sync
from ..services.telegram import TelegramClient
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

def trigger_sync_window(background, start_date: str, end_date: str) -> str:
    run_id = str(uuid.uuid4())
    background.add_task(_sync_impl, run_id, lm_start=start_date, lm_end=end_date)
    return run_id

def _sync_impl(run_id: str, lm_start: str | None = None, lm_end: str | None = None):
    conn = get_conn(settings.db_path)
    migrate(conn)
    cur = conn.cursor()
    run_count = cur.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    first_run = run_count == 0
    verbose = first_run or os.getenv("LOG_VERBOSE", "").lower() in ("1", "true", "yes")
    start_run(conn, run_id)
    deadline = time.monotonic() + settings.sync_time_budget_seconds
    log.info(
        "sync_started",
        run_id=run_id,
        time_budget_seconds=settings.sync_time_budget_seconds,
        first_run=first_run,
        verbose=verbose,
    )
    lock_acquired = False
    lock_ttl = max(300, int(settings.sync_time_budget_seconds or 0) + 300)
    if not acquire_lock(conn, "sync", run_id, ttl_seconds=lock_ttl, stale_after_seconds=lock_ttl):
        finish_run_fail(conn, run_id, "lock_held")
        return
    lock_acquired = True

    try:
        def _check_deadline(stage: str):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"time_budget_exceeded_{stage}")

        def _log_detail(event: str, **fields):
            if verbose:
                log.info(event, run_id=run_id, **fields)
            else:
                log.debug(event, run_id=run_id, **fields)

        def _step_start(step: str):
            log.info("sync_step_start", run_id=run_id, step=step)
            return time.monotonic()

        def _step_done(step: str, started: float, **fields):
            log.info(
                "sync_step_done",
                run_id=run_id,
                step=step,
                elapsed_sec=round(time.monotonic() - started, 2),
                **fields,
            )

        # 1) Pull Lunch Money (append-only raw)
        started = _step_start("lunchmoney_pull")
        _check_deadline("before_lm_pull")
        lm_result = append_lm_raw(
            conn,
            run_id,
            deadline=deadline,
            start_date=lm_start,
            end_date=lm_end,
            append_only=True,
        )
        _step_done("lunchmoney_pull", started)
        _log_detail("lunchmoney_pull_detail", **(lm_result or {}))

        started = _step_start("normalize_transactions")
        ensure_cusip_map(conn)
        tx_upserted, new_tx_count = normalize_investment_transactions(conn, run_id)
        _step_done("normalize_transactions", started, inserted=tx_upserted, new=new_tx_count)

        started = _step_start("lm_dividends_upsert")
        lm_div_count = upsert_lm_dividend_events(conn, run_id)
        _step_done("lm_dividends_upsert", started, inserted=lm_div_count)

        started = _step_start("provider_actions")
        _check_deadline("before_provider_actions")
        action_symbols = symbols_for_actions(conn)
        provider_counts = load_provider_actions(conn, run_id, action_symbols, deadline=deadline)
        _step_done(
            "provider_actions",
            started,
            symbols_count=len(action_symbols),
            **(provider_counts or {}),
        )

        started = _step_start("dividend_schedule")
        matched = estimate_dividend_schedule(conn, run_id, window_days=14, ex_window_days=21)
        _step_done("dividend_schedule", started, matched=matched)

        # 2) Reconstruct holdings purely from LM first
        started = _step_start("reconstruct_holdings")
        _check_deadline("before_holdings")
        holdings, symbols = reconstruct_holdings(conn)
        _step_done(
            "reconstruct_holdings",
            started,
            holdings_count=len(holdings),
            symbols_count=len(symbols),
        )

        # 3) Load market data for needed symbols (include benchmarks)
        started = _step_start("market_data")
        _check_deadline("before_market_data")
        md = MarketData()
        bench_symbols = [settings.benchmark_primary, settings.benchmark_secondary]
        price_symbols = sorted({*symbols, *[s for s in bench_symbols if s]})
        md.load(price_symbols, deadline=deadline)
        md.load_quotes(price_symbols, deadline=deadline)
        _step_done("market_data", started, symbols_count=len(price_symbols))

        # 4) Build daily snapshot (validates internally)
        started = _step_start("build_daily_snapshot")
        _check_deadline("before_snapshot_build")
        daily, sources = build_daily_snapshot(conn, holdings, md)
        _step_done("build_daily_snapshot", started, sources_count=len(sources))

        started = _step_start("persist_daily_snapshot")
        as_of_date_local = daily.get("as_of_date_local") or daily["as_of"][:10]

        def _market_value(payload: dict | None):
            if not isinstance(payload, dict):
                return None
            try:
                return round(float((payload.get("totals") or {}).get("market_value")), 2)
            except (TypeError, ValueError):
                return None

        existing_row = cur.execute(
            "SELECT payload_json FROM snapshot_daily_current WHERE as_of_date_local=?",
            (as_of_date_local,),
        ).fetchone()
        existing_payload = None
        if existing_row and existing_row[0]:
            try:
                existing_payload = json.loads(existing_row[0])
            except json.JSONDecodeError:
                existing_payload = None
        has_daily = existing_payload is not None
        force_daily = bool(has_daily and new_tx_count > 0)
        current_mv = _market_value(daily)
        existing_mv = _market_value(existing_payload)
        market_value_changed = False
        if has_daily:
            if existing_mv is None or current_mv is None:
                market_value_changed = True
            else:
                market_value_changed = existing_mv != current_mv
        prices_as_of_changed = False
        if has_daily:
            existing_prices_as_of = (existing_payload or {}).get("prices_as_of_utc") or (existing_payload or {}).get("prices_as_of")
            current_prices_as_of = daily.get("prices_as_of_utc") or daily.get("prices_as_of")
            prices_as_of_changed = existing_prices_as_of != current_prices_as_of

        should_persist_daily = force_daily or not has_daily or market_value_changed or prices_as_of_changed

        if should_persist_daily:
            ok, reasons = validate_daily_snapshot(daily)
            if not ok:
                log.error("daily_validation_failed", run_id=run_id, reasons=reasons)
                finish_run_fail(conn, run_id, "daily_validation_failed")
                return
            wrote_daily = persist_daily_snapshot(conn, daily, run_id, force=force_daily)
            _step_done(
                "persist_daily_snapshot",
                started,
                wrote_daily=wrote_daily,
                forced=force_daily,
                new_transactions=new_tx_count,
                existing_daily=has_daily,
                market_value_changed=market_value_changed,
                prices_as_of_changed=prices_as_of_changed,
            )
            if wrote_daily:
                upsert_facts_from_sources(conn, run_id, daily, sources)
                # Update rolling summaries (week-to-date, month-to-date, etc.)
                try:
                    persist_rolling_summaries(conn, run_id, daily)
                except Exception as rolling_err:
                    log.warning("rolling_summaries_failed", run_id=run_id, err=str(rolling_err))
                # Alert evaluation + immediate notifications (dedup handles repeats)
                try:
                    migrate_alerts_table(conn)
                    alerts = evaluate_alerts(conn)
                    if alerts and getattr(settings, "telegram_bot_token", None) and getattr(settings, "telegram_chat_id", None):
                        tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
                        send_alerts_sync(conn, alerts, tg)
                except Exception as alert_err:
                    log.warning("alerts_eval_failed", run_id=run_id, err=str(alert_err))
        else:
            _step_done(
                "persist_daily_snapshot",
                started,
                wrote_daily=False,
                skipped=True,
                new_transactions=new_tx_count,
                existing_daily=has_daily,
            )

        # 6) Persist periodic snapshots if boundary can be closed
        started = _step_start("persist_periodic_snapshots")
        maybe_persist_periodic(conn, run_id, daily)
        _step_done("persist_periodic_snapshots", started)

        finish_run_ok(conn, run_id)
        log.info("sync_finished", run_id=run_id, status="succeeded")
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
