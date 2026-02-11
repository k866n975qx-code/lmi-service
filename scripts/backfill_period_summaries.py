#!/usr/bin/env python3
"""
Backfill/rebuild period summaries (WEEK, MONTH, QUARTER, YEAR) from existing daily snapshots.
Generates both final summaries for completed periods and rolling summaries for
currently-incomplete periods.

Usage:
    python scripts/backfill_period_summaries.py                        # Only generate missing summaries
    python scripts/backfill_period_summaries.py 2025-01-01             # Missing summaries from date onwards
    python scripts/backfill_period_summaries.py 2025-01-01 2025-12-31  # Missing summaries in date range
    python scripts/backfill_period_summaries.py --rebuild-all          # Rebuild ALL summaries (even existing)
    python scripts/backfill_period_summaries.py --rebuild-all 2025-01-01  # Rebuild all from date onwards
    python scripts/backfill_period_summaries.py --dry-run              # Show what would be done
"""
from pathlib import Path
import os
import sys
from datetime import date, timedelta

# Ensure repo root is on sys.path and is the working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.db import get_conn
from app.config import settings
from app.pipeline.snapshots import _period_bounds, now_utc_iso
from app.pipeline.periods import build_period_snapshot
from app.pipeline.validation import validate_period_snapshot
from app.pipeline.flat_persist import _write_period_flat
import structlog

log = structlog.get_logger()

_PERIOD_TYPES = [
    ("WEEK", "weekly"),
    ("MONTH", "monthly"),
    ("QUARTER", "quarterly"),
    ("YEAR", "yearly"),
]


def backfill_period_summaries(
    conn,
    start_date: str | None = None,
    end_date: str | None = None,
    only_missing: bool = True,
    dry_run: bool = False,
):
    """
    Backfill period summaries by rebuilding them from daily snapshots.

    Generates:
    1. Final summaries for all completed periods
    2. One rolling (to-date) summary for each currently-incomplete period

    Args:
        conn: Database connection
        start_date: Optional start date (YYYY-MM-DD) to limit backfill range
        end_date: Optional end date (YYYY-MM-DD) to limit backfill range
        only_missing: If True, skip periods that already have a final summary.
                      If False (--rebuild-all), delete and regenerate.
        dry_run: If True, log what would be done but don't write.
    """
    cur = conn.cursor()

    # Get all daily snapshot dates
    query = "SELECT as_of_date_local FROM daily_portfolio"
    params = []
    if start_date and end_date:
        query += " WHERE as_of_date_local BETWEEN ? AND ?"
        params = [start_date, end_date]
    elif start_date:
        query += " WHERE as_of_date_local >= ?"
        params = [start_date]
    elif end_date:
        query += " WHERE as_of_date_local <= ?"
        params = [end_date]
    query += " ORDER BY as_of_date_local ASC"

    rows = cur.execute(query, params).fetchall()
    dates = [date.fromisoformat(row[0]) for row in rows]

    if not dates:
        log.info("backfill_no_daily_snapshots")
        return

    # Existing final summaries (for skip check when only_missing)
    existing_finals = set()
    if only_missing:
        existing_rows = cur.execute(
            "SELECT period_type, period_start_date, period_end_date FROM period_summary WHERE is_rolling=0"
        ).fetchall()
        existing_finals = {(row[0], row[1], row[2]) for row in existing_rows}
        log.info("backfill_existing_finals", count=len(existing_finals))

    log.info("backfill_started", total_dates=len(dates), only_missing=only_missing, dry_run=dry_run)

    run_id = "backfill_periods_" + now_utc_iso().replace(":", "").replace("-", "").replace(".", "")[:19]

    processed_periods = set()
    rebuilt_count = 0
    skipped_count = 0
    failed_count = 0

    # ── Phase 1: Final summaries for completed periods ──────────────────
    for dt in dates:
        for period_type, snapshot_type in _PERIOD_TYPES:
            start, end = _period_bounds(period_type, dt)

            # Only build final when dt is the period end date
            if dt != end:
                continue

            period_key = (period_type, str(start), str(end))
            if period_key in processed_periods:
                continue
            processed_periods.add(period_key)

            # Skip if already exists and only_missing mode
            if only_missing and period_key in existing_finals:
                skipped_count += 1
                continue

            if dry_run:
                log.info("dry_run_would_build_final", period=period_type, start=str(start), end=str(end))
                rebuilt_count += 1
                continue

            try:
                snapshot = build_period_snapshot(conn, snapshot_type=snapshot_type, as_of=str(end), mode="final")
                ok, reasons = validate_period_snapshot(snapshot)
                if not ok:
                    log.error("backfill_final_invalid", period=period_type, start=str(start), end=str(end), reasons=reasons)
                    failed_count += 1
                    continue

                # In rebuild mode, _write_period_flat does DELETE then INSERT (idempotent)
                _write_period_flat(conn, snapshot, run_id)
                conn.commit()
                log.info("backfill_final_rebuilt", period=period_type, start=str(start), end=str(end))
                rebuilt_count += 1

            except Exception as exc:
                log.error("backfill_final_failed", period=period_type, start=str(start), end=str(end), err=str(exc))
                failed_count += 1

            if rebuilt_count % 10 == 0 and rebuilt_count > 0:
                log.info("backfill_progress", rebuilt=rebuilt_count, skipped=skipped_count, failed=failed_count)

    # ── Phase 2: Rolling summaries for currently-incomplete periods ──────
    latest_date = dates[-1]
    rolling_count = 0
    for period_type, snapshot_type in _PERIOD_TYPES:
        start, end = _period_bounds(period_type, latest_date)

        # If the latest daily IS the period end, a final summary covers it — no rolling needed
        if latest_date == end:
            continue

        if dry_run:
            log.info("dry_run_would_build_rolling", period=period_type, start=str(start), end=str(latest_date))
            rolling_count += 1
            continue

        try:
            snapshot = build_period_snapshot(conn, snapshot_type=snapshot_type, as_of=str(latest_date), mode="to_date")
            ok, reasons = validate_period_snapshot(snapshot)
            if not ok:
                log.error("backfill_rolling_invalid", period=period_type, start=str(start), reasons=reasons)
                failed_count += 1
                continue

            # _write_period_flat handles delete-before-insert; stores period_type=WEEK + is_rolling=1
            _write_period_flat(conn, snapshot, run_id)
            conn.commit()
            log.info("backfill_rolling_built", period=period_type, start=str(start), end=str(latest_date))
            rolling_count += 1

        except Exception as exc:
            log.error("backfill_rolling_failed", period=period_type, start=str(start), err=str(exc))
            failed_count += 1

    log.info(
        "backfill_complete",
        finals_rebuilt=rebuilt_count,
        rolling_built=rolling_count,
        skipped=skipped_count,
        failed=failed_count,
        total_periods=len(processed_periods),
    )


if __name__ == '__main__':
    args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    flags = [arg for arg in sys.argv[1:] if arg.startswith('--')]

    rebuild_all = '--rebuild-all' in flags
    dry_run = '--dry-run' in flags
    only_missing = not rebuild_all

    start_date = args[0] if len(args) > 0 else None
    end_date = args[1] if len(args) > 1 else None

    mode_label = 'DRY RUN' if dry_run else ('Rebuilding ALL' if rebuild_all else 'Generating MISSING')
    print(f'{mode_label} period summaries from daily snapshots...')

    if start_date and end_date:
        print(f'Date range: {start_date} to {end_date}')
    elif start_date:
        print(f'From: {start_date} onwards')
    else:
        print('All dates')

    conn = get_conn(settings.db_path)
    backfill_period_summaries(conn, start_date=start_date, end_date=end_date, only_missing=only_missing, dry_run=dry_run)
    print('Done.')
