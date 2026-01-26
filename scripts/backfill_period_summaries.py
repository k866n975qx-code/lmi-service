#!/usr/bin/env python3
"""
Backfill/rebuild period summaries (WEEK, MONTH, QUARTER, YEAR) from existing daily snapshots.

Useful when:
- Schema changes have caused artifacts in existing period summaries
- You want to regenerate historical period summaries with updated logic
- Data corrections were made to daily snapshots

Usage:
    python scripts/backfill_period_summaries.py                        # Only generate missing summaries
    python scripts/backfill_period_summaries.py 2025-01-01             # Missing summaries from date onwards
    python scripts/backfill_period_summaries.py 2025-01-01 2025-12-31  # Missing summaries in date range
    python scripts/backfill_period_summaries.py --rebuild-all          # Rebuild ALL summaries (even existing)
    python scripts/backfill_period_summaries.py --rebuild-all 2025-01-01  # Rebuild all from date onwards
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
from app.pipeline.snapshots import _period_bounds, sha256_json, now_utc_iso
from app.pipeline.periods import build_period_snapshot
from app.pipeline.validation import validate_period_snapshot
import structlog
import json
import uuid

log = structlog.get_logger()

def backfill_period_summaries(conn, start_date: str | None = None, end_date: str | None = None, only_missing: bool = True):
    """
    Backfill period summaries by rebuilding them from daily snapshots.

    Args:
        conn: Database connection
        start_date: Optional start date (YYYY-MM-DD) to limit backfill range
        end_date: Optional end date (YYYY-MM-DD) to limit backfill range
        only_missing: If True, only generate missing summaries. If False, regenerate all.
    """
    cur = conn.cursor()

    # Get all daily snapshot dates
    query = "SELECT as_of_date_local FROM snapshot_daily_current"
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

    # Get existing period summaries if only_missing is True
    existing_periods = set()
    if only_missing:
        existing_rows = cur.execute(
            "SELECT period_type, period_start_date, period_end_date FROM snapshots"
        ).fetchall()
        existing_periods = {(row[0], row[1], row[2]) for row in existing_rows}
        log.info("backfill_existing_summaries", count=len(existing_periods))

    log.info("backfill_period_summaries_started", total_dates=len(dates), start_date=start_date, end_date=end_date, only_missing=only_missing)

    run_id = "backfill_periods_" + now_utc_iso().replace(":", "").replace("-", "").replace(".", "")[:19]

    # Track which periods we've already processed
    processed_periods = set()
    rebuilt_count = 0
    skipped_count = 0
    failed_count = 0

    for dt in dates:
        # Check each period type to see if this date is a period end
        for period_type, snapshot_type in [
            ("WEEK", "weekly"),
            ("MONTH", "monthly"),
            ("QUARTER", "quarterly"),
            ("YEAR", "yearly"),
        ]:
            start, end = _period_bounds(period_type, dt)

            # Skip if this date is not a period end
            if dt != end:
                continue

            # Skip if we've already processed this period
            period_key = (period_type, str(start), str(end))
            if period_key in processed_periods:
                continue

            processed_periods.add(period_key)

            # Skip if only_missing is True and this period already exists
            if only_missing and period_key in existing_periods:
                log.debug("backfill_period_exists", period=period_type, start=str(start), end=str(end))
                skipped_count += 1
                continue

            try:
                # Build the period snapshot from daily snapshots
                snapshot = build_period_snapshot(
                    conn,
                    snapshot_type=snapshot_type,
                    as_of=str(end),
                    mode="final"
                )

                # Validate the snapshot
                ok, reasons = validate_period_snapshot(snapshot)
                if not ok:
                    log.error("backfill_period_invalid", period=period_type, start=str(start), end=str(end), reasons=reasons)
                    failed_count += 1
                    continue

                # Only check hash if not only_missing (for regeneration mode)
                if not only_missing:
                    payload_sha = sha256_json(snapshot)
                    existing = cur.execute(
                        """
                        SELECT payload_sha256
                        FROM snapshots
                        WHERE period_type=? AND period_start_date=? AND period_end_date=?
                        """,
                        (period_type, str(start), str(end)),
                    ).fetchone()

                    if existing and existing[0] == payload_sha:
                        log.debug("backfill_period_unchanged", period=period_type, start=str(start), end=str(end))
                        skipped_count += 1
                        continue
                else:
                    payload_sha = sha256_json(snapshot)

                # Insert or replace the period snapshot
                cur.execute(
                    """
                    INSERT OR REPLACE INTO snapshots(snapshot_id, period_type, period_start_date, period_end_date, built_from_run_id, payload_json, payload_sha256, created_at_utc)
                    VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (str(uuid.uuid4()), period_type, str(start), str(end), run_id, json.dumps(snapshot), payload_sha, now_utc_iso()),
                )
                conn.commit()

                log.info("backfill_period_rebuilt", period=period_type, start=str(start), end=str(end))
                rebuilt_count += 1

                if rebuilt_count % 10 == 0:
                    log.info("backfill_progress", rebuilt=rebuilt_count, skipped=skipped_count, failed=failed_count)

            except Exception as exc:
                log.error("backfill_period_failed", period=period_type, start=str(start), end=str(end), err=str(exc))
                failed_count += 1
                continue

    log.info("backfill_period_summaries_complete", rebuilt=rebuilt_count, skipped=skipped_count, failed=failed_count, total_processed=len(processed_periods))

if __name__ == '__main__':
    # Parse arguments
    args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    flags = [arg for arg in sys.argv[1:] if arg.startswith('--')]

    rebuild_all = '--rebuild-all' in flags
    only_missing = not rebuild_all

    start_date = args[0] if len(args) > 0 else None
    end_date = args[1] if len(args) > 1 else None

    mode = 'Rebuilding ALL' if rebuild_all else 'Generating MISSING'
    print(f'{mode} period summaries from daily snapshots...')

    if start_date and end_date:
        print(f'Date range: {start_date} to {end_date}')
    elif start_date:
        print(f'From: {start_date} onwards')
    else:
        print('All dates')

    conn = get_conn(settings.db_path)
    backfill_period_summaries(conn, start_date=start_date, end_date=end_date, only_missing=only_missing)
    print('Done.')
