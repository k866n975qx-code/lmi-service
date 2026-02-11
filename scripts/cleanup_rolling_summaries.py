#!/usr/bin/env python3
"""
Clean up orphaned rolling summaries - removes rolling snapshots for periods that have final snapshots.

Usage:
    python scripts/cleanup_rolling_summaries.py
"""
from pathlib import Path
import os
import sys

# Ensure repo root is on sys.path and is the working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.db import get_conn
from app.config import settings

def cleanup_rolling_summaries():
    """Remove rolling snapshots for periods that have final snapshots (flat period_summary)."""
    conn = get_conn(settings.db_path)
    cur = conn.cursor()

    total_deleted = 0

    for period in ["WEEK", "MONTH", "QUARTER", "YEAR"]:
        # Flat schema: rolling rows have is_rolling=1, same period_type (e.g. WEEK)
        rolling_starts = cur.execute(
            "SELECT DISTINCT period_start_date FROM period_summary WHERE period_type=? AND is_rolling=1",
            (period,),
        ).fetchall()

        print(f"\nChecking {period} rolling snapshots...")
        print(f"  Found {len(rolling_starts)} unique period starts")

        for (start_date,) in rolling_starts:
            # Check if a final snapshot exists for this period start
            final_exists = cur.execute(
                """
                SELECT period_end_date FROM period_summary
                WHERE period_type=? AND period_start_date=? AND is_rolling=0
                LIMIT 1
                """,
                (period, start_date),
            ).fetchone()

            if final_exists:
                # Get rolling (period_end_date)s for this start so we can delete child rows
                rolling_ends = cur.execute(
                    """
                    SELECT period_end_date FROM period_summary
                    WHERE period_type=? AND period_start_date=? AND is_rolling=1
                    """,
                    (period, start_date),
                ).fetchall()
                for (end_date,) in rolling_ends:
                    cur.execute(
                        "DELETE FROM period_risk_stats WHERE period_type=? AND period_start_date=? AND period_end_date=?",
                        (period, start_date, end_date),
                    )
                    cur.execute(
                        "DELETE FROM period_intervals WHERE period_type=? AND period_start_date=? AND period_end_date=?",
                        (period, start_date, end_date),
                    )
                    cur.execute(
                        "DELETE FROM period_holding_changes WHERE period_type=? AND period_start_date=? AND period_end_date=?",
                        (period, start_date, end_date),
                    )
                deleted = cur.execute(
                    "DELETE FROM period_summary WHERE period_type=? AND period_start_date=? AND is_rolling=1",
                    (period, start_date),
                )
                deleted_count = deleted.rowcount
                if deleted_count > 0:
                    conn.commit()
                    total_deleted += deleted_count
                    print(f"  Deleted {deleted_count} rolling snapshots for {start_date} (final exists)")

    print(f"\nTotal rolling snapshots deleted: {total_deleted}")

if __name__ == '__main__':
    print('Cleaning up orphaned rolling summaries...')
    cleanup_rolling_summaries()
    print('Done.')
