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
    """Remove rolling snapshots for periods that have final snapshots."""
    conn = get_conn(settings.db_path)
    cur = conn.cursor()

    total_deleted = 0

    for period in ["WEEK", "MONTH", "QUARTER", "YEAR"]:
        rolling_type = f"{period}_ROLLING"

        # Find all rolling snapshots
        rolling_snapshots = cur.execute(
            "SELECT DISTINCT period_start_date FROM snapshots WHERE period_type=?",
            (rolling_type,),
        ).fetchall()

        print(f"\nChecking {rolling_type} snapshots...")
        print(f"  Found {len(rolling_snapshots)} unique period starts")

        for (start_date,) in rolling_snapshots:
            # Check if a final snapshot exists for this period start
            final_exists = cur.execute(
                """
                SELECT period_end_date FROM snapshots
                WHERE period_type=? AND period_start_date=?
                LIMIT 1
                """,
                (period, start_date),
            ).fetchone()

            if final_exists:
                # Delete all rolling snapshots for this period
                deleted = cur.execute(
                    "DELETE FROM snapshots WHERE period_type=? AND period_start_date=?",
                    (rolling_type, start_date),
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
