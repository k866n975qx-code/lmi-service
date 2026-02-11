#!/usr/bin/env python3
"""
Clear all flat snapshot/summary data so you can backfill from scratch.

Deletes rows from:
- daily_portfolio and child tables (daily_holdings, daily_goal_tiers, etc.)
- period_summary and child tables (period_risk_stats, period_intervals, period_holding_changes)

Does NOT touch: account_balances, margin_balance_history, runs, investment_transactions, lm_raw, etc.

Usage:
    python scripts/clear_flat_snapshots.py           # dry run (print only)
    python scripts/clear_flat_snapshots.py --execute # actually delete
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.db import get_conn
from app.config import settings


# Child tables first, then parents (for clarity; SQLite without FK enforcement doesn't require order)
DAILY_CHILD_TABLES = [
    "daily_holdings",
    "daily_goal_tiers",
    "daily_margin_rate_scenarios",
    "daily_return_attribution",
    "daily_dividends_upcoming",
]
PERIOD_CHILD_TABLES = [
    "period_risk_stats",
    "period_intervals",
    "period_holding_changes",
]


def main():
    import argparse
    p = argparse.ArgumentParser(description="Clear flat snapshot/summary data for fresh backfill.")
    p.add_argument("--execute", action="store_true", help="Actually delete rows (default is dry run)")
    args = p.parse_args()
    dry_run = not args.execute

    conn = get_conn(settings.db_path)
    cur = conn.cursor()

    if dry_run:
        print("DRY RUN (use --execute to delete)\n")

    total_deleted = 0
    for table in DAILY_CHILD_TABLES + ["daily_portfolio"] + PERIOD_CHILD_TABLES + ["period_summary"]:
        try:
            n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception:
            n = 0
        if n == 0:
            continue
        if dry_run:
            print(f"  Would delete {n} rows from {table}")
        else:
            cur.execute(f"DELETE FROM {table}")
            total_deleted += cur.rowcount
            print(f"  Deleted {cur.rowcount} rows from {table}")
    if not dry_run and total_deleted:
        conn.commit()
        print(f"\nDone. Total rows deleted: {total_deleted}")
    elif dry_run:
        print("\nRun with --execute to apply.")


if __name__ == "__main__":
    main()
