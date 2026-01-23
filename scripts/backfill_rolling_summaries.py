#!/usr/bin/env python3
"""
Backfill rolling summaries for all existing daily snapshots.

Usage:
    python scripts/backfill_rolling_summaries.py                    # Backfill all dates
    python scripts/backfill_rolling_summaries.py 2025-01-01         # Backfill from date onwards
    python scripts/backfill_rolling_summaries.py 2025-01-01 2025-12-31  # Backfill date range
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
from app.pipeline.snapshots import backfill_rolling_summaries

if __name__ == '__main__':
    start_date = sys.argv[1] if len(sys.argv) > 1 else None
    end_date = sys.argv[2] if len(sys.argv) > 2 else None

    print('Backfilling rolling summaries...')
    if start_date and end_date:
        print(f'Date range: {start_date} to {end_date}')
    elif start_date:
        print(f'From: {start_date} onwards')
    else:
        print('All dates')

    conn = get_conn(settings.db_path)
    backfill_rolling_summaries(conn, start_date=start_date, end_date=end_date)
    print('Done.')
