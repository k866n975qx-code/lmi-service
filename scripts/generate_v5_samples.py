#!/usr/bin/env python3
"""Generate V5 sample JSON files from actual database data."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import date
from app.db import get_conn
from app.config import settings
from app.pipeline.diff_daily import diff_daily_from_db
from app.pipeline.diff_periods import diff_periods_from_db


def slim_snapshot(snap: dict, is_period: bool = False) -> dict:
    """Remove provenance/cache metadata but keep all data fields."""
    if not snap:
        return snap

    # Remove provenance and cache metadata fields recursively
    def remove_metadata(obj):
        if isinstance(obj, dict):
            # Remove all provenance and cache fields
            obj = {k: v for k, v in obj.items()
                   if not k.endswith('_provenance') and k != '_cache' and k != 'provenance'}
            return {k: remove_metadata(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [remove_metadata(item) for item in obj]
        return obj

    return remove_metadata(snap)


def main():
    conn = get_conn(settings.db_path)
    cur = conn.cursor()

    from app.pipeline.snapshot_views import assemble_daily_snapshot
    daily_row = cur.execute(
        "SELECT as_of_date_local FROM daily_portfolio ORDER BY as_of_date_local DESC LIMIT 1"
    ).fetchone()
    if not daily_row:
        print("No daily snapshots found!")
        return
    as_of_date = daily_row[0]
    snap = assemble_daily_snapshot(conn, as_of_date=as_of_date)
    if not snap:
        print("Could not assemble daily snapshot.")
        return

    print(f"Generating V5 samples from {as_of_date}...")
    print(f"Schema version: {snap.get('meta', {}).get('schema_version')}")

    # Daily samples
    with open('samples_generated_v3/daily.json', 'w') as f:
        json.dump(snap, f, indent=2, sort_keys=True)
    print(f"✓ Generated samples_generated_v3/daily.json ({len(json.dumps(snap))} bytes)")

    daily_slim = slim_snapshot(snap.copy())
    with open('samples_generated_v3/daily_slim.json', 'w') as f:
        json.dump(daily_slim, f, indent=2, sort_keys=True)
    print(f"✓ Generated samples_generated_v3/daily_slim.json ({len(json.dumps(daily_slim))} bytes)")

    from app.pipeline.snapshot_views import assemble_period_snapshot
    period_types = ['WEEK', 'MONTH', 'QUARTER', 'YEAR']
    for ptype in period_types:
        period_row = cur.execute(
            "SELECT period_start_date, period_end_date FROM period_summary WHERE period_type=? AND is_rolling=0 ORDER BY period_end_date DESC LIMIT 1",
            (ptype,)
        ).fetchone()
        if period_row:
            start_date, end_date = period_row[0], period_row[1]
            period_snap = assemble_period_snapshot(conn, ptype, end_date, period_start_date=start_date)
        if period_row and period_snap:
            start_date, end_date = period_row[0], period_row[1]
            # Full period sample
            filename = f"samples_generated_v3/period_{ptype.lower()}.json"
            with open(filename, 'w') as f:
                json.dump(period_snap, f, indent=2, sort_keys=True)
            print(f"✓ Generated {filename} ({start_date} to {end_date})")

            # Slim period sample
            period_slim = slim_snapshot(period_snap.copy(), is_period=True)
            filename_slim = f"samples_generated_v3/period_{ptype.lower()}_slim.json"
            with open(filename_slim, 'w') as f:
                json.dump(period_slim, f, indent=2, sort_keys=True)
            print(f"✓ Generated {filename_slim}")

    # Diff daily sample (compare latest two days)
    prev_date_row = cur.execute(
        "SELECT as_of_date_local FROM daily_portfolio WHERE as_of_date_local < ? ORDER BY as_of_date_local DESC LIMIT 1",
        (as_of_date,)
    ).fetchone()

    if prev_date_row:
        prev_date = prev_date_row[0]
        diff = diff_daily_from_db(conn, as_of_date, prev_date)

        if diff:
            with open('samples_generated_v3/diff_daily.json', 'w') as f:
                json.dump(diff, f, indent=2, sort_keys=True)
            print(f"✓ Generated samples_generated_v3/diff_daily.json ({as_of_date} vs {prev_date})")

    # Diff period sample (compare two periods)
    period_row1 = cur.execute(
        "SELECT period_start_date, period_end_date FROM period_summary WHERE period_type='MONTH' AND is_rolling=0 ORDER BY period_end_date DESC LIMIT 2"
    ).fetchall()

    if len(period_row1) >= 2:
        start1, end1 = period_row1[0]
        start2, end2 = period_row1[1]

        # Use period boundaries
        period_diff = diff_periods_from_db(conn, 'monthly', end1, end2)

        if period_diff:
            with open('samples_generated_v3/diff_period.json', 'w') as f:
                json.dump(period_diff, f, indent=2, sort_keys=True)
            print(f"✓ Generated samples_generated_v3/diff_period.json (monthly {end1} vs {end2})")

    conn.close()
    print("\n" + "=" * 50)
    print("All V5 sample files generated successfully!")


if __name__ == "__main__":
    main()
