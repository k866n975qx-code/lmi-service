#!/usr/bin/env python3
"""
Rebuild snapshots to v4.0 format using current pipeline with historical data.

This script:
1. Extracts holdings (shares, cost_basis) from investment_transactions
2. Loads historical price data (filtered to snapshot date)
3. Loads account balances as of snapshot date
4. Freezes time to the snapshot date (all timestamps reflect that date)
5. Rebuilds snapshot from scratch using build_daily_snapshot
6. Updates schema_version to 4.0

Modes:
- Migration: Updates existing v3.x snapshots to v4.0
- Backfill: Creates snapshots for dates where none exist (margin balance = 0)

Usage:
    # Migrate existing snapshots
    python scripts/migrate_snapshots_to_v3.py [--dry-run] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]

    # Backfill historical snapshots
    python scripts/migrate_snapshots_to_v3.py --backfill [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone, date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.db import get_conn
from app.pipeline.market import MarketData
from app.pipeline.snapshots import build_daily_snapshot


# Key v4.0 fields to check for incomplete snapshots
V4_KEY_FIELDS = {'goal_progress_optimistic', 'goal_tiers', 'goal_pace', 'margin_stress', 'prices_as_of_utc'}


def get_snapshots_to_migrate(conn: sqlite3.Connection, start_date: str | None, end_date: str | None):
    """Get list of snapshots that need migration (< v4.0 or incomplete v4.0)."""
    # Get all snapshots in date range
    query = """
        SELECT as_of_date_local, payload_json, CAST(json_extract(payload_json, '$.meta.schema_version') AS REAL) as schema_version
        FROM snapshot_daily_current
        WHERE 1=1
    """
    params = []
    if start_date:
        query += " AND as_of_date_local >= ?"
        params.append(start_date)
    if end_date:
        query += " AND as_of_date_local <= ?"
        params.append(end_date)
    query += " ORDER BY as_of_date_local"

    results = []
    for row in conn.execute(query, params).fetchall():
        as_of_date, payload_json, schema_version = row[0], row[1], row[2]

        # Include snapshots < v4.0
        if schema_version < 4.0:
            results.append((as_of_date, schema_version))
            continue

        # Check if v4.0 snapshot is missing key fields
        snapshot = json.loads(payload_json)
        missing_fields = V4_KEY_FIELDS - set(snapshot.keys())

        # Rebuild if missing key v4.0 fields
        if missing_fields:
            results.append((as_of_date, schema_version))

    return results


def get_dates_for_backfill(conn: sqlite3.Connection, start_date: str, end_date: str | None):
    """Get list of dates that need backfilling (no snapshot exists)."""
    from datetime import datetime, timedelta

    start = datetime.fromisoformat(start_date).date()

    if end_date:
        end = datetime.fromisoformat(end_date).date()
    else:
        end = datetime.now().date()

    # Get all existing snapshot dates
    existing = set()
    for row in conn.execute("SELECT as_of_date_local FROM snapshot_daily_current").fetchall():
        existing.add(row[0])

    # Generate date range and filter out existing
    results = []
    current = start
    while current <= end:
        date_str = current.isoformat()
        if date_str not in existing:
            results.append((date_str, None))  # None for schema_version since no snapshot exists
        current += timedelta(days=1)

    return results


def load_snapshot(conn: sqlite3.Connection, as_of_date: str):
    """Load existing snapshot for a given date."""
    row = conn.execute("SELECT payload_json FROM snapshot_daily_current WHERE as_of_date_local = ?",
                      (as_of_date,)).fetchone()
    return json.loads(row[0]) if row else None


def _load_account_balances_as_of(conn: sqlite3.Connection, as_of_date_str: str):
    """Load account balances AS OF a specific date."""
    rows = conn.execute("""
        SELECT plaid_account_id, name, type, subtype, balance, as_of_date_local
        FROM account_balances
        WHERE as_of_date_local = ?
        ORDER BY as_of_date_local DESC
    """, (as_of_date_str,)).fetchall()

    return [{"plaid_account_id": r[0], "name": r[1], "type": r[2], "subtype": r[3],
             "balance": r[4], "as_of_date_local": r[5]} for r in rows]


def migrate_snapshot(conn: sqlite3.Connection, as_of_date_str: str, old_snapshot: dict | None, schema_version: float | None, verbose: bool = False):
    """
    Rebuild snapshot from scratch using current pipeline with historical data.
    Rebuilds holdings from investment_transactions as of that date.
    Freezes time to the snapshot date so all timestamps reflect that date.
    """
    print(f"  Migrating {as_of_date_str}...")

    try:
        as_of_date = date.fromisoformat(as_of_date_str)

        # Rebuild holdings from transactions as of that date
        if verbose:
            print(f"    Rebuilding holdings from transactions...")

        query = """
            SELECT symbol,
                   SUM(CASE
                       WHEN transaction_type IN ('buy', 'sell') THEN quantity
                       ELSE 0
                   END) as shares,
                   SUM(CASE
                       WHEN transaction_type IN ('buy', 'sell') THEN amount
                       ELSE 0
                   END) as cost_basis
            FROM investment_transactions
            WHERE date <= ?
            GROUP BY symbol
            HAVING shares > 0.001
        """

        rows = conn.execute(query, (as_of_date_str,)).fetchall()

        holdings = {}
        symbols = set()
        for symbol, shares, cost_basis in rows:
            if symbol and shares > 0:
                symbols.add(symbol)
                holdings[symbol] = {'symbol': symbol, 'shares': shares, 'cost_basis': cost_basis}

        if not holdings:
            print(f"    ERROR: No holdings from transactions")
            return None

        if verbose:
            print(f"    Rebuilt {len(holdings)} holdings from transactions")

        # Load market data and filter to as_of_date
        md = MarketData()
        bench_symbols = [s for s in [settings.benchmark_primary, settings.benchmark_secondary] if s]
        price_symbols = sorted({*symbols, *bench_symbols})

        if verbose:
            print(f"    Loading price history...")
        md.load(price_symbols, deadline=None)

        # Filter prices to as_of_date
        import pandas as pd
        for symbol in md.prices:
            if md.prices[symbol] is not None and not md.prices[symbol].empty:
                df = md.prices[symbol]
                if 'date' in df.columns:
                    df_dates = pd.to_datetime(df['date']).dt.date
                    mask = df_dates <= as_of_date
                    md.prices[symbol] = df[mask].reset_index(drop=True)

        # Wrap FRED adapter to filter data by date
        if hasattr(md, 'fred') and md.fred is not None:
            original_fred_series = md.fred.series

            def filtered_fred_series(series_id: str):
                df = original_fred_series(series_id)
                if df is not None and not df.empty:
                    # Filter FRED data to as_of_date
                    # FredAdapter uses 'observation_date' or 'date' column
                    date_col = 'observation_date' if 'observation_date' in df.columns else 'date' if 'date' in df.columns else None
                    if date_col:
                        df_copy = df.copy()
                        df_dates = pd.to_datetime(df_copy[date_col]).dt.date
                        mask = df_dates <= as_of_date
                        return df_copy[mask].reset_index(drop=True)
                return df

            md.fred.series = filtered_fred_series

            if verbose:
                print(f"    Filtered FRED data to {as_of_date}")

        # Load account balances as of that date
        # This includes margin account balance if it exists in the DB
        account_balances_as_of = _load_account_balances_as_of(conn, as_of_date_str)

        if verbose:
            margin_accts = [a for a in account_balances_as_of if a.get('type') == 'loan' or 'borrow' in (a.get('name') or '').lower()]
            if margin_accts:
                total_margin = sum(abs(a.get('balance', 0)) for a in margin_accts)
                print(f"    Found margin balance in account_balances: ${total_margin:.2f}")

        # Build new snapshot from scratch
        if verbose:
            print(f"    Building snapshot...")

        # CRITICAL: Freeze time to the snapshot date
        # This ensures all datetime.now() calls return the historical date
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        from freezegun import freeze_time

        # Create frozen datetime for the snapshot date
        # Use end of day for the snapshot date
        local_tz = ZoneInfo(settings.local_tz)
        frozen_local_dt = datetime.combine(as_of_date, datetime.min.time()).replace(hour=23, minute=59, tzinfo=local_tz)
        frozen_utc_dt = frozen_local_dt.astimezone(timezone.utc)

        if verbose:
            print(f"    Freezing time to {frozen_utc_dt.isoformat()}")

        # Use freezegun to freeze time globally
        from app.pipeline import snapshots as snapshots_module
        from collections import defaultdict

        # Patch trade counts to filter by date
        TRADE_TYPES = {"buy", "sell"}

        def filtered_trade_counts(conn_inner):
            """Load trade counts filtered to snapshot date."""
            cur = conn_inner.cursor()
            rows = cur.execute(
                "SELECT symbol, transaction_type FROM investment_transactions WHERE symbol IS NOT NULL AND date <= ?",
                (as_of_date_str,)
            ).fetchall()
            counts = defaultdict(int)
            for symbol, tx_type in rows:
                if not symbol:
                    continue
                tx_type = (tx_type or "").lower()
                if tx_type in TRADE_TYPES:
                    counts[str(symbol).upper()] += 1
            if verbose:
                print(f"    Loaded {len(counts)} symbols with trade counts (e.g., AMLP: {counts.get('AMLP', 0)})")
            return counts

        with patch.object(snapshots_module, '_load_trade_counts', filtered_trade_counts):
            with patch.object(snapshots_module, '_load_account_balances', return_value=account_balances_as_of):
                with freeze_time(frozen_utc_dt):
                    new_snapshot, _ = build_daily_snapshot(conn, holdings, md)

        # Update schema version
        if 'meta' not in new_snapshot:
            new_snapshot['meta'] = {}
        new_snapshot['meta']['schema_version'] = "4.0"

        if verbose:
            print(f"    Snapshot rebuilt successfully")

        return new_snapshot

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_snapshot(conn: sqlite3.Connection, as_of_date: str, snapshot: dict):
    """Save snapshot to database."""
    try:
        payload_json = json.dumps(snapshot, separators=(',', ':'))
        import hashlib
        payload_sha256 = hashlib.sha256(payload_json.encode('utf-8')).hexdigest()
        updated_at_utc = datetime.now(timezone.utc).isoformat()
        built_from_run_id = f"migration_v3_{as_of_date}_{updated_at_utc[:19].replace(':', '-')}"

        conn.execute("""
            INSERT OR REPLACE INTO snapshot_daily_current
            (as_of_date_local, built_from_run_id, payload_json, payload_sha256, updated_at_utc)
            VALUES (?, ?, ?, ?, ?)
        """, (as_of_date, built_from_run_id, payload_json, payload_sha256, updated_at_utc))

        return True
    except Exception as e:
        print(f"    ERROR saving: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--backfill", action="store_true", help="Backfill snapshots for dates with no existing snapshot")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    conn = get_conn(settings.db_path)
    print(f"Connected to: {settings.db_path}\n")

    if args.backfill:
        if not args.start_date:
            print("ERROR: --start-date is required for backfill mode")
            return 1
        print(f"[BACKFILL MODE] Creating snapshots for dates with no existing snapshot\n")
        snapshots = get_dates_for_backfill(conn, args.start_date, args.end_date)
        mode_label = "backfill"
    else:
        print(f"[MIGRATION MODE] Updating existing snapshots to v4.0\n")
        snapshots = get_snapshots_to_migrate(conn, args.start_date, args.end_date)
        mode_label = "migrate"

    if not snapshots:
        print(f"No snapshots to {mode_label}.")
        return 0

    print(f"Found {len(snapshots)} snapshot(s):")
    for date, ver in snapshots[:20]:  # Show first 20
        ver_label = f"v{ver}" if ver else "missing"
        print(f"  - {date} ({ver_label})")
    if len(snapshots) > 20:
        print(f"  ... and {len(snapshots) - 20} more")

    if args.dry_run:
        print(f"\n[DRY RUN] Run without --dry-run to {mode_label}.")
        return 0

    response = input(f"\n{mode_label.title()}? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        return 0

    print(f"\n{'='*60}")
    success, failed = 0, 0

    for as_of_date, old_ver in snapshots:
        old = load_snapshot(conn, as_of_date)

        # In backfill mode, old snapshot won't exist (margin balance will be 0)
        # In migration mode, old snapshot must exist
        if not args.backfill and not old:
            print(f"  ERROR: Could not load {as_of_date}")
            failed += 1
            continue

        migrated = migrate_snapshot(conn, as_of_date, old, old_ver, args.verbose)
        if not migrated:
            failed += 1
            continue

        # Verify (skip for backfill since there's no old snapshot to compare)
        if old:
            old_mv = old.get('totals', {}).get('market_value', 0)
            new_mv = migrated.get('totals', {}).get('market_value', 0)
            if abs(old_mv - new_mv) > 100:  # Allow for small price differences
                print(f"  WARNING: Market value changed {old_mv:.2f} -> {new_mv:.2f}")

        if save_snapshot(conn, as_of_date, migrated):
            if args.backfill:
                print(f"  ✓ Created {as_of_date} (v4.0)\n")
            else:
                print(f"  ✓ Migrated {as_of_date} (v{old_ver} -> v4.0)\n")
            success += 1
        else:
            failed += 1

    print(f"{'='*60}")
    print(f"Success: {success}, Failed: {failed}")
    conn.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
