#!/usr/bin/env python3
"""
Rebuild snapshots to v5.0 format using current pipeline with historical data.

This script:
1. Extracts holdings (shares, cost_basis) from investment_transactions
2. Loads historical price data (filtered to snapshot date)
3. Loads account balances as of snapshot date
4. Freezes time to the snapshot date (all timestamps reflect that date)
5. Rebuilds snapshot from scratch using build_daily_snapshot (which now outputs v5.0)

Modes:
- Migration: Updates existing snapshots < v5.0 to v5.0
- Backfill: Creates snapshots for dates where none exist (margin balance = 0)

Usage:
    # Migrate a single date (recommended: test one at a time)
    python scripts/migrate_snapshots_to_v3.py --single-date 2026-01-25

    # Migrate all existing snapshots
    python scripts/migrate_snapshots_to_v3.py [--dry-run] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]

    # Backfill historical snapshots
    python scripts/migrate_snapshots_to_v3.py --backfill --start-date YYYY-MM-DD [--end-date YYYY-MM-DD]

    # Rebuild: delete flat daily snapshots only (then run --backfill separately if desired)
    python scripts/migrate_snapshots_to_v3.py --rebuild [--dry-run]
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
from app.pipeline.null_reasons import replace_nulls_with_reasons


def get_snapshots_to_migrate(conn: sqlite3.Connection, start_date: str | None, end_date: str | None, include_v5: bool = False):
    """Get list of daily snapshot dates from flat daily_portfolio to rebuild.

    With flat tables we no longer have payload_json/schema_version; we simply
    list as_of_date_local from daily_portfolio in the given range for rebuilding
    (e.g. to re-apply null reasons or refresh data).
    """
    query = "SELECT as_of_date_local FROM daily_portfolio WHERE 1=1"
    params = []
    if start_date:
        query += " AND as_of_date_local >= ?"
        params.append(start_date)
    if end_date:
        query += " AND as_of_date_local <= ?"
        params.append(end_date)
    query += " ORDER BY as_of_date_local"

    rows = conn.execute(query, params).fetchall()
    return [(row[0], None) for row in rows]


def get_dates_for_backfill(conn: sqlite3.Connection, start_date: str, end_date: str | None):
    """Get list of dates that need backfilling (no row in daily_portfolio)."""
    from datetime import timedelta

    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date() if end_date else datetime.now().date()

    existing = set()
    for row in conn.execute("SELECT as_of_date_local FROM daily_portfolio").fetchall():
        existing.add(row[0])

    results = []
    current = start
    while current <= end:
        date_str = current.isoformat()
        if date_str not in existing:
            results.append((date_str, None))
        current += timedelta(days=1)

    return results


# Flat daily tables only. Does NOT touch: period_summary, account_balances, margin_balance_history, runs, investment_transactions, lm_raw, etc.
DAILY_CHILD_TABLES = [
    "daily_holdings",
    "daily_goal_tiers",
    "daily_margin_rate_scenarios",
    "daily_return_attribution",
    "daily_dividends_upcoming",
]


def clear_flat_daily_snapshots(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """Delete all rows from flat daily snapshot tables. Returns total rows deleted."""
    cur = conn.cursor()
    total_deleted = 0
    for table in DAILY_CHILD_TABLES + ["daily_portfolio"]:
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
    return total_deleted


def load_snapshot(conn: sqlite3.Connection, as_of_date: str):
    """Load existing snapshot for a given date (assembled from flat tables)."""
    from app.pipeline.snapshot_views import assemble_daily_snapshot
    return assemble_daily_snapshot(conn, as_of_date=as_of_date)


def _load_account_balances_as_of(conn: sqlite3.Connection, as_of_date_str: str):
    """Load account balances: latest row on or before as_of_date per account (same logic as main pipeline)."""
    rows = conn.execute("""
        SELECT plaid_account_id, name, type, subtype, balance, as_of_date_local
        FROM account_balances
        WHERE as_of_date_local <= ?
        ORDER BY as_of_date_local DESC
    """, (as_of_date_str,)).fetchall()

    latest = {}
    for r in rows:
        plaid_account_id = r[0]
        if plaid_account_id in latest:
            continue
        latest[plaid_account_id] = {
            "plaid_account_id": plaid_account_id,
            "name": r[1],
            "type": r[2],
            "subtype": r[3],
            "balance": r[4],
            "as_of_date_local": r[5],
        }
    return list(latest.values())


def migrate_snapshot(conn: sqlite3.Connection, as_of_date_str: str, old_snapshot: dict | None, schema_version: float | None, verbose: bool = False):
    """
    Rebuild snapshot from scratch using current pipeline with historical data.
    build_daily_snapshot() now outputs v5.0 format via the transform_to_v5() transformer.
    """
    print(f"  Migrating {as_of_date_str}...")

    try:
        as_of_date = date.fromisoformat(as_of_date_str)

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

        account_balances_as_of = _load_account_balances_as_of(conn, as_of_date_str)

        if verbose:
            margin_accts = [a for a in account_balances_as_of if a.get('type') == 'loan' or 'borrow' in (a.get('name') or '').lower()]
            if margin_accts:
                total_margin = sum(abs(a.get('balance', 0)) for a in margin_accts)
                print(f"    Found margin balance in account_balances: ${total_margin:.2f}")

        if verbose:
            print(f"    Building snapshot...")

        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        from freezegun import freeze_time

        local_tz = ZoneInfo(settings.local_tz)
        frozen_local_dt = datetime.combine(as_of_date, datetime.min.time()).replace(hour=23, minute=59, tzinfo=local_tz)
        frozen_utc_dt = frozen_local_dt.astimezone(timezone.utc)

        if verbose:
            print(f"    Freezing time to {frozen_utc_dt.isoformat()}")

        from app.pipeline import snapshots as snapshots_module
        from collections import defaultdict

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
                print(f"    Loaded {len(counts)} symbols with trade counts")
            return counts

        with patch.object(snapshots_module, '_load_trade_counts', filtered_trade_counts):
            with patch.object(snapshots_module, '_load_account_balances', return_value=account_balances_as_of):
                with freeze_time(frozen_utc_dt):
                    new_snapshot, _ = build_daily_snapshot(conn, holdings, md)

        # build_daily_snapshot now outputs v5.0 via transform_to_v5()
        # No manual schema_version override needed

        if verbose:
            ver = (new_snapshot.get("meta") or {}).get("schema_version", "?")
            print(f"    Snapshot rebuilt successfully (v{ver})")

        return new_snapshot

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_snapshot(conn: sqlite3.Connection, as_of_date: str, snapshot: dict):
    """Save snapshot to flat tables via persist_daily_snapshot (force overwrite)."""
    try:
        from app.pipeline.snapshots import persist_daily_snapshot

        # Apply null reason engine before saving
        snapshot = replace_nulls_with_reasons(snapshot, kind="daily", conn=conn)
        updated_at_utc = datetime.now(timezone.utc).isoformat()
        run_id = f"migration_v5_{as_of_date}_{updated_at_utc[:19].replace(':', '-').replace(' ', '_')}"

        persist_daily_snapshot(conn, snapshot, run_id, force=True)
        return True
    except Exception as e:
        print(f"    ERROR saving: {e}")
        return False


def _get_market_value(snapshot: dict) -> float:
    """Extract market value from either v4 or v5 format."""
    # V5 path
    mv = (snapshot.get("portfolio") or {}).get("totals", {}).get("market_value")
    if mv is not None:
        return mv
    # V4 path
    mv = (snapshot.get("totals") or {}).get("market_value")
    if mv is not None:
        return mv
    return snapshot.get("total_market_value", 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--backfill", action="store_true", help="Backfill snapshots for dates with no existing snapshot")
    parser.add_argument("--rebuild", action="store_true", help="Delete flat daily snapshot tables only (no backfill; run --backfill separately)")
    parser.add_argument("--single-date", help="Migrate a single date (YYYY-MM-DD) — for testing one at a time")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--include-v5", action="store_true", help="Include complete v5.0 snapshots for rebuilding (e.g., to apply null reasons)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dump", action="store_true", help="Dump migrated snapshot JSON to stdout (for single-date)")
    args = parser.parse_args()

    conn = get_conn(settings.db_path)
    print(f"Connected to: {settings.db_path}\n")

    # Rebuild only: delete flat daily snapshots, then exit (backfill is separate)
    if args.rebuild:
        print("[REBUILD] Delete flat daily snapshot data only.\n")
        deleted = clear_flat_daily_snapshots(conn, dry_run=args.dry_run)
        if args.dry_run:
            print("\nRun without --dry-run to delete.")
        elif deleted:
            conn.commit()
            print(f"\nDone. Deleted {deleted} rows. Run --backfill --start-date ... to repopulate.")
        else:
            print("\nNo rows to delete.")
        conn.close()
        return 0

    # Single-date mode: migrate one date and optionally dump output
    if args.single_date:
        print(f"[SINGLE DATE MODE] Migrating {args.single_date} to v5.0\n")
        old = load_snapshot(conn, args.single_date)
        old_ver = None
        if old:
            old_ver = (old.get("meta") or {}).get("schema_version")
            print(f"  Existing snapshot: v{old_ver}")

        migrated = migrate_snapshot(conn, args.single_date, old, old_ver, verbose=True)
        if not migrated:
            print("\n  FAILED to migrate.")
            conn.close()
            return 1

        # Print summary
        summary = migrated.get("summary") or {}
        alerts = migrated.get("alerts") or []
        meta = migrated.get("meta") or {}
        print(f"\n  Result: v{meta.get('schema_version')}")
        print(f"  Market value: ${summary.get('total_market_value', 0):,.2f}")
        print(f"  Holdings: {summary.get('holdings_count', 0)}")
        print(f"  Alerts: {len(alerts)}")
        for a in alerts:
            print(f"    [{a.get('severity', '?').upper()}] {a.get('message', '')}")
        print(f"  Top-level keys: {sorted(migrated.keys())}")

        if args.dump:
            print(f"\n{'='*60}\nFull JSON:\n{'='*60}")
            print(json.dumps(migrated, indent=2, default=str))

        if not args.dry_run:
            response = input(f"\nSave to DB? [y/N]: ")
            if response.lower() in ['y', 'yes']:
                if save_snapshot(conn, args.single_date, migrated):
                    conn.commit()
                    print(f"  Saved {args.single_date} (v5.0)")
                else:
                    print(f"  FAILED to save")
        else:
            print(f"\n[DRY RUN] Not saving.")

        conn.close()
        return 0

    # Batch modes
    if args.backfill:
        if not args.start_date:
            print("ERROR: --start-date is required for backfill mode")
            return 1
        print(f"[BACKFILL MODE] Creating snapshots for dates with no existing snapshot\n")
        snapshots = get_dates_for_backfill(conn, args.start_date, args.end_date)
        mode_label = "backfill"
    else:
        if args.include_v5:
            print(f"[REBUILD MODE] Rebuilding all v5.0 snapshots (includes null reasons)\n")
        else:
            print(f"[MIGRATION MODE] Updating existing snapshots to v5.0\n")
        snapshots = get_snapshots_to_migrate(conn, args.start_date, args.end_date, include_v5=args.include_v5)
        mode_label = "rebuild" if args.include_v5 else "migrate"

    if not snapshots:
        print(f"No snapshots to {mode_label}.")
        return 0

    print(f"Found {len(snapshots)} snapshot(s):")
    for d, ver in snapshots[:20]:
        ver_label = f"v{ver}" if ver else "missing"
        print(f"  - {d} ({ver_label})")
    if len(snapshots) > 20:
        print(f"  ... and {len(snapshots) - 20} more")

    if args.dry_run:
        print(f"\n[DRY RUN] Run without --dry-run to {mode_label}.")
        return 0

    response = input(f"\n{mode_label.title()} {len(snapshots)} snapshot(s)? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        return 0

    print(f"\n{'='*60}")
    success, failed = 0, 0

    for as_of_date, old_ver in snapshots:
        old = load_snapshot(conn, as_of_date)

        if not args.backfill and not old:
            print(f"  ERROR: Could not load {as_of_date}")
            failed += 1
            continue

        migrated = migrate_snapshot(conn, as_of_date, old, old_ver, args.verbose)
        if not migrated:
            failed += 1
            continue

        # Verify market value consistency
        if old:
            old_mv = _get_market_value(old)
            new_mv = _get_market_value(migrated)
            if abs(old_mv - new_mv) > 100:
                print(f"  WARNING: Market value changed {old_mv:.2f} -> {new_mv:.2f}")

        if save_snapshot(conn, as_of_date, migrated):
            ver_from = f"v{old_ver}" if old_ver else "new"
            print(f"  Done {as_of_date} ({ver_from} -> v5.0)")
            success += 1
        else:
            failed += 1

    conn.commit()
    print(f"{'='*60}")
    print(f"Success: {success}, Failed: {failed}")
    conn.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
