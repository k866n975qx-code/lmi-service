#!/usr/bin/env python3
"""
Verify the most recent daily snapshot (existing service) by rebuilding it with
live market data and comparing.

- Reads latest snapshot from daily_portfolio (assembled; what the API returns).
- Rebuilds that same date via pipeline: holdings from transactions, market data
  (loaded and filtered to as_of_date), build_daily_snapshot with frozen time.
- Compares stored vs rebuilt on key paths; writes report to samples_generated_v3/.

Usage (from repo root; activate venv first so app deps and .env load):
  source .venv/bin/activate   # or your venv
  python scripts/verify_latest_with_live.py

Optional:
  DB_PATH=/path/to/app.db python scripts/verify_latest_with_live.py

Report is written to repo root: verify_latest_report.txt
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

_db_path = os.environ.get("DB_PATH", str(_root / "data" / "app.db"))
SAMPLES_DIR = _root / "samples_generated_v3"
REPORT_PATH = _root / "verify_latest_report.txt"
SNAPSHOT_AT_ROOT = _root / "daily_rebuilt.json"

# Key paths to compare (stored vs rebuilt)
KEY_PATHS = [
    ("timestamps.portfolio_data_as_of_local", "as_of date"),
    ("portfolio.totals.market_value", "market value"),
    ("portfolio.totals.net_liquidation_value", "net liquidation value"),
    ("portfolio.totals.cost_basis", "cost basis"),
    ("portfolio.totals.holdings_count", "holdings count"),
    ("portfolio.totals.margin_loan_balance", "margin balance"),
    ("portfolio.income.projected_monthly_income", "projected monthly income"),
    ("portfolio.performance.twr_1m_pct", "twr 1m"),
    ("portfolio.performance.twr_12m_pct", "twr 12m"),
    ("meta.schema_version", "schema version"),
]


def _get_path(obj: dict, path: str):
    for p in path.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(p)
    return obj


def _compare_val(stored, rebuilt, path: str) -> tuple[bool, str]:
    if stored is None and rebuilt is None:
        return True, ""
    if stored is None and rebuilt is not None:
        return False, f"  {path}: stored None, rebuilt {rebuilt!r}"
    if stored is not None and rebuilt is None:
        return False, f"  {path}: stored {stored!r}, rebuilt missing"
    if isinstance(stored, (int, float)) and isinstance(rebuilt, (int, float)):
        diff = abs(float(stored) - float(rebuilt))
        # Allow small numeric tolerance; for $ values allow up to 0.01
        tol = 0.01 if "value" in path or "balance" in path or "income" in path or "cost" in path else 1e-6
        if diff <= tol:
            return True, ""
        return False, f"  {path}: stored {stored!r} != rebuilt {rebuilt!r} (diff={diff})"
    if stored != rebuilt:
        return False, f"  {path}: stored {stored!r} != rebuilt {rebuilt!r}"
    return True, ""


def main():
    import importlib.util
    from app.db import get_conn, migrate

    spec = importlib.util.spec_from_file_location(
        "migrate_snapshots_to_v3",
        _root / "scripts" / "migrate_snapshots_to_v3.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    migrate_snapshot = mod.migrate_snapshot

    conn = get_conn(_db_path)
    migrate(conn)
    cur = conn.cursor()

    from app.pipeline.snapshot_views import assemble_daily_snapshot
    as_of_row = cur.execute(
        "SELECT as_of_date_local FROM daily_portfolio ORDER BY as_of_date_local DESC LIMIT 1"
    ).fetchone()
    if not as_of_row:
        print("No snapshot in daily_portfolio. Running pipeline to generate one...")
        conn.close()
        try:
            import uuid
            from app.pipeline.orchestrator import _sync_impl
            run_id = str(uuid.uuid4())
            _sync_impl(run_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
            _write_report(None, None, None, [f"Pipeline failed: {e}"], failed_rebuild=True)
            sys.exit(1)
        conn = get_conn(_db_path)
        cur = conn.cursor()
        as_of_row = cur.execute(
            "SELECT as_of_date_local FROM daily_portfolio ORDER BY as_of_date_local DESC LIMIT 1"
        ).fetchone()
        if not as_of_row:
            _write_report(None, None, None, ["Pipeline ran but no snapshot was created (no holdings or pipeline error)."], failed_rebuild=True)
            conn.close()
            sys.exit(1)
        print("Snapshot created. Proceeding to verify...")

    as_of_date = as_of_row[0]
    stored = assemble_daily_snapshot(conn, as_of_date=as_of_date)
    if not stored:
        _write_report(None, None, None, ["Could not assemble snapshot from flat tables."], failed_rebuild=True)
        conn.close()
        sys.exit(1)
    schema_version = (stored.get("meta") or {}).get("schema_version")

    print(f"Latest snapshot: {as_of_date} (schema v{schema_version})")
    print("Rebuilding with live market data (holdings + prices as of that date)...")

    rebuilt = migrate_snapshot(conn, as_of_date, stored, schema_version, verbose=True)
    conn.close()

    if not rebuilt:
        print("\nFAIL: Rebuild returned None.")
        _write_report(as_of_date, stored, None, [], failed_rebuild=True)
        sys.exit(1)

    print("\nComparing stored (existing service) vs rebuilt (live pipeline)...")
    ok = True
    msgs = []
    for path, label in KEY_PATHS:
        s_val = _get_path(stored, path)
        r_val = _get_path(rebuilt, path)
        match, msg = _compare_val(s_val, r_val, path)
        if not match:
            ok = False
            msgs.append(msg)
            print(msg)
        else:
            print(f"  OK {label} ({path})")

    report_path = _write_report(as_of_date, stored, rebuilt, msgs, failed_rebuild=False)
    print(f"\nReport written to: {report_path}")
    with open(SNAPSHOT_AT_ROOT, "w") as f:
        json.dump(rebuilt, f, indent=2, sort_keys=True)
    print(f"Snapshot written to: {SNAPSHOT_AT_ROOT}")

    if ok:
        print("\nPASS: Stored snapshot matches rebuild with live market data.")
    else:
        print("\nFAIL: Some paths differ. See report.")
        sys.exit(1)


def _write_report(
    as_of_date: str | None,
    stored: dict | None,
    rebuilt: dict | None,
    diff_msgs: list[str],
    failed_rebuild: bool,
) -> Path:
    report_path = REPORT_PATH
    lines = [
        "Verify latest snapshot with live market data",
        f"Date: {as_of_date or 'N/A'}",
        f"Rebuild: {'FAILED' if failed_rebuild else 'OK'}",
        "",
    ]
    if diff_msgs:
        lines.append("Diffs (stored vs rebuilt):")
        lines.extend(diff_msgs)
        lines.append("")
    if rebuilt and stored:
        stored_mv = _get_path(stored, "portfolio.totals.market_value")
        rebuilt_mv = _get_path(rebuilt, "portfolio.totals.market_value")
        lines.append(f"Stored market_value:  {stored_mv}")
        lines.append(f"Rebuilt market_value: {rebuilt_mv}")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


if __name__ == "__main__":
    main()
