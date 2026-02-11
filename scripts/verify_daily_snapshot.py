"""
Verify that a daily snapshot round-trip (example -> flat -> assembled) matches the example.

Usage:
  python scripts/verify_daily_snapshot.py [path_to_daily.json]

If no path given, uses samples_generated_v3/daily.json.
Loads the example, writes to flat tables via _write_daily_flat, then assembles via
assemble_daily_snapshot and compares key fields to the example. Reports mismatches
so we can confirm the daily snapshot is filled correctly vs the example.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

# Use DB_PATH env or default so we don't require dotenv
import os
_db_path = os.environ.get("DB_PATH", str(_root / "data" / "app.db"))

from app.db import get_conn, migrate
from app.pipeline.flat_persist import _write_daily_flat
from app.pipeline.snapshot_views import assemble_daily_snapshot


# Key paths to compare (example path -> description)
KEY_PATHS = [
    ("timestamps.portfolio_data_as_of_local", "as_of date"),
    ("timestamps.price_data_as_of_utc", "prices as of"),
    ("portfolio.totals.market_value", "market value"),
    ("portfolio.totals.net_liquidation_value", "net liquidation value"),
    ("portfolio.totals.cost_basis", "cost basis"),
    ("portfolio.totals.holdings_count", "holdings count"),
    ("portfolio.totals.margin_loan_balance", "margin balance"),
    ("portfolio.totals.margin_to_portfolio_pct", "ltv %"),
    ("portfolio.income.projected_monthly_income", "projected monthly income"),
    ("portfolio.income.forward_12m_total", "forward 12m total"),
    ("portfolio.performance.twr_1m_pct", "twr 1m"),
    ("portfolio.performance.twr_12m_pct", "twr 12m"),
    ("portfolio.risk.volatility.vol_30d_pct", "vol 30d"),
    ("portfolio.risk.ratios.sharpe_1y", "sharpe 1y"),
    ("portfolio.risk.ratios.sortino_1y", "sortino 1y"),
    ("portfolio.risk.drawdown.max_drawdown_1y_pct", "max dd 1y"),
    ("goals.baseline.progress_pct", "goal progress %"),
    ("goals.baseline.months_to_goal", "months to goal"),
    ("meta.schema_version", "schema version"),
]


def _get_path(obj: dict, path: str):
    parts = path.split(".")
    for p in parts:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(p)
    return obj


def _compare_val(expected, actual, path: str) -> tuple[bool, str]:
    if expected is None and actual is None:
        return True, ""
    if expected is None and actual is not None:
        return False, f"  {path}: example has None, assembled has {actual!r}"
    if expected is not None and actual is None:
        return False, f"  {path}: example has {expected!r}, assembled has None (missing)"
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if abs(float(expected) - float(actual)) < 1e-6:
            return True, ""
        return False, f"  {path}: example {expected!r} != assembled {actual!r}"
    if expected != actual:
        return False, f"  {path}: example {expected!r} != assembled {actual!r}"
    return True, ""


def main():
    sample_path = Path(__file__).resolve().parents[1] / "samples_generated_v3" / "daily.json"
    if len(sys.argv) > 1:
        sample_path = Path(sys.argv[1])
    if not sample_path.exists():
        print(f"Example snapshot not found: {sample_path}")
        sys.exit(1)

    with open(sample_path) as f:
        example = json.load(f)

    as_of = (
        (example.get("timestamps") or {}).get("portfolio_data_as_of_local")
        or example.get("as_of_date_local")
        or (example.get("timestamps") or {}).get("portfolio_data_as_of_utc", "")[:10]
    )
    if not as_of:
        print("Could not determine as_of_date from example")
        sys.exit(1)

    conn = get_conn(_db_path)
    migrate(conn)

    # Ensure example has created_at_utc for flat write
    if not example.get("created_at_utc") and (example.get("meta") or {}).get("snapshot_created_at"):
        example["created_at_utc"] = example["meta"]["snapshot_created_at"]

    print(f"Writing example snapshot (as_of={as_of}) to flat tables...")
    _write_daily_flat(conn, example, "verify_run")

    print("Assembling daily snapshot from flat tables...")
    assembled = assemble_daily_snapshot(conn, as_of)
    if not assembled:
        print("ERROR: assemble_daily_snapshot returned None")
        sys.exit(1)

    print("Comparing key paths (example vs assembled)...")
    ok = True
    for path, label in KEY_PATHS:
        exp_val = _get_path(example, path)
        act_val = _get_path(assembled, path)
        match, msg = _compare_val(exp_val, act_val, path)
        if not match:
            ok = False
            print(msg)
        else:
            print(f"  OK {label} ({path})")

    if ok:
        print("\nAll key paths match. Daily snapshot is filled correctly vs example.")
    else:
        print("\nSome paths differ. Review above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
