#!/usr/bin/env python3
"""
Phase 19 invariant checks for flat schema.

Validates consistency of daily_portfolio, daily_holdings, period_summary, etc.
Run after backfill to verify data integrity.

Usage:
    python scripts/validate_flat_invariants.py
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


def main():
    conn = get_conn(settings.db_path)
    conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
    cur = conn.cursor()
    warnings = []
    errors = []

    # Check 1: Weight sums ≈ 100% per date
    try:
        for row in cur.execute("SELECT as_of_date_local FROM daily_portfolio").fetchall():
            d = row["as_of_date_local"]
            r = cur.execute("SELECT SUM(weight_pct) as s FROM daily_holdings WHERE as_of_date_local=?", (d,)).fetchone()
            w = r["s"] if r else None
            if w is not None and abs(float(w) - 100.0) > 0.5:
                warnings.append(f"{d}: weight sum = {w} (expected ~100)")
    except Exception as e:
        errors.append(f"Check 1 (weight sums): {e}")

    # Check 2: NLV ≈ MV - margin_loan_balance
    try:
        for row in cur.execute(
            "SELECT as_of_date_local, market_value, net_liquidation_value, margin_loan_balance FROM daily_portfolio"
        ).fetchall():
            mv = row.get("market_value")
            nlv = row.get("net_liquidation_value")
            margin = row.get("margin_loan_balance")
            if mv is not None and nlv is not None and margin is not None:
                expected = float(mv) - float(margin)
                actual = float(nlv)
                if abs(expected - actual) > 0.10:
                    warnings.append(f"{row['as_of_date_local']}: NLV mismatch {actual} vs expected {expected}")
    except Exception as e:
        errors.append(f"Check 2 (NLV): {e}")

    # Check 3: Holdings count matches daily_holdings rows
    try:
        for row in cur.execute("SELECT as_of_date_local, holdings_count FROM daily_portfolio").fetchall():
            d = row["as_of_date_local"]
            expected_count = row.get("holdings_count")
            if expected_count is None:
                continue
            r = cur.execute("SELECT COUNT(*) as c FROM daily_holdings WHERE as_of_date_local=?", (d,)).fetchone()
            actual = r["c"] if r else 0
            if actual != expected_count:
                warnings.append(f"{d}: holdings count {actual} vs daily_portfolio.holdings_count {expected_count}")
    except Exception as e:
        errors.append(f"Check 3 (holdings count): {e}")

    # Check 4: Goal tiers count (expect 6 per date when present)
    try:
        for row in cur.execute("SELECT DISTINCT as_of_date_local FROM daily_goal_tiers").fetchall():
            d = row["as_of_date_local"]
            r = cur.execute("SELECT COUNT(*) as c FROM daily_goal_tiers WHERE as_of_date_local=?", (d,)).fetchone()
            count = r["c"] if r else 0
            if count != 6:
                warnings.append(f"{d}: daily_goal_tiers has {count} rows (expected 6)")
    except Exception as e:
        errors.append(f"Check 4 (goal tiers): {e}")

    # Check 5: Account balances (margin/loan comes from account_balances by date)
    try:
        r = cur.execute(
            "SELECT COUNT(DISTINCT as_of_date_local) as c FROM account_balances WHERE type = 'loan'"
        ).fetchone()
        loan_dates = r["c"] if r else 0
        if loan_dates == 0:
            warnings.append("account_balances: no loan-account rows (margin will be 0)")
        elif loan_dates < 30:
            warnings.append(f"account_balances: {loan_dates} distinct dates for loan (typical 30+ for 30d window)")
    except Exception as e:
        errors.append(f"Check 5 (account_balances loan): {e}")

    # Report
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  {e}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")
    if not errors and not warnings:
        print("Validation complete — no issues found.")
    if errors:
        sys.exit(2)
    if warnings:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
