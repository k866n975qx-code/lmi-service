#!/usr/bin/env python3
"""Verify attribution and target LTV from DB.

  From repo root:  python3 scripts/verify_attribution_and_ltv.py

Uses data/app.db (or DB_PATH env, or data/app.db.backup). No app deps.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    db_path = os.environ.get("DB_PATH") or str(ROOT / "data" / "app.db")
    db_path = Path(db_path)
    if not db_path.exists():
        db_path = ROOT / "data" / "app.db.backup"
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1
    print(f"Using DB: {db_path}\n", flush=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print("=== Attribution (daily) ===\n")
    print("Calculation: snapshots._return_attribution()")
    print("  - Window: 1m=30d, 3m=90d, 6m=180d, 12m=365d")
    print("  - Per symbol: contribution_pct = (price_return_dollars + income_dollars) / portfolio_start_val * 100")
    print("  - Weight = position_start_value / portfolio_start_val * 100")
    print("  - Portfolio total = sum of per-symbol contributions (stored as symbol=_portfolio)\n")

    cur.execute(
        """SELECT as_of_date_local, window, symbol, contribution_pct, weight_avg_pct, return_pct
           FROM daily_return_attribution
           ORDER BY as_of_date_local DESC, window, CASE WHEN symbol = '_portfolio' THEN 0 ELSE 1 END, symbol
           LIMIT 32"""
    )
    rows = cur.fetchall()
    if not rows:
        print("No rows in daily_return_attribution.")
    else:
        last_date = None
        for r in rows:
            r = dict(r)
            if r["as_of_date_local"] != last_date:
                last_date = r["as_of_date_local"]
                print(f"  Date: {last_date}")
            print(f"    {r['window']:>3}  {r['symbol']:12}  contribution_pct={r['contribution_pct']}  weight_avg_pct={r['weight_avg_pct']}  return_pct={r['return_pct']}")
        print()

    print("=== Target LTV vs actual LTV ===\n")
    print("  - assumption_target_ltv_pct: goal tier assumption (e.g. 30% for leveraged tiers).")
    print("  - daily_portfolio.ltv_pct: actual portfolio LTV = margin_loan_balance / market_value * 100.\n")

    cur.execute(
        """SELECT as_of_date_local, tier, name, assumption_ltv_maintained, assumption_target_ltv_pct
           FROM daily_goal_tiers
           ORDER BY as_of_date_local DESC, tier
           LIMIT 18"""
    )
    tiers = cur.fetchall()
    cur.execute(
        """SELECT as_of_date_local, ltv_pct, margin_loan_balance, market_value
           FROM daily_portfolio
           ORDER BY as_of_date_local DESC
           LIMIT 5"""
    )
    portfolios = cur.fetchall()

    if tiers:
        print("  daily_goal_tiers (latest dates):")
        last_d = None
        for r in tiers:
            r = dict(r)
            d = r["as_of_date_local"]
            if d != last_d:
                last_d = d
                print(f"    {d}:")
            print(f"      tier {r['tier']} {r['name'] or '':20}  ltv_maintained={r['assumption_ltv_maintained']}  target_ltv_pct={r['assumption_target_ltv_pct']}")
    if portfolios:
        print("\n  daily_portfolio (actual LTV):")
        for r in portfolios:
            r = dict(r)
            print(f"    {r['as_of_date_local']}  ltv_pct={r['ltv_pct']}  margin_loan_balance={r['margin_loan_balance']}  market_value={r['market_value']}")
        latest = dict(portfolios[0])
        print(f"\n  Summary: tier target_ltv_pct=30 (leveraged tiers). Actual LTV (latest {latest['as_of_date_local']}) = {latest['ltv_pct']}%.")
    else:
        print("\n  No daily_portfolio rows.")

    conn.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
