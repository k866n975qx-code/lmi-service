from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path and is the working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

try:
    from app.config import settings
except Exception:
    class _Settings:
        db_path = os.getenv("DB_PATH", "./data/app.db")
        quote_ttl_minutes = int(os.getenv("QUOTE_TTL_MINUTES", "60") or 60)
    settings = _Settings()

from app.db import get_conn, migrate
from app.pipeline.validation import validate_daily_snapshot, validate_period_snapshot


def _as_float(val):
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _parse_iso(val: str | None):
    if not val:
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_date(val: str | None):
    if not val:
        return None
    try:
        return date.fromisoformat(val)
    except ValueError:
        return None


def _close(a: float | None, b: float | None, rel: float = 1e-6, abs_tol: float = 0.02) -> bool:
    if a is None or b is None:
        return True
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def _check_daily_math(snap: dict):
    errors: list[str] = []
    warnings: list[str] = []

    holdings = snap.get("holdings")
    if not isinstance(holdings, list):
        errors.append("holdings missing or not a list")
        return errors, warnings

    sum_mv = 0.0
    sum_cb = 0.0
    sum_unreal = 0.0
    sum_weight = 0.0
    weight_count = 0

    for h in holdings:
        sym = h.get("symbol") or "unknown"
        shares = _as_float(h.get("shares"))
        last_price = _as_float(h.get("last_price"))
        mv = _as_float(h.get("market_value"))
        cost_basis = _as_float(h.get("cost_basis"))
        avg_cost = _as_float(h.get("avg_cost"))
        unreal = _as_float(h.get("unrealized_pnl"))
        unreal_pct = _as_float(h.get("unrealized_pct"))
        weight = _as_float(h.get("weight_pct"))

        if mv is not None:
            sum_mv += mv
        if cost_basis is not None:
            sum_cb += cost_basis
        if unreal is not None:
            sum_unreal += unreal
        if weight is not None:
            sum_weight += weight
            weight_count += 1

        if shares is not None and last_price is not None and mv is not None:
            expected_mv = shares * last_price
            if not _close(mv, expected_mv, rel=1e-6, abs_tol=0.02):
                warnings.append(f"{sym} market_value mismatch (got {mv}, expected {expected_mv})")

        if shares and cost_basis is not None and avg_cost is not None:
            expected_avg = cost_basis / shares
            if not _close(avg_cost, expected_avg, rel=1e-6, abs_tol=0.01):
                warnings.append(f"{sym} avg_cost mismatch (got {avg_cost}, expected {expected_avg})")

        if mv is not None and cost_basis is not None and unreal is not None:
            expected_unreal = mv - cost_basis
            if not _close(unreal, expected_unreal, rel=1e-6, abs_tol=0.05):
                warnings.append(f"{sym} unrealized_pnl mismatch (got {unreal}, expected {expected_unreal})")

        if cost_basis and unreal is not None and unreal_pct is not None:
            expected_unreal_pct = (unreal / cost_basis) * 100.0
            if not _close(unreal_pct, expected_unreal_pct, rel=1e-5, abs_tol=0.05):
                warnings.append(f"{sym} unrealized_pct mismatch (got {unreal_pct}, expected {expected_unreal_pct})")

    totals = snap.get("totals") or {}
    totals_mv = _as_float(totals.get("market_value"))
    totals_cb = _as_float(totals.get("cost_basis"))
    totals_unreal = _as_float(totals.get("unrealized_pnl"))
    totals_unreal_pct = _as_float(totals.get("unrealized_pct"))
    total_market_value = _as_float(snap.get("total_market_value"))

    if totals_mv is not None and not _close(totals_mv, sum_mv, rel=1e-6, abs_tol=0.05):
        warnings.append(f"totals.market_value mismatch (got {totals_mv}, expected {sum_mv})")
    if totals_cb is not None and not _close(totals_cb, sum_cb, rel=1e-6, abs_tol=0.05):
        warnings.append(f"totals.cost_basis mismatch (got {totals_cb}, expected {sum_cb})")
    if totals_unreal is not None and not _close(totals_unreal, sum_unreal, rel=1e-6, abs_tol=0.1):
        warnings.append(f"totals.unrealized_pnl mismatch (got {totals_unreal}, expected {sum_unreal})")
    if total_market_value is not None and totals_mv is not None and not _close(total_market_value, totals_mv, rel=1e-6, abs_tol=0.05):
        warnings.append(f"total_market_value mismatch (got {total_market_value}, expected {totals_mv})")
    if totals_cb and totals_unreal is not None and totals_unreal_pct is not None:
        expected_pct = (totals_unreal / totals_cb) * 100.0
        if not _close(totals_unreal_pct, expected_pct, rel=1e-5, abs_tol=0.05):
            warnings.append(f"totals.unrealized_pct mismatch (got {totals_unreal_pct}, expected {expected_pct})")

    if weight_count:
        if abs(sum_weight - 100.0) > 0.5:
            warnings.append(f"weight_pct sum off 100 (sum {sum_weight})")

    count_val = snap.get("count")
    if isinstance(count_val, int) and count_val != len(holdings):
        warnings.append(f"count mismatch (got {count_val}, expected {len(holdings)})")

    missing_prices = snap.get("missing_prices") or []
    coverage = snap.get("coverage") or {}
    coverage_missing_pct = _as_float(coverage.get("missing_pct"))
    if isinstance(missing_prices, list) and holdings:
        expected_missing_pct = (len(missing_prices) / max(len(holdings), 1)) * 100.0
        if coverage_missing_pct is not None and not _close(
            coverage_missing_pct, expected_missing_pct, rel=1e-4, abs_tol=0.2
        ):
            warnings.append(
                f"coverage.missing_pct mismatch (got {coverage_missing_pct}, expected {expected_missing_pct})"
            )

    as_of_utc = _parse_iso(snap.get("as_of_utc"))
    prices_as_of_utc = _parse_iso(snap.get("prices_as_of_utc"))
    quote_ttl_minutes = float(getattr(settings, "quote_ttl_minutes", 0) or 0)
    if quote_ttl_minutes > 0:
        if prices_as_of_utc and as_of_utc:
            age_min = (as_of_utc - prices_as_of_utc).total_seconds() / 60.0
            max_age = max(quote_ttl_minutes * 1.5, 90.0)
            if age_min > max_age:
                warnings.append(f"prices_as_of_utc stale by {round(age_min, 1)} minutes")
        elif as_of_utc:
            warnings.append("prices_as_of_utc missing (intraday quotes not captured)")

    as_of_date_local = _parse_date(snap.get("as_of_date_local"))
    prices_as_of = _parse_date(snap.get("prices_as_of"))
    if as_of_date_local and prices_as_of:
        day_delta = (as_of_date_local - prices_as_of).days
        if day_delta > 1:
            warnings.append(f"prices_as_of date lags by {day_delta} days")

    return errors, warnings


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_latest_daily(conn):
    from app.pipeline.snapshot_views import assemble_daily_snapshot
    return assemble_daily_snapshot(conn, as_of_date=None)


def _load_latest_periods(conn):
    from app.pipeline.snapshot_views import assemble_period_snapshot
    cur = conn.execute(
        "SELECT period_type, period_end_date, period_start_date FROM period_summary WHERE is_rolling=0 ORDER BY period_end_date DESC"
    )
    rows = cur.fetchall()
    latest: dict[str, dict] = {}
    for row in rows:
        period_type, end_date, start_date = row[0], row[1], row[2]
        if period_type in latest:
            continue
        snap = assemble_period_snapshot(conn, period_type, end_date, period_start_date=start_date)
        if snap:
            latest[period_type] = snap
    return latest


def main():
    parser = argparse.ArgumentParser(description="Validate pipeline outputs without running sync.")
    parser.add_argument("--daily-file", help="Path to a daily snapshot JSON file.")
    parser.add_argument("--period-file", action="append", help="Path to a period snapshot JSON file.")
    parser.add_argument("--skip-period", action="store_true", help="Skip period snapshot validation.")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    conn = None
    if not args.daily_file or (not args.skip_period and not args.period_file):
        conn = get_conn(settings.db_path)
        migrate(conn)

    # Daily snapshot validation
    daily = None
    if args.daily_file:
        daily = _load_json(args.daily_file)
    elif conn is not None:
        daily = _load_latest_daily(conn)
    if daily is None:
        errors.append("no daily snapshot found")
    else:
        ok, reasons = validate_daily_snapshot(daily)
        if not ok:
            errors.extend([f"daily validation: {r}" for r in reasons])
        extra_err, extra_warn = _check_daily_math(daily)
        errors.extend(extra_err)
        warnings.extend(extra_warn)

    # Period snapshot validation
    if not args.skip_period:
        period_snaps = []
        if args.period_file:
            for path in args.period_file:
                period_snaps.append((path, _load_json(path)))
        elif conn is not None:
            latest = _load_latest_periods(conn)
            for key, snap in latest.items():
                period_snaps.append((f"latest:{key}", snap))

        for label, snap in period_snaps:
            ok, reasons = validate_period_snapshot(snap)
            if not ok:
                errors.extend([f"period {label}: {r}" for r in reasons])

    if conn is not None:
        conn.close()

    if errors:
        print("pipeline validation: FAIL")
        for msg in errors:
            print(f"ERROR: {msg}")
    else:
        print("pipeline validation: OK")

    for msg in warnings:
        print(f"WARN: {msg}")

    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
