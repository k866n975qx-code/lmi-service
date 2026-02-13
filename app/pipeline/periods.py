from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from collections import defaultdict

import pandas as pd
import numpy as np

from ..config import settings
from ..utils import to_local_date
from .market import MarketData
from . import metrics
from .diff_daily import (
    _totals,
    _income,
    _perf,
    _risk_flat,
    _rollups,
    _goal_progress,
    _goal_progress_net,
    _margin_stress,
    _coverage,
    _holdings_flat,
)
from ..services.ai_insights import _safe_get




def _as_of(snap: dict | None) -> str | None:
    if not snap:
        return None
    return snap.get("as_of_date_local") or (snap.get("timestamps") or {}).get("portfolio_data_as_of_local")


def _goal_pace(snap: dict | None) -> dict:
    if not snap:
        return {}
    return (snap.get("goals") or {}).get("pace") or {}


def _goal_tiers(snap: dict | None) -> dict:
    if not snap:
        return {}
    g = snap.get("goals") or {}
    return {"tiers": g.get("tiers"), "current_state": g.get("current_state"), "provenance": g.get("tiers_provenance")}


def _macro_snapshot(snap: dict | None) -> dict:
    if not snap:
        return {}
    return (snap.get("macro") or {}).get("snapshot") or {}


def _parse_date(val):
    if not val:
        return None
    text = str(val).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _total_market_value(snapshot: dict | None):
    if not isinstance(snapshot, dict):
        return None
    val = snapshot.get("total_market_value") or snapshot.get("market_value")
    if isinstance(val, (int, float)):
        return float(val)
    totals = _totals(snapshot)
    val = totals.get("market_value")
    return float(val) if isinstance(val, (int, float)) else None


def _strip_provenance(value):
    if isinstance(value, dict):
        return {
            key: _strip_provenance(val)
            for key, val in value.items()
            if key != "provenance" and not key.endswith("_provenance")
        }
    if isinstance(value, list):
        return [_strip_provenance(item) for item in value]
    return value


def _holding_metric(holding: dict, key: str):
    if not isinstance(holding, dict):
        return None
    val = holding.get(key)
    if val is None and holding.get("valuation"):
        v = holding["valuation"]
        val = v.get("portfolio_weight_pct" if key == "weight_pct" else key)
    if val is None and holding.get("income") and key in ("projected_monthly_dividend", "current_yield_pct", "yield_on_cost_pct"):
        val = (holding.get("income") or {}).get(key)
    if val is None and holding.get("analytics"):
        val = ((holding.get("analytics") or {}).get("risk") or {}).get(key)
    return val


def _coverage_summary(dailies):
    derived = []
    pulled = []
    for snap in dailies:
        cov = _coverage(snap)
        if isinstance(cov.get("derived_pct"), (int, float)):
            derived.append(float(cov.get("derived_pct")))
        if isinstance(cov.get("pulled_pct"), (int, float)):
            pulled.append(float(cov.get("pulled_pct")))
    def _avg(vals):
        if not vals:
            return None
        return round(sum(vals) / len(vals), 2)
    return {
        "derived_pct": _avg(derived),
        "pulled_pct": _avg(pulled),
    }


def _calc_period_stats(values_list: list, values_dict: dict) -> dict:
    """Calculate avg, min, max, std with dates for a list of values."""
    if not values_list:
        return {"avg": None, "min": None, "max": None, "std": None, "min_date": None, "max_date": None}
    avg = round(sum(values_list) / len(values_list), 2)
    min_val = min(values_list)
    max_val = max(values_list)
    # Calculate sample standard deviation
    std = round(np.std(values_list, ddof=1), 2) if len(values_list) > 1 else None
    # Find dates for min/max (skip None values and non-numeric)
    min_date = None
    max_date = None
    for dt, val in values_dict.items():
        if isinstance(val, (int, float)):
            if abs(val - min_val) < 0.01:
                min_date = dt
            if abs(val - max_val) < 0.01:
                max_date = dt
    return {
        "avg": avg,
        "min": min_val,
        "max": max_val,
        "std": std,
        "min_date": min_date.isoformat() if min_date else None,
        "max_date": max_date.isoformat() if max_date else None,
    }


def _safe_avg(values_list: list) -> float | None:
    """Calculate average safely, returning None for empty list."""
    return round(sum(values_list) / len(values_list), 2) if values_list else None


def _calc_period_drawdown(series: pd.Series) -> dict:
    """Calculate period-specific drawdown metrics from portfolio series."""
    if series is None or series.empty or len(series) < 2:
        return {}
    
    # Calculate running maximum
    running_max = series.cummax()
    # Calculate drawdown at each point
    drawdown = (series - running_max) / running_max * 100
    # Find max drawdown for period
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin() if hasattr(drawdown.idxmin, "__call__") else None
    
    # Find recovery date (when portfolio exceeds previous peak after max drawdown)
    recovery_date = None
    if max_dd_date is not None:
        peak_before_dd = running_max.loc[max_dd_date]
        after_dd = series.loc[max_dd_date:]
        recovered = after_dd[after_dd >= peak_before_dd * 0.99]  # Within 1% of peak
        if not recovered.empty:
            recovery_date = recovered.index[0]
    
    # Count drawdown episodes (transitions into drawdown state)
    in_drawdown = drawdown < 0
    drawdown_count = int((in_drawdown & ~in_drawdown.shift(fill_value=False)).sum())
    
    # Calculate days in max drawdown
    days_in_drawdown = None
    if max_dd_date is not None and recovery_date is not None:
        days_in_drawdown = (recovery_date - max_dd_date).days
    
    return {
        "period_max_drawdown_pct": round(float(max_dd), 2) if pd.notna(max_dd) else None,
        "period_max_drawdown_date": max_dd_date.strftime("%Y-%m-%d") if max_dd_date is not None else None,
        "period_recovery_date": recovery_date.strftime("%Y-%m-%d") if recovery_date is not None else None,
        "period_days_in_drawdown": days_in_drawdown,
        "period_drawdown_count": int(drawdown_count),
    }


def _period_bounds(snapshot_type: str, on: date):
    if snapshot_type == "weekly":
        start = on - timedelta(days=on.weekday())
        end = start + timedelta(days=6)
    elif snapshot_type == "monthly":
        start = on.replace(day=1)
        if start.month == 12:
            end = date(start.year, 12, 31)
        else:
            end = start.replace(month=start.month + 1, day=1) - timedelta(days=1)
    elif snapshot_type == "quarterly":
        q = (on.month - 1) // 3
        start_month = 1 + q * 3
        start = date(on.year, start_month, 1)
        if start_month == 10:
            end = date(on.year, 12, 31)
        else:
            end = date(on.year, start_month + 3, 1) - timedelta(days=1)
    elif snapshot_type == "yearly":
        start = date(on.year, 1, 1)
        end = date(on.year, 12, 31)
    else:
        raise ValueError(f"unknown snapshot_type {snapshot_type}")
    return start, end


def _period_label(snapshot_type: str, on: date):
    if snapshot_type == "weekly":
        iso = on.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if snapshot_type == "monthly":
        return f"{on.year}-{on.month:02d}"
    if snapshot_type == "quarterly":
        q = (on.month - 1) // 3 + 1
        return f"{on.year}-Q{q}"
    if snapshot_type == "yearly":
        return f"{on.year}"
    return on.isoformat()


def _expected_days(snapshot_type: str, start: date, end: date):
    if snapshot_type == "weekly":
        return 7
    if snapshot_type == "monthly":
        return (end - start).days + 1
    if snapshot_type == "quarterly":
        return (end - start).days + 1
    if snapshot_type == "yearly":
        return (end - start).days + 1
    return (end - start).days + 1


def _load_daily_snapshots(conn: sqlite3.Connection, start: date, end: date):
    """Load daily snapshots for date range. Prefer flat tables (rebuilt from live-filled data)."""
    from .snapshot_views import assemble_daily_snapshot

    cur = conn.cursor()
    flat_dates = cur.execute(
        """
        SELECT as_of_date_local FROM daily_portfolio
        WHERE as_of_date_local BETWEEN ? AND ?
        ORDER BY as_of_date_local ASC
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    snapshots = []
    for (as_of_date,) in flat_dates:
        snap = assemble_daily_snapshot(conn, as_of_date)
        if snap:
            snap["as_of_date_local"] = as_of_date
            snapshots.append(snap)
    return snapshots


def _load_period_snapshots(conn: sqlite3.Connection, period_type: str, start: date, end: date):
    """Load period snapshots from period_summary (flat), assembled to full snapshot shape."""
    from .snapshot_views import assemble_period_snapshot

    period_type_upper = period_type.upper()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT period_start_date, period_end_date
        FROM period_summary
        WHERE period_type=? AND is_rolling=0 AND period_end_date BETWEEN ? AND ?
        ORDER BY period_end_date ASC
        """,
        (period_type_upper, start.isoformat(), end.isoformat()),
    ).fetchall()
    out = []
    for start_date, end_date in rows:
        snap = assemble_period_snapshot(conn, period_type_upper, end_date, period_start_date=start_date, rolling=False)
        if snap and isinstance(snap, dict):
            snap["_period_start"] = start_date
            snap["_period_end"] = end_date
            out.append(snap)
    return out


def _load_daily_snapshot(conn: sqlite3.Connection, as_of: date):
    """Load single daily snapshot from flat tables (assembled to V5)."""
    from .snapshot_views import assemble_daily_snapshot
    return assemble_daily_snapshot(conn, as_of.isoformat())


def _daily_series(dailies, key_path):
    data = {}
    for snap in dailies:
        as_of = _as_of(snap)
        dt = _parse_date(as_of)
        if not dt:
            continue
        val = snap
        for part in key_path:
            if not isinstance(val, dict):
                val = None
                break
            val = val.get(part)
        if isinstance(val, (int, float)):
            data[dt] = float(val)
    if not data:
        return pd.Series(dtype=float)
    series = pd.Series(data).sort_index()
    series.index = pd.to_datetime(series.index)
    return series


def _pct(val):
    return None if val is None else round(val * 100, 3)


def _round(val, precision=2):
    return None if val is None else round(float(val), precision)


def _top5_concentration(holdings):
    weights = [h.get("weight_pct") for h in holdings if isinstance(h.get("weight_pct"), (int, float))]
    weights = sorted(weights, reverse=True)
    if not weights:
        return None
    return round(sum(weights[:5]), 2)


def _benchmark_block(symbol: str, start: date, end: date):
    if not symbol:
        return {"symbol": None, "period_return_pct": None, "twr_1y_pct_start": None, "twr_1y_pct_end": None, "twr_1y_pct_delta": None}
    md = MarketData()
    md.load([symbol])
    series = None
    df = md.prices.get(symbol)
    if df is not None and not getattr(df, "empty", True):
        if "date" in df.columns:
            series = df.set_index("date")["adj_close" if "adj_close" in df.columns else "close"]
        else:
            series = df["adj_close" if "adj_close" in df.columns else "close"]
        series = series.dropna()
        series.index = pd.to_datetime(series.index)
    if series is None or series.empty:
        return {"symbol": symbol, "period_return_pct": None, "twr_1y_pct_start": None, "twr_1y_pct_end": None, "twr_1y_pct_delta": None}

    period = series[(series.index.date >= start) & (series.index.date <= end)]
    period_return = None
    if not period.empty:
        period_return = (period.iloc[-1] / period.iloc[0] - 1.0) * 100

    def _twr_trailing(end_date: date):
        window_start = end_date - timedelta(days=365)
        sub = series[(series.index.date >= window_start) & (series.index.date <= end_date)]
        if sub.empty:
            return None
        return (sub.iloc[-1] / sub.iloc[0] - 1.0) * 100

    twr_start = _twr_trailing(start)
    twr_end = _twr_trailing(end)
    twr_delta = None if twr_start is None or twr_end is None else round(twr_end - twr_start, 2)

    return {
        "symbol": symbol,
        "period_return_pct": _round(period_return),
        "twr_1y_pct_start": _round(twr_start),
        "twr_1y_pct_end": _round(twr_end),
        "twr_1y_pct_delta": _round(twr_delta),
    }


def _portfolio_changes(start_holdings, end_holdings, dailies=None):
    """Calculate portfolio changes including new enhanced metrics for period_holding_changes table.

    Args:
        start_holdings: Holdings at period start
        end_holdings: Holdings at period end
        dailies: List of daily snapshots for calculating period metrics (optional)

    Returns:
        Dict with holdings_added, holdings_removed, weight_increases, weight_decreases, top_gainers, top_losers
    """
    start_map = {h.get("symbol"): h for h in start_holdings if h.get("symbol")}
    end_map = {h.get("symbol"): h for h in end_holdings if h.get("symbol")}
    symbols = set(start_map.keys()) | set(end_map.keys())
    added = []
    removed = []
    increases = []
    decreases = []
    gainers = []
    losers = []

    for sym in symbols:
        s = start_map.get(sym)
        e = end_map.get(sym)

        # Base entry with all enhanced fields
        entry = {"symbol": sym}

        # Weight metrics
        if s:
            entry["weight_start_pct"] = s.get("weight_pct")
        if e:
            entry["weight_end_pct"] = e.get("weight_pct")
        if s and e:
            w_start = s.get("weight_pct")
            w_end = e.get("weight_pct")
            if isinstance(w_start, (int, float)) and isinstance(w_end, (int, float)):
                entry["weight_delta_pct"] = round(w_end - w_start, 2)

        # Calculate avg_weight_pct across period if dailies provided
        if dailies:
            weights = []
            for d in dailies:
                holdings = _holdings_flat(d) or []
                h = next((h for h in holdings if h.get("symbol") == sym), None)
                if h and isinstance(h.get("weight_pct"), (int, float)):
                    weights.append(h["weight_pct"])
            if weights:
                entry["avg_weight_pct"] = round(sum(weights) / len(weights), 2)

        # Share metrics
        if s:
            entry["start_shares"] = s.get("shares")
        if e:
            entry["end_shares"] = e.get("shares")
        if s and e and s.get("shares") and e.get("shares"):
            delta = e["shares"] - s["shares"]
            entry["shares_delta"] = round(delta, 4)
            if s["shares"]:
                entry["shares_delta_pct"] = round((delta / s["shares"]) * 100, 2)

        # Market value metrics
        if s:
            entry["start_market_value"] = s.get("market_value")
        if e:
            entry["end_market_value"] = e.get("market_value")
        if s and e:
            mv_start = s.get("market_value")
            mv_end = e.get("market_value")
            if isinstance(mv_start, (int, float)) and isinstance(mv_end, (int, float)):
                delta = mv_end - mv_start
                entry["market_value_delta"] = round(delta, 2)
                if mv_start:
                    entry["market_value_delta_pct"] = round((delta / mv_start) * 100, 2)
                    pnl_pct = ((mv_end / mv_start) - 1.0) * 100
                    entry["pnl_pct_period"] = round(pnl_pct, 2)
                    entry["pnl_dollar_period"] = round(delta, 2)

                    # Contribution to portfolio = avg_weight * period_return
                    if entry.get("avg_weight_pct") is not None:
                        contribution = (entry["avg_weight_pct"] / 100) * pnl_pct
                        entry["contribution_to_portfolio_pct"] = round(contribution, 2)

        # Performance metrics
        if s:
            entry["start_twr_12m_pct"] = _safe_get(s, "analytics", "performance", "twr_12m_pct")
        if e:
            entry["end_twr_12m_pct"] = _safe_get(e, "analytics", "performance", "twr_12m_pct")

        # Income metrics
        if s:
            entry["start_yield_pct"] = _safe_get(s, "income", "current_yield_pct")
            entry["start_projected_monthly"] = _safe_get(s, "income", "projected_monthly_dividend")
        if e:
            entry["end_yield_pct"] = _safe_get(e, "income", "current_yield_pct")
            entry["end_projected_monthly"] = _safe_get(e, "income", "projected_monthly_dividend")

        # Risk metrics from dailies
        if dailies:
            # Calculate avg volatility
            vols = []
            for d in dailies:
                holdings = _holdings_flat(d) or []
                h = next((h for h in holdings if h.get("symbol") == sym), None)
                if h:
                    vol = _safe_get(h, "analytics", "risk", "vol_30d_pct")
                    if isinstance(vol, (int, float)):
                        vols.append(vol)
            if vols:
                entry["avg_vol_30d_pct"] = round(sum(vols) / len(vols), 2)

            # Calculate period drawdown and best/worst days
            symbol_values = []
            for d in dailies:
                holdings = _holdings_flat(d) or []
                h = next((h for h in holdings if h.get("symbol") == sym), None)
                if h and isinstance(h.get("market_value"), (int, float)):
                    symbol_values.append((h["market_value"], _as_of(d)))

            if len(symbol_values) > 1:
                # Drawdown
                peak = symbol_values[0][0]
                max_dd = 0
                for val, _ in symbol_values:
                    if val > peak:
                        peak = val
                    dd = ((val - peak) / peak) * 100 if peak else 0
                    if dd < max_dd:
                        max_dd = dd
                if max_dd < 0:
                    entry["period_max_drawdown_pct"] = round(max_dd, 2)

                # Best/worst days
                daily_returns = []
                for i in range(1, len(symbol_values)):
                    prev_val, _ = symbol_values[i-1]
                    curr_val, curr_date = symbol_values[i]
                    if prev_val:
                        ret_pct = ((curr_val - prev_val) / prev_val) * 100
                        daily_returns.append((ret_pct, curr_date))

                if daily_returns:
                    worst = min(daily_returns, key=lambda x: x[0])
                    best = max(daily_returns, key=lambda x: x[0])
                    entry["worst_day_pct"] = round(worst[0], 2)
                    entry["worst_day_date"] = worst[1]
                    entry["best_day_pct"] = round(best[0], 2)
                    entry["best_day_date"] = best[1]

        # Categorize
        if s is None and e is not None:
            added.append(entry)
        elif e is None and s is not None:
            removed.append(entry)
        else:
            w_start = s.get("weight_pct")
            w_end = e.get("weight_pct")
            if isinstance(w_start, (int, float)) and isinstance(w_end, (int, float)):
                delta = w_end - w_start
                if delta > 0.01:
                    increases.append(entry)
                elif delta < -0.01:
                    decreases.append(entry)

        # Top gainers/losers
        if entry.get("pnl_dollar_period") is not None:
            if entry["pnl_dollar_period"] >= 0:
                gainers.append(entry)
            else:
                losers.append(entry)

    gainers = sorted(gainers, key=lambda x: x.get("pnl_dollar_period", 0), reverse=True)[:5]
    losers = sorted(losers, key=lambda x: x.get("pnl_dollar_period", 0))[:5]

    return {
        "holdings_added": added,
        "holdings_removed": removed,
        "weight_increases": increases,
        "weight_decreases": decreases,
        "top_gainers": gainers,
        "top_losers": losers,
    }


def _generate_period_activity(conn: sqlite3.Connection, start: date, end: date):
    """Generate activity section for a period from investment_transactions."""
    cur = conn.cursor()

    # Query all transactions in the period
    rows = cur.execute(
        """
        SELECT date, amount, transaction_type, symbol, plaid_account_id
        FROM investment_transactions
        WHERE date BETWEEN ? AND ?
        ORDER BY date
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchall()

    contributions = []
    withdrawals = []
    dividends = []
    interest = []
    margin_borrows = []
    margin_repays = []
    trades_by_symbol = {}

    for row in rows:
        tx_date, amount, tx_type, symbol, account_id = row
        if not tx_date:
            continue

        tx_dict = {
            "date": tx_date,
            "amount": float(amount) if isinstance(amount, (int, float)) else 0.0,
            "symbol": symbol,
        }

        if tx_type == "contribution":
            contrib_dict = {**tx_dict, "account_id": account_id}
            contributions.append(contrib_dict)
        elif tx_type == "withdrawal":
            withdrawal_dict = {**tx_dict, "account_id": account_id}
            withdrawals.append(withdrawal_dict)
        elif tx_type == "dividend":
            dividends.append(tx_dict)
        elif tx_type == "interest":
            interest.append(tx_dict)
        elif tx_type == "margin_borrow":
            margin_borrows.append(tx_dict)
        elif tx_type == "margin_repay":
            margin_repays.append(tx_dict)
        elif tx_type in ("buy", "sell"):
            if not symbol:
                continue
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = {"buy_count": 0, "sell_count": 0}
            if tx_type == "buy":
                trades_by_symbol[symbol]["buy_count"] += 1
            elif tx_type == "sell":
                trades_by_symbol[symbol]["sell_count"] += 1

    # Calculate totals
    contributions_total = sum(c["amount"] for c in contributions)
    withdrawals_total = sum(w["amount"] for w in withdrawals)
    dividends_total = sum(d["amount"] for d in dividends)
    interest_total = sum(i["amount"] for i in interest)

    total_buys = sum(t["buy_count"] for t in trades_by_symbol.values())
    total_sells = sum(t["sell_count"] for t in trades_by_symbol.values())

    # Group dividends by symbol
    dividends_by_symbol = {}
    for d in dividends:
        sym = d.get("symbol")
        if sym:
            if sym not in dividends_by_symbol:
                dividends_by_symbol[sym] = 0.0
            dividends_by_symbol[sym] += d["amount"]

    # Deduplicate dividend events by (symbol, ex_date) and combine amounts
    dividend_events_dict = {}
    for d in dividends:
        sym = d.get("symbol")
        ex_date = d.get("date")
        if sym and ex_date:
            key = (sym, ex_date)
            if key in dividend_events_dict:
                # Sum amounts for same symbol+ex_date
                dividend_events_dict[key]["amount"] += d["amount"]
            else:
                dividend_events_dict[key] = {
                    "symbol": sym,
                    "ex_date": ex_date,
                    "pay_date": None,
                    "amount": d["amount"]
                }
    deduped_events = list(dividend_events_dict.values())

    # Calculate margin activity
    margin_borrowed_total = sum(abs(m["amount"]) for m in margin_borrows)
    margin_repaid_total = sum(abs(m["amount"]) for m in margin_repays)
    margin_net_change = margin_borrowed_total - margin_repaid_total

    # Calculate position changes by comparing holdings at start vs end
    # Get holdings at start of period (day before period starts)
    start_holdings_rows = cur.execute(
        """
        SELECT symbol, shares
        FROM daily_holdings
        WHERE as_of_date_local = ?
        """,
        ((start - timedelta(days=1)).isoformat(),)
    ).fetchall()

    start_holdings = {row[0]: row[1] for row in start_holdings_rows if row[0] and row[1]}

    # Get holdings at end of period
    end_holdings_rows = cur.execute(
        """
        SELECT symbol, shares
        FROM daily_holdings
        WHERE as_of_date_local = ?
        """,
        (end.isoformat(),)
    ).fetchall()

    end_holdings = {row[0]: row[1] for row in end_holdings_rows if row[0] and row[1]}

    # Calculate position changes
    all_symbols = set(start_holdings.keys()) | set(end_holdings.keys())
    positions_added = []
    positions_removed = []
    positions_increased = []
    positions_decreased = []

    for symbol in all_symbols:
        start_shares = start_holdings.get(symbol, 0)
        end_shares = end_holdings.get(symbol, 0)

        if start_shares == 0 and end_shares > 0:
            positions_added.append(symbol)
        elif start_shares > 0 and end_shares == 0:
            positions_removed.append(symbol)
        elif end_shares > start_shares:
            positions_increased.append(symbol)
        elif end_shares < start_shares:
            positions_decreased.append(symbol)

    return {
        "contributions": {
            "total": contributions_total,
            "count": len(contributions),
            "dates": [c["date"] for c in contributions],
            "details": contributions,
        },
        "withdrawals": {
            "total": withdrawals_total,
            "count": len(withdrawals),
            "dates": [w["date"] for w in withdrawals],
            "details": withdrawals,
        },
        "dividends": {
            "total_received": dividends_total,
            "count": len(dividends),
            "by_symbol": dividends_by_symbol,
            "events": deduped_events,
        },
        "interest": {
            "total_paid": abs(interest_total),
            "avg_daily_balance": None,  # Requires daily margin balance series
            "avg_rate_pct": None,
            "annualized": None,
        },
        "trades": {
            "total_count": total_buys + total_sells,
            "buy_count": total_buys,
            "sell_count": total_sells,
            "by_symbol": trades_by_symbol,
        },
        "positions": {
            "added": positions_added,
            "removed": positions_removed,
            "symbols_increased": positions_increased,
            "symbols_decreased": positions_decreased,
        },
        "margin": {
            "borrowed": margin_borrowed_total,
            "repaid": margin_repaid_total,
            "net_change": margin_net_change,
        },
    }


def _interval_label(snapshot_type: str, start: date, end: date):
    if snapshot_type == "weekly":
        return end.isoformat()
    if snapshot_type == "monthly":
        iso = end.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return f"{end.year}-{end.month:02d}"


def _intervals(dailies, snapshot_type: str, as_of_date: date, as_of_daily: dict | None = None):
    groups = defaultdict(list)
    for snap in dailies:
        dt = _parse_date(_as_of(snap))
        if not dt:
            continue
        if snapshot_type == "weekly":
            key = dt
        elif snapshot_type == "monthly":
            key = dt.isocalendar().week
        else:
            key = (dt.year, dt.month)
        groups[key].append((dt, snap))

    intervals = []
    for key, items in sorted(groups.items(), key=lambda x: x[0]):
        items.sort(key=lambda x: x[0])
        start_dt, start_snap = items[0]
        end_dt, end_snap = items[-1]
        start_totals = _totals(start_snap) if start_snap else {}
        end_totals = _totals(end_snap) if end_snap else {}
        end_perf = _perf(end_snap)
        end_risk = _risk_flat(end_snap)
        end_rollups = _rollups(end_snap)
        start_mv = _total_market_value(start_snap)
        end_mv = _total_market_value(end_snap)
        pnl = None
        pnl_pct = None
        if isinstance(start_mv, (int, float)) and isinstance(end_mv, (int, float)) and start_mv:
            pnl = end_mv - start_mv
            pnl_pct = (end_mv / start_mv - 1.0) * 100
        margin_loan = end_totals.get("margin_loan_balance")
        net_liquidation_value = None
        if isinstance(end_mv, (int, float)) and isinstance(margin_loan, (int, float)):
            net_liquidation_value = end_mv - margin_loan

        interval = {
            "interval_label": _interval_label(snapshot_type, start_dt, end_dt),
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "totals": {
                "total_market_value": end_mv,
                "cost_basis": end_totals.get("cost_basis"),
                "net_liquidation_value": _round(net_liquidation_value),
                "unrealized_pct": end_totals.get("unrealized_pct"),
                "unrealized_pnl": end_totals.get("unrealized_pnl"),
            },
            "income": _income(end_snap),
            "performance": {
                "pnl_dollar_period": _round(pnl),
                "pnl_pct_period": _round(pnl_pct),
                "twr_1m_pct": end_perf.get("twr_1m_pct"),
                "twr_3m_pct": end_perf.get("twr_3m_pct"),
                "twr_6m_pct": end_perf.get("twr_6m_pct"),
                "twr_12m_pct": end_perf.get("twr_12m_pct"),
            },
            "risk": {
                "vol_30d_pct": end_risk.get("vol_30d_pct"),
                "vol_90d_pct": end_risk.get("vol_90d_pct"),
                "sharpe_1y": end_risk.get("sharpe_1y"),
                "sortino_1y": end_risk.get("sortino_1y"),
                "sortino_6m": end_risk.get("sortino_6m"),
                "sortino_3m": end_risk.get("sortino_3m"),
                "sortino_1m": end_risk.get("sortino_1m"),
                "sortino_sharpe_ratio": end_risk.get("sortino_sharpe_ratio"),
                "sortino_sharpe_divergence": end_risk.get("sortino_sharpe_divergence"),
                "portfolio_risk_quality": end_risk.get("portfolio_risk_quality"),
                "var_90_1d_pct": end_risk.get("var_90_1d_pct"),
                "var_95_1d_pct": end_risk.get("var_95_1d_pct"),
                "var_99_1d_pct": end_risk.get("var_99_1d_pct"),
                "var_95_1w_pct": end_risk.get("var_95_1w_pct"),
                "var_95_1m_pct": end_risk.get("var_95_1m_pct"),
                "cvar_90_1d_pct": end_risk.get("cvar_90_1d_pct"),
                "cvar_95_1d_pct": end_risk.get("cvar_95_1d_pct"),
                "cvar_99_1d_pct": end_risk.get("cvar_99_1d_pct"),
                "cvar_95_1w_pct": end_risk.get("cvar_95_1w_pct"),
                "cvar_95_1m_pct": end_risk.get("cvar_95_1m_pct"),
                "ulcer_index_1y": end_risk.get("ulcer_index_1y"),
                "omega_ratio_1y": end_risk.get("omega_ratio_1y"),
                "information_ratio_1y": end_risk.get("information_ratio_1y"),
                "tracking_error_1y_pct": end_risk.get("tracking_error_1y_pct"),
                "income_stability_score": end_risk.get("income_stability_score"),
                "max_drawdown_1y_pct": end_risk.get("max_drawdown_1y_pct"),
                "drawdown_duration_1y_days": end_risk.get("drawdown_duration_1y_days"),
                "calmar_1y": end_risk.get("calmar_1y"),
                "beta_portfolio": end_risk.get("beta_portfolio"),
            },
            "income_stability": end_rollups.get("income_stability"),
            "income_growth": end_rollups.get("income_growth"),
            "tail_risk": end_rollups.get("tail_risk"),
            "vs_benchmark": end_rollups.get("vs_benchmark"),
            "return_attribution_1m": end_rollups.get("return_attribution_1m"),
            "return_attribution_3m": end_rollups.get("return_attribution_3m"),
            "return_attribution_6m": end_rollups.get("return_attribution_6m"),
            "return_attribution_12m": end_rollups.get("return_attribution_12m"),
            "margin": {
                "margin_loan_balance": end_totals.get("margin_loan_balance"),
                "margin_to_portfolio_pct": end_totals.get("margin_to_portfolio_pct") or end_totals.get("ltv_pct"),
                "ltv_pct": end_totals.get("margin_to_portfolio_pct") or end_totals.get("ltv_pct"),
                "available_to_withdraw": 0.0 if end_totals.get("margin_loan_balance") is not None else None,
            },
            "margin_stress": _margin_stress(end_snap),
            "goal_progress": _goal_progress(end_snap),
            "goal_progress_net": _goal_progress_net(end_snap),
            "goal_tiers": _goal_tiers(end_snap),
            "goal_pace": _goal_pace(end_snap),
            "holdings": [
                {
                    "symbol": h.get("symbol"),
                    "weight_pct": _holding_metric(h, "weight_pct"),
                    "market_value": _holding_metric(h, "market_value"),
                    "pnl_pct": _holding_metric(h, "unrealized_pct"),
                    "pnl_dollar": _holding_metric(h, "unrealized_pnl"),
                    "projected_monthly_dividend": _holding_metric(h, "projected_monthly_dividend"),
                    "current_yield_pct": _holding_metric(h, "current_yield_pct"),
                    "yield_on_cost_pct": _holding_metric(h, "yield_on_cost_pct"),
                    "sharpe_1y": _holding_metric(h, "sharpe_1y"),
                    "sortino_1y": _holding_metric(h, "sortino_1y"),
                    "sortino_6m": _holding_metric(h, "sortino_6m"),
                    "sortino_3m": _holding_metric(h, "sortino_3m"),
                    "sortino_1m": _holding_metric(h, "sortino_1m"),
                    "risk_quality_score": _holding_metric(h, "risk_quality_score"),
                    "risk_quality_category": _holding_metric(h, "risk_quality_category"),
                    "volatility_profile": _holding_metric(h, "volatility_profile"),
                }
                for h in _holdings_flat(end_snap)
            ],
        }
        if end_dt == as_of_date:
            interval["daily_snapshot"] = _strip_provenance(as_of_daily or end_snap)
        intervals.append(interval)

    return intervals


def _extract_daily_from_period_snap(period_snap: dict, fallback_end: str | None = None):
    intervals = period_snap.get("intervals") or []
    if fallback_end:
        for interval in intervals:
            if interval.get("end_date") == fallback_end and interval.get("daily_snapshot"):
                return interval.get("daily_snapshot")
    for interval in reversed(intervals):
        if interval.get("daily_snapshot"):
            return interval.get("daily_snapshot")
    return period_snap.get("daily_snapshot")


def _intervals_from_periods(period_snaps: list[dict], snapshot_type: str, as_of_date: date, as_of_daily: dict | None = None):
    intervals = []
    for snap in period_snaps:
        period = snap.get("period") or {}
        start_date = period.get("start_date")
        end_date = period.get("end_date")
        end_dt = _parse_date(end_date) if end_date else None
        end_daily = _extract_daily_from_period_snap(snap, end_date)

        _end = end_daily or {}
        totals = _totals(_end)
        total_mv = _total_market_value(end_daily)
        margin_loan = totals.get("margin_loan_balance")
        net_liquidation_value = None
        if isinstance(total_mv, (int, float)) and isinstance(margin_loan, (int, float)):
            net_liquidation_value = total_mv - margin_loan
        income = _income(_end)
        perf = _perf(_end)
        risk = _risk_flat(_end)
        rollups = _rollups(_end)
        holdings = _holdings_flat(end_daily) if end_daily else []

        interval = {
            "interval_label": period.get("label") or _interval_label(snapshot_type, _parse_date(start_date) or as_of_date, _parse_date(end_date) or as_of_date),
            "start_date": start_date,
            "end_date": end_date,
            "totals": {
                "total_market_value": total_mv,
                "cost_basis": totals.get("cost_basis"),
                "net_liquidation_value": _round(net_liquidation_value),
                "unrealized_pct": totals.get("unrealized_pct"),
                "unrealized_pnl": totals.get("unrealized_pnl"),
            },
            "income": income,
            "performance": {
                "pnl_dollar_period": None,
                "pnl_pct_period": None,
                "twr_1m_pct": perf.get("twr_1m_pct"),
                "twr_3m_pct": perf.get("twr_3m_pct"),
                "twr_6m_pct": perf.get("twr_6m_pct"),
                "twr_12m_pct": perf.get("twr_12m_pct"),
            },
            "risk": {
                "vol_30d_pct": risk.get("vol_30d_pct"),
                "vol_90d_pct": risk.get("vol_90d_pct"),
                "sharpe_1y": risk.get("sharpe_1y"),
                "sortino_1y": risk.get("sortino_1y"),
                "sortino_6m": risk.get("sortino_6m"),
                "sortino_3m": risk.get("sortino_3m"),
                "sortino_1m": risk.get("sortino_1m"),
                "sortino_sharpe_ratio": risk.get("sortino_sharpe_ratio"),
                "sortino_sharpe_divergence": risk.get("sortino_sharpe_divergence"),
                "portfolio_risk_quality": risk.get("portfolio_risk_quality"),
                "var_90_1d_pct": risk.get("var_90_1d_pct"),
                "var_95_1d_pct": risk.get("var_95_1d_pct"),
                "var_99_1d_pct": risk.get("var_99_1d_pct"),
                "var_95_1w_pct": risk.get("var_95_1w_pct"),
                "var_95_1m_pct": risk.get("var_95_1m_pct"),
                "cvar_90_1d_pct": risk.get("cvar_90_1d_pct"),
                "cvar_95_1d_pct": risk.get("cvar_95_1d_pct"),
                "cvar_99_1d_pct": risk.get("cvar_99_1d_pct"),
                "cvar_95_1w_pct": risk.get("cvar_95_1w_pct"),
                "cvar_95_1m_pct": risk.get("cvar_95_1m_pct"),
                "ulcer_index_1y": risk.get("ulcer_index_1y"),
                "omega_ratio_1y": risk.get("omega_ratio_1y"),
                "information_ratio_1y": risk.get("information_ratio_1y"),
                "tracking_error_1y_pct": risk.get("tracking_error_1y_pct"),
                "income_stability_score": risk.get("income_stability_score"),
                "max_drawdown_1y_pct": risk.get("max_drawdown_1y_pct"),
                "drawdown_duration_1y_days": risk.get("drawdown_duration_1y_days"),
                "calmar_1y": risk.get("calmar_1y"),
                "beta_portfolio": risk.get("beta_portfolio"),
            },
            "income_stability": rollups.get("income_stability"),
            "income_growth": rollups.get("income_growth"),
            "tail_risk": rollups.get("tail_risk"),
            "vs_benchmark": rollups.get("vs_benchmark"),
            "return_attribution_1m": rollups.get("return_attribution_1m"),
            "return_attribution_3m": rollups.get("return_attribution_3m"),
            "return_attribution_6m": rollups.get("return_attribution_6m"),
            "return_attribution_12m": rollups.get("return_attribution_12m"),
            "margin": {
                "margin_loan_balance": totals.get("margin_loan_balance"),
                "margin_to_portfolio_pct": totals.get("margin_to_portfolio_pct"),
                "ltv_pct": totals.get("margin_to_portfolio_pct"),
                "available_to_withdraw": 0.0 if totals.get("margin_loan_balance") is not None else None,
            },
            "margin_stress": _margin_stress(_end),
            "goal_progress": _goal_progress(_end),
            "goal_progress_net": _goal_progress_net(_end),
            "goal_tiers": _goal_tiers(_end),
            "goal_pace": _goal_pace(_end),
            "holdings": [
                {
                    "symbol": h.get("symbol"),
                    "weight_pct": _holding_metric(h, "weight_pct"),
                    "market_value": _holding_metric(h, "market_value"),
                    "pnl_pct": _holding_metric(h, "unrealized_pct"),
                    "pnl_dollar": _holding_metric(h, "unrealized_pnl"),
                    "projected_monthly_dividend": _holding_metric(h, "projected_monthly_dividend"),
                    "current_yield_pct": _holding_metric(h, "current_yield_pct"),
                    "yield_on_cost_pct": _holding_metric(h, "yield_on_cost_pct"),
                    "sharpe_1y": _holding_metric(h, "sharpe_1y"),
                    "sortino_1y": _holding_metric(h, "sortino_1y"),
                    "sortino_6m": _holding_metric(h, "sortino_6m"),
                    "sortino_3m": _holding_metric(h, "sortino_3m"),
                    "sortino_1m": _holding_metric(h, "sortino_1m"),
                    "risk_quality_score": _holding_metric(h, "risk_quality_score"),
                    "risk_quality_category": _holding_metric(h, "risk_quality_category"),
                    "volatility_profile": _holding_metric(h, "volatility_profile"),
                }
                for h in holdings
            ],
        }
        if end_dt and end_dt == as_of_date:
            interval["daily_snapshot"] = _strip_provenance(as_of_daily or end_daily)
        intervals.append(interval)
    return intervals


def build_period_snapshot(conn: sqlite3.Connection, snapshot_type: str, as_of: str | None = None, mode: str = "to_date"):
    if mode not in ("to_date", "final"):
        raise ValueError("mode must be to_date or final")
    if as_of:
        as_of_date = _parse_date(as_of)
    else:
        as_of_date = to_local_date(datetime.now(timezone.utc), settings.local_tz, settings.daily_cutover)
    if not as_of_date:
        raise ValueError("invalid as_of date")

    period_start, period_end = _period_bounds(snapshot_type, as_of_date)
    end_date = min(as_of_date, period_end) if mode == "to_date" else period_end

    dailies = _load_daily_snapshots(conn, period_start, end_date)
    if not dailies:
        raise ValueError("no daily snapshots found for period")

    as_of_daily = _load_daily_snapshot(conn, end_date)

    daily_dates = sorted({_as_of(s) for s in dailies if _as_of(s)})
    daily_dates = [d for d in daily_dates if d]
    observed_days = len(daily_dates)
    expected_days = _expected_days(snapshot_type, period_start, period_end)
    missing_dates = []
    if daily_dates:
        cursor = period_start
        while cursor <= end_date:
            if cursor not in daily_dates:
                missing_dates.append(cursor.isoformat())
            cursor += timedelta(days=1)

    coverage_pct = round((observed_days / expected_days) * 100, 1) if expected_days else 0.0
    is_complete = coverage_pct == 100.0

    start_snap = dailies[0]
    end_snap = dailies[-1]

    totals_start = _totals(start_snap) or {}
    totals_end = _totals(end_snap) or {}
    mv_start = _total_market_value(start_snap)
    mv_end = _total_market_value(end_snap)
    net_start = None
    net_end = None
    if isinstance(mv_start, (int, float)) and isinstance(totals_start.get("margin_loan_balance"), (int, float)):
        net_start = mv_start - totals_start.get("margin_loan_balance")
    if isinstance(mv_end, (int, float)) and isinstance(totals_end.get("margin_loan_balance"), (int, float)):
        net_end = mv_end - totals_end.get("margin_loan_balance")
    pnl = None
    pnl_pct = None
    if isinstance(mv_start, (int, float)) and isinstance(mv_end, (int, float)) and mv_start:
        pnl = mv_end - mv_start
        pnl_pct = (mv_end / mv_start - 1.0) * 100

    portfolio_data = {}
    for snap in dailies:
        dt = _parse_date(_as_of(snap))
        mv = _total_market_value(snap)
        if dt and isinstance(mv, (int, float)):
            portfolio_data[dt] = mv
    portfolio_series = pd.Series(portfolio_data).sort_index()
    if not portfolio_series.empty:
        portfolio_series.index = pd.to_datetime(portfolio_series.index)
    twr_period = metrics.twr(portfolio_series)
    twr_period_pct = None if twr_period is None else round(twr_period * 100, 2)
    if twr_period_pct is None and pnl_pct is not None:
        twr_period_pct = round(pnl_pct, 2)

    def _twr_window_delta(key):
        start_perf = _perf(start_snap)
        end_perf = _perf(end_snap)
        start_val = start_perf.get(key)
        end_val = end_perf.get(key)
        delta = None
        if isinstance(start_val, (int, float)) and isinstance(end_val, (int, float)):
            delta = round(end_val - start_val, 2)
        return start_val, end_val, delta

    twr_1m_start, twr_1m_end, twr_1m_delta = _twr_window_delta("twr_1m_pct")
    twr_3m_start, twr_3m_end, twr_3m_delta = _twr_window_delta("twr_3m_pct")
    twr_6m_start, twr_6m_end, twr_6m_delta = _twr_window_delta("twr_6m_pct")
    twr_12m_start, twr_12m_end, twr_12m_delta = _twr_window_delta("twr_12m_pct")

    risk_keys = [
        "vol_30d_pct",
        "vol_90d_pct",
        "sharpe_1y",
        "sortino_1y",
        "sortino_6m",
        "sortino_3m",
        "sortino_1m",
        "sortino_sharpe_ratio",
        "sortino_sharpe_divergence",
        "portfolio_risk_quality",
        "beta_portfolio",
        "corr_1y",
        "information_ratio_1y",
        "tracking_error_1y_pct",
        "ulcer_index_1y",
        "omega_ratio_1y",
        "pain_adjusted_return",
        "income_stability_score",
        "var_90_1d_pct",
        "var_95_1d_pct",
        "var_99_1d_pct",
        "var_95_1w_pct",
        "var_95_1m_pct",
        "cvar_90_1d_pct",
        "cvar_95_1d_pct",
        "cvar_99_1d_pct",
        "cvar_95_1w_pct",
        "cvar_95_1m_pct",
        "drawdown_duration_1y_days",
        "calmar_1y",
        "max_drawdown_1y_pct",
    ]
    risk_start_raw = _risk_flat(start_snap)
    risk_end_raw = _risk_flat(end_snap)
    risk_start = {key: risk_start_raw.get(key) for key in risk_keys}
    risk_end = {key: risk_end_raw.get(key) for key in risk_keys}
    risk_delta = {}
    for key in risk_keys:
        if isinstance(risk_start.get(key), (int, float)) and isinstance(risk_end.get(key), (int, float)):
            risk_delta[key] = round(risk_end.get(key) - risk_start.get(key), 2)

    macro_keys = ["ten_year_yield", "two_year_yield", "vix", "cpi_yoy", "hy_spread_bps", "macro_stress_score"]
    macro_start_raw = _macro_snapshot(start_snap)
    macro_end_raw = _macro_snapshot(end_snap)
    macro_start = {key: macro_start_raw.get(key) for key in macro_keys}
    macro_end = {key: macro_end_raw.get(key) for key in macro_keys}
    macro_avg = {}
    for key in macro_keys:
        values = [_macro_snapshot(s).get(key) for s in dailies]
        values = [v for v in values if isinstance(v, (int, float))]
        if values:
            macro_avg[key] = round(sum(values) / len(values), 2)

    # Calculate macro period stats with min/max/std/dates
    macro_period_stats = {}
    for key in ["vix", "ten_year_yield", "hy_spread_bps", "macro_stress_score"]:
        values_dict = {}
        values_list = []
        for snap in dailies:
            dt = _parse_date(_as_of(snap))
            val = _macro_snapshot(snap).get(key)
            if dt and isinstance(val, (int, float)):
                values_dict[dt] = val
                values_list.append(val)
        if values_list:
            macro_period_stats[key] = _calc_period_stats(values_list, values_dict)
    risk_stats = {}
    for key in ["vol_30d_pct", "vol_90d_pct", "sharpe_1y", "sortino_1y", "sortino_6m", "sortino_3m", "sortino_1m", "calmar_1y"]:
        values = [
            _risk_flat(s).get(key)
            for s in dailies
        ]
        values = [v for v in values if isinstance(v, (int, float))]
        if values:
            risk_stats[key] = {
                "avg": round(sum(values) / len(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
            }

    # Extract V5-compatible data for period summary
    start_income = _income(start_snap)
    end_income = _income(end_snap)
    start_gp = _goal_progress(start_snap)
    end_gp = _goal_progress(end_snap)
    start_gpn = _goal_progress_net(start_snap)
    end_gpn = _goal_progress_net(end_snap)
    start_holdings = _holdings_flat(start_snap)
    end_holdings = _holdings_flat(end_snap)

    # Read concentration from the assembled snapshot's allocation block (more reliable than recomputing from holdings)
    _start_conc = ((start_snap or {}).get("portfolio") or {}).get("allocation", {}).get("concentration", {})
    _end_conc = ((end_snap or {}).get("portfolio") or {}).get("allocation", {}).get("concentration", {})
    conc_top5_start = _start_conc.get("top5_weight_pct") or _top5_concentration(start_holdings)
    conc_top5_end = _end_conc.get("top5_weight_pct") or _top5_concentration(end_holdings)

    margin_start = totals_start.get("margin_loan_balance")
    margin_end = totals_end.get("margin_loan_balance")
    ltv_start = totals_start.get("margin_to_portfolio_pct")
    ltv_end = totals_end.get("margin_to_portfolio_pct")

    # Calculate period stats for portfolio values
    portfolio_values = [v for v in portfolio_data.values() if isinstance(v, (int, float))]
    mv_stats = _calc_period_stats(portfolio_values, portfolio_data) if portfolio_values else {}

    # Calculate period stats for net liquidation values
    net_values = {}
    for snap in dailies:
        dt = _parse_date(_as_of(snap))
        mv = _total_market_value(snap)
        if dt and isinstance(mv, (int, float)):
            margin = (_totals(snap) or {}).get("margin_loan_balance")
            if isinstance(margin, (int, float)):
                net_values[dt] = mv - margin
    net_list = [v for v in net_values.values() if isinstance(v, (int, float))]
    nlv_stats = _calc_period_stats(net_list, net_values) if net_list else {}

    # Calculate period stats for income
    projected_monthly_values = []
    yield_pct_values = []
    for snap in dailies:
        inc = _income(snap)
        pm = inc.get("projected_monthly_income")
        yp = inc.get("portfolio_current_yield_pct")
        if isinstance(pm, (int, float)):
            projected_monthly_values.append(pm)
        if isinstance(yp, (int, float)):
            yield_pct_values.append(yp)
    income_stats = {
        "projected_monthly_avg": _safe_avg(projected_monthly_values),
        "projected_monthly_min": min(projected_monthly_values) if projected_monthly_values else None,
        "projected_monthly_max": max(projected_monthly_values) if projected_monthly_values else None,
        "yield_pct_avg": _safe_avg(yield_pct_values),
        "yield_pct_min": min(yield_pct_values) if yield_pct_values else None,
        "yield_pct_max": max(yield_pct_values) if yield_pct_values else None,
    }

    # Calculate period stats for margin_to_portfolio_pct
    ltv_values = []
    ltv_values_dict = {}
    for snap in dailies:
        dt = _parse_date(_as_of(snap))
        totals = _totals(snap)
        ltv = totals.get("margin_to_portfolio_pct") or totals.get("ltv_pct")
        if dt and isinstance(ltv, (int, float)):
            ltv_values.append(ltv)
            ltv_values_dict[dt] = ltv
    ltv_stats = _calc_period_stats(ltv_values, ltv_values_dict) if ltv_values else {}
    
    # Calculate period stats for margin balance
    margin_balance_values = []
    margin_balance_dict = {}
    for snap in dailies:
        dt = _parse_date(_as_of(snap))
        mb = (_totals(snap) or {}).get("margin_loan_balance")
        if dt and isinstance(mb, (int, float)):
            margin_balance_values.append(mb)
            margin_balance_dict[dt] = mb
    margin_balance_stats = _calc_period_stats(margin_balance_values, margin_balance_dict) if margin_balance_values else {}
    
    # Calculate concentration metrics (top3, herfindahl)
    conc_top3_start = _start_conc.get("top3_weight_pct")
    conc_top3_end = _end_conc.get("top3_weight_pct")
    conc_top3_delta = round(conc_top3_end - conc_top3_start, 2) if isinstance(conc_top3_start, (int, float)) and isinstance(conc_top3_end, (int, float)) else None
    
    conc_herfindahl_start = _start_conc.get("herfindahl_index")
    conc_herfindahl_end = _end_conc.get("herfindahl_index")
    conc_herfindahl_delta = round(conc_herfindahl_end - conc_herfindahl_start, 3) if isinstance(conc_herfindahl_start, (int, float)) and isinstance(conc_herfindahl_end, (int, float)) else None
    
    # Calculate margin interest stats
    apr_values = []
    for snap in dailies:
        margin_guidance = (snap.get("margin") or {}).get("guidance") or {}
        rates = margin_guidance.get("rates") or {}
        apr = rates.get("apr_current_pct")
        if isinstance(apr, (int, float)):
            apr_values.append(apr)
    margin_interest_start_apr_pct = apr_values[0] if apr_values else None
    margin_interest_end_apr_pct = apr_values[-1] if apr_values else None
    margin_interest_avg_apr_pct = round(sum(apr_values) / len(apr_values), 2) if apr_values else None

    # Calculate interest metrics from margin balance time series
    margin_avg_daily_balance = round(sum(margin_balance_values) / len(margin_balance_values), 2) if margin_balance_values else None

    # Calculate total interest paid and annualized
    if margin_balance_values and margin_interest_avg_apr_pct:
        days_in_period = len(margin_balance_values)
        total_interest = sum([
            (balance * (margin_interest_avg_apr_pct / 100) / 365)
            for balance in margin_balance_values
        ])
        margin_interest_total_paid = round(total_interest, 2)
        margin_interest_annualized = round(total_interest * (365 / days_in_period), 2) if days_in_period else None
        margin_interest_avg_daily_cost = round(total_interest / days_in_period, 2) if days_in_period else None
    else:
        margin_interest_total_paid = None
        margin_interest_annualized = None
        margin_interest_avg_daily_cost = None
    
    # Calculate goal pace tier metrics
    start_pace = _goal_pace(start_snap)
    end_pace = _goal_pace(end_snap)
    start_pace_current = (start_pace or {}).get("current_pace", {}) or {}
    end_pace_current = (end_pace or {}).get("current_pace", {}) or {}
    pace_tier_start = start_pace_current.get("pct_of_tier_pace")
    if pace_tier_start is None:
        pace_tier_start = start_pace_current.get("pct_of_tier")
    pace_tier_end = end_pace_current.get("pct_of_tier_pace")
    if pace_tier_end is None:
        pace_tier_end = end_pace_current.get("pct_of_tier")
    pace_tier_delta = round(pace_tier_end - pace_tier_start, 2) if isinstance(pace_tier_start, (int, float)) and isinstance(pace_tier_end, (int, float)) else None
    pace_months_start = start_pace_current.get("months_ahead_behind")
    pace_months_end = end_pace_current.get("months_ahead_behind")
    pace_months_delta = (
        round(pace_months_end - pace_months_start, 2)
        if isinstance(pace_months_start, (int, float)) and isinstance(pace_months_end, (int, float))
        else None
    )
    pace_category_start = start_pace_current.get("pace_category")
    pace_category_end = end_pace_current.get("pace_category")

    # Calculate VaR/CVaR averages for period
    var_95_1d_values = []
    var_90_1d_values = []
    var_99_1d_values = []
    var_95_1w_values = []
    var_95_1m_values = []
    cvar_95_1d_values = []
    cvar_90_1d_values = []
    cvar_99_1d_values = []
    cvar_95_1w_values = []
    cvar_95_1m_values = []
    for snap in dailies:
        risk_data = _risk_flat(snap)
        if isinstance(risk_data.get("var_95_1d_pct"), (int, float)):
            var_95_1d_values.append(risk_data["var_95_1d_pct"])
        if isinstance(risk_data.get("var_90_1d_pct"), (int, float)):
            var_90_1d_values.append(risk_data["var_90_1d_pct"])
        if isinstance(risk_data.get("var_99_1d_pct"), (int, float)):
            var_99_1d_values.append(risk_data["var_99_1d_pct"])
        if isinstance(risk_data.get("var_95_1w_pct"), (int, float)):
            var_95_1w_values.append(risk_data["var_95_1w_pct"])
        if isinstance(risk_data.get("var_95_1m_pct"), (int, float)):
            var_95_1m_values.append(risk_data["var_95_1m_pct"])
        if isinstance(risk_data.get("cvar_95_1d_pct"), (int, float)):
            cvar_95_1d_values.append(risk_data["cvar_95_1d_pct"])
        if isinstance(risk_data.get("cvar_90_1d_pct"), (int, float)):
            cvar_90_1d_values.append(risk_data["cvar_90_1d_pct"])
        if isinstance(risk_data.get("cvar_99_1d_pct"), (int, float)):
            cvar_99_1d_values.append(risk_data["cvar_99_1d_pct"])
        if isinstance(risk_data.get("cvar_95_1w_pct"), (int, float)):
            cvar_95_1w_values.append(risk_data["cvar_95_1w_pct"])
        if isinstance(risk_data.get("cvar_95_1m_pct"), (int, float)):
            cvar_95_1m_values.append(risk_data["cvar_95_1m_pct"])
    
    var_95_1d_avg = round(sum(var_95_1d_values) / len(var_95_1d_values), 2) if var_95_1d_values else None
    var_90_1d_avg = round(sum(var_90_1d_values) / len(var_90_1d_values), 2) if var_90_1d_values else None
    var_99_1d_avg = round(sum(var_99_1d_values) / len(var_99_1d_values), 2) if var_99_1d_values else None
    var_95_1w_avg = round(sum(var_95_1w_values) / len(var_95_1w_values), 2) if var_95_1w_values else None
    var_95_1m_avg = round(sum(var_95_1m_values) / len(var_95_1m_values), 2) if var_95_1m_values else None
    cvar_95_1d_avg = round(sum(cvar_95_1d_values) / len(cvar_95_1d_values), 2) if cvar_95_1d_values else None
    cvar_90_1d_avg = round(sum(cvar_90_1d_values) / len(cvar_90_1d_values), 2) if cvar_90_1d_values else None
    cvar_99_1d_avg = round(sum(cvar_99_1d_values) / len(cvar_99_1d_values), 2) if cvar_99_1d_values else None
    cvar_95_1w_avg = round(sum(cvar_95_1w_values) / len(cvar_95_1w_values), 2) if cvar_95_1w_values else None
    cvar_95_1m_avg = round(sum(cvar_95_1m_values) / len(cvar_95_1m_values), 2) if cvar_95_1m_values else None

    # Calculate margin safety metrics
    margin_safety = {}
    buffer_values = []
    buffer_dates = []
    for snap in dailies:
        dt = _parse_date(_as_of(snap))
        margin_stress = _margin_stress(snap)
        buffer = ((margin_stress.get("stress_scenarios") or {}).get("margin_call_distance") or {}).get("buffer_to_margin_call_pct")
        if dt and isinstance(buffer, (int, float)):
            buffer_values.append(buffer)
            buffer_dates.append(dt)
    if buffer_values:
        min_buffer = min(buffer_values)
        min_buffer_idx = buffer_values.index(min_buffer)
        min_buffer_date = buffer_dates[min_buffer_idx]
        margin_safety = {
            "min_buffer_to_call_pct": round(min_buffer, 2),
            "min_buffer_date": min_buffer_date.isoformat(),
            "days_below_50pct_buffer": sum(1 for b in buffer_values if b < 50),
            "days_below_40pct_buffer": sum(1 for b in buffer_values if b < 40),
            "margin_call_events": sum(1 for b in buffer_values if b < 0),
        }

    # Calculate period-specific drawdown metrics
    period_drawdown = _calc_period_drawdown(portfolio_series)

    # Calculate VaR breach metrics
    var_95_values = []
    daily_returns = []
    dates_list = []
    for i in range(1, len(dailies)):
        prev_mv = _total_market_value(dailies[i-1])
        curr_mv = _total_market_value(dailies[i])
        if isinstance(prev_mv, (int, float)) and isinstance(curr_mv, (int, float)) and prev_mv > 0:
            daily_return = (curr_mv / prev_mv - 1.0) * 100
            daily_returns.append(daily_return)
            dates_list.append(_parse_date(_as_of(dailies[i])))
            # Get VaR 95 from risk data
            risk_data = _risk_flat(dailies[i])
            var_95 = risk_data.get("var_95_1d_pct")
            if isinstance(var_95, (int, float)):
                var_95_values.append((var_95, daily_return, dates_list[-1]))
    # Count days exceeding VaR 95
    days_exceeding_var_95 = sum(1 for var_95, ret, _ in var_95_values 
                                  if isinstance(var_95, (int, float)) and isinstance(ret, (int, float)) and ret < -abs(var_95))
    worst_day_return = min(daily_returns) if daily_returns else None
    worst_day_date = dates_list[daily_returns.index(worst_day_return)] if worst_day_return is not None and daily_returns.index(worst_day_return) < len(dates_list) else None
    best_day_return = max(daily_returns) if daily_returns else None
    best_day_date = dates_list[daily_returns.index(best_day_return)] if best_day_return is not None and daily_returns.index(best_day_return) < len(dates_list) else None

    # Generate activity section early so we can extract dividends_received_period
    activity = _generate_period_activity(conn, period_start, end_date)

    # Update interest metrics with calculated values
    activity["interest"]["avg_daily_balance"] = margin_avg_daily_balance
    activity["interest"]["avg_rate_pct"] = margin_interest_avg_apr_pct
    activity["interest"]["annualized"] = margin_interest_annualized
    if activity["interest"]["total_paid"] is None:
        activity["interest"]["total_paid"] = margin_interest_total_paid

    # Extract dividends received for period_summary
    dividends_received_period = activity.get("dividends", {}).get("total_received")

    period_summary = {
        "totals": {
            "start": {
                "total_market_value": mv_start,
                "net_liquidation_value": _round(net_start),
                "cost_basis": totals_start.get("cost_basis"),
                "unrealized_pct": totals_start.get("unrealized_pct"),
                "unrealized_pnl": totals_start.get("unrealized_pnl"),
                "margin_loan_balance": margin_start,
                "margin_to_portfolio_pct": ltv_start,
            },
            "end": {
                "total_market_value": mv_end,
                "net_liquidation_value": _round(net_end),
                "cost_basis": totals_end.get("cost_basis"),
                "unrealized_pct": totals_end.get("unrealized_pct"),
                "unrealized_pnl": totals_end.get("unrealized_pnl"),
                "margin_loan_balance": margin_end,
                "margin_to_portfolio_pct": ltv_end,
            },
            "delta": {
                "total_market_value": _round(pnl),
                "net_liquidation_value": _round(net_end - net_start, 2)
                if isinstance(net_end, (int, float)) and isinstance(net_start, (int, float))
                else None,
                "cost_basis": _round(totals_end.get("cost_basis") - totals_start.get("cost_basis"), 2)
                if isinstance(totals_end.get("cost_basis"), (int, float)) and isinstance(totals_start.get("cost_basis"), (int, float))
                else None,
                "unrealized_pct": _round(totals_end.get("unrealized_pct") - totals_start.get("unrealized_pct"), 2)
                if isinstance(totals_end.get("unrealized_pct"), (int, float)) and isinstance(totals_start.get("unrealized_pct"), (int, float))
                else None,
                "unrealized_pnl": _round(totals_end.get("unrealized_pnl") - totals_start.get("unrealized_pnl"), 2)
                if isinstance(totals_end.get("unrealized_pnl"), (int, float)) and isinstance(totals_start.get("unrealized_pnl"), (int, float))
                else None,
                "margin_loan_balance": _round(margin_end - margin_start, 2)
                if isinstance(margin_end, (int, float)) and isinstance(margin_start, (int, float))
                else None,
                "margin_to_portfolio_pct": _round(ltv_end - ltv_start, 2)
                if isinstance(ltv_end, (int, float)) and isinstance(ltv_start, (int, float))
                else None,
            },
        },
        "income": {
            "start": _income(start_snap),
            "end": _income(end_snap),
            "delta": {
                "projected_monthly_income": _round((end_income or {}).get("projected_monthly_income") - (start_income or {}).get("projected_monthly_income"), 2)
                if isinstance((end_income or {}).get("projected_monthly_income"), (int, float)) and isinstance((start_income or {}).get("projected_monthly_income"), (int, float))
                else None,
                "forward_12m_total": _round((end_income or {}).get("forward_12m_total") - (start_income or {}).get("forward_12m_total"), 2)
                if isinstance((end_income or {}).get("forward_12m_total"), (int, float)) and isinstance((start_income or {}).get("forward_12m_total"), (int, float))
                else None,
                "portfolio_current_yield_pct": _round((end_income or {}).get("portfolio_current_yield_pct") - (start_income or {}).get("portfolio_current_yield_pct"), 2)
                if isinstance((end_income or {}).get("portfolio_current_yield_pct"), (int, float)) and isinstance((start_income or {}).get("portfolio_current_yield_pct"), (int, float))
                else None,
                "portfolio_yield_on_cost_pct": _round((end_income or {}).get("portfolio_yield_on_cost_pct") - (start_income or {}).get("portfolio_yield_on_cost_pct"), 2)
                if isinstance((end_income or {}).get("portfolio_yield_on_cost_pct"), (int, float)) and isinstance((start_income or {}).get("portfolio_yield_on_cost_pct"), (int, float))
                else None,
            },
            "dividends_received_period": dividends_received_period,
        },
        "performance": {
            "period": {"twr_period_pct": twr_period_pct, "pnl_dollar_period": _round(pnl), "pnl_pct_period": _round(pnl_pct)},
            "twr_windows": {
                "twr_1m_pct_start": twr_1m_start,
                "twr_1m_pct_end": twr_1m_end,
                "twr_1m_pct_delta": twr_1m_delta,
                "twr_3m_pct_start": twr_3m_start,
                "twr_3m_pct_end": twr_3m_end,
                "twr_3m_pct_delta": twr_3m_delta,
                "twr_6m_pct_start": twr_6m_start,
                "twr_6m_pct_end": twr_6m_end,
                "twr_6m_pct_delta": twr_6m_delta,
                "twr_12m_pct_start": twr_12m_start,
                "twr_12m_pct_end": twr_12m_end,
                "twr_12m_pct_delta": twr_12m_delta,
            },
        },
        "risk": {"start": risk_start, "end": risk_end, "delta": risk_delta},
        "goal_progress": {
            "start": _goal_progress(start_snap),
            "end": _goal_progress(end_snap),
            "delta": {
                "progress_pct": _round((end_gp or {}).get("progress_pct") - (start_gp or {}).get("progress_pct"), 2)
                if isinstance((end_gp or {}).get("progress_pct"), (int, float)) and isinstance((start_gp or {}).get("progress_pct"), (int, float))
                else None,
                "current_projected_monthly": _round((end_gp or {}).get("current_projected_monthly") - (start_gp or {}).get("current_projected_monthly"), 2)
                if isinstance((end_gp or {}).get("current_projected_monthly"), (int, float)) and isinstance((start_gp or {}).get("current_projected_monthly"), (int, float))
                else None,
                "months_to_goal": _round((end_gp or {}).get("months_to_goal") - (start_gp or {}).get("months_to_goal"), 0)
                if isinstance((end_gp or {}).get("months_to_goal"), (int, float)) and isinstance((start_gp or {}).get("months_to_goal"), (int, float))
                else None,
            },
        },
        "goal_progress_net": {
            "start": _goal_progress_net(start_snap),
            "end": _goal_progress_net(end_snap),
            "delta": {
                "progress_pct": _round((end_gpn or {}).get("progress_pct") - (start_gpn or {}).get("progress_pct"), 2)
                if isinstance((end_gpn or {}).get("progress_pct"), (int, float)) and isinstance((start_gpn or {}).get("progress_pct"), (int, float))
                else None,
                "current_projected_monthly_net": _round((end_gpn or {}).get("current_projected_monthly_net") - (start_gpn or {}).get("current_projected_monthly_net"), 2)
                if isinstance((end_gpn or {}).get("current_projected_monthly_net"), (int, float)) and isinstance((start_gpn or {}).get("current_projected_monthly_net"), (int, float))
                else None,
            },
        },
        "goal_tiers": _goal_tiers(end_snap),
        "goal_pace": {
            **_goal_pace(end_snap),
            "pace_category": {
                "start": pace_category_start,
                "end": pace_category_end,
            },
            "months_ahead_behind": {
                "start": pace_months_start,
                "end": pace_months_end,
                "delta": pace_months_delta,
            },
            "tier_pace_pct": {
                "start": pace_tier_start,
                "end": pace_tier_end,
                "delta": pace_tier_delta,
            },
        },
        "composition": {
            "start": {
                "holding_count": len(start_holdings),
                "concentration_top3_pct": conc_top3_start,
                "concentration_top5_pct": conc_top5_start,
                "concentration_herfindahl": conc_herfindahl_start,
            },
            "end": {
                "holding_count": len(end_holdings),
                "concentration_top3_pct": conc_top3_end,
                "concentration_top5_pct": conc_top5_end,
                "concentration_herfindahl": conc_herfindahl_end,
            },
            "delta": {
                "holding_count": len(end_holdings) - len(start_holdings),
                "concentration_top3_pct": conc_top3_delta,
                "concentration_top5_pct": _round(conc_top5_end - conc_top5_start, 2)
                if isinstance(conc_top5_start, (int, float)) and isinstance(conc_top5_end, (int, float))
                else None,
                "concentration_herfindahl": conc_herfindahl_delta,
            },
        },
        "macro": {
            "start": {
                "ten_year_yield": macro_start.get("ten_year_yield"),
                "two_year_yield": macro_start.get("two_year_yield"),
                "vix": macro_start.get("vix"),
                "cpi_yoy": macro_start.get("cpi_yoy"),
            },
            "end": {
                "ten_year_yield": macro_end.get("ten_year_yield"),
                "two_year_yield": macro_end.get("two_year_yield"),
                "vix": macro_end.get("vix"),
                "cpi_yoy": macro_end.get("cpi_yoy"),
            },
            "avg": macro_avg,
            "delta": {
                "ten_year_yield": _round(macro_end.get("ten_year_yield") - macro_start.get("ten_year_yield"), 2)
                if isinstance(macro_end.get("ten_year_yield"), (int, float)) and isinstance(macro_start.get("ten_year_yield"), (int, float))
                else None,
                "two_year_yield": _round(macro_end.get("two_year_yield") - macro_start.get("two_year_yield"), 2)
                if isinstance(macro_end.get("two_year_yield"), (int, float)) and isinstance(macro_start.get("two_year_yield"), (int, float))
                else None,
                "vix": _round(macro_end.get("vix") - macro_start.get("vix"), 2)
                if isinstance(macro_end.get("vix"), (int, float)) and isinstance(macro_start.get("vix"), (int, float))
                else None,
                "cpi_yoy": _round(macro_end.get("cpi_yoy") - macro_start.get("cpi_yoy"), 2)
                if isinstance(macro_end.get("cpi_yoy"), (int, float)) and isinstance(macro_start.get("cpi_yoy"), (int, float))
                else None,
            },
        },
    }
    if risk_stats:
        period_summary["risk"]["stats"] = risk_stats

    # Add period stats for portfolio values
    period_summary["period_stats"] = {
        "market_value": mv_stats,
        "net_liquidation_value": nlv_stats,
        "projected_monthly_income": {
            "avg": income_stats.get("projected_monthly_avg"),
            "min": income_stats.get("projected_monthly_min"),
            "max": income_stats.get("projected_monthly_max"),
        },
        "yield_pct": {
            "avg": income_stats.get("yield_pct_avg"),
            "min": income_stats.get("yield_pct_min"),
            "max": income_stats.get("yield_pct_max"),
        },
        "margin_to_portfolio_pct": ltv_stats if ltv_stats else {},
        "margin_balance": margin_balance_stats if margin_balance_stats else {},
        "var_95_1d_avg": var_95_1d_avg,
        "var_90_1d_avg": var_90_1d_avg,
        "var_99_1d_avg": var_99_1d_avg,
        "var_95_1w_avg": var_95_1w_avg,
        "var_95_1m_avg": var_95_1m_avg,
        "cvar_95_1d_avg": cvar_95_1d_avg,
        "cvar_90_1d_avg": cvar_90_1d_avg,
        "cvar_99_1d_avg": cvar_99_1d_avg,
        "cvar_95_1w_avg": cvar_95_1w_avg,
        "cvar_95_1m_avg": cvar_95_1m_avg,
        "margin_interest": {
            "start_apr_pct": margin_interest_start_apr_pct,
            "end_apr_pct": margin_interest_end_apr_pct,
            "avg_apr_pct": margin_interest_avg_apr_pct,
        },
    }

    # Add macro period stats
    if macro_period_stats:
        period_summary["macro_period_stats"] = macro_period_stats

    # Add period-specific drawdown metrics
    period_summary["period_drawdown"] = period_drawdown

    # Add VaR breach metrics
    period_summary["var_breach"] = {
        "days_exceeding_var_95": days_exceeding_var_95,
        "worst_day_return_pct": worst_day_return,
        "worst_day_date": worst_day_date.isoformat() if worst_day_date else None,
        "best_day_return_pct": best_day_return,
        "best_day_date": best_day_date.isoformat() if best_day_date else None,
    }

    # Add margin safety metrics
    if margin_safety:
        period_summary["margin_safety"] = margin_safety

    # Add activity section (already generated earlier)
    period_summary["activity"] = activity

    summary_id = f"{snapshot_type}_{end_date.isoformat()}"
    benchmark_symbol = settings.benchmark_primary

    intervals = None
    if snapshot_type == "monthly":
        week_start = end_date - timedelta(days=end_date.weekday())
        weekly_snaps = _load_period_snapshots(conn, "weekly", period_start, end_date)
        completed = [s for s in weekly_snaps if _parse_date((s.get("period") or {}).get("end_date")) and _parse_date((s.get("period") or {}).get("end_date")) < week_start]
        current_wtd = build_period_snapshot(conn, "weekly", as_of=end_date.isoformat(), mode="to_date")
        base_snaps = completed + [current_wtd]
        if base_snaps:
            intervals = _intervals_from_periods(base_snaps, snapshot_type, end_date, as_of_daily)
    elif snapshot_type in ("quarterly", "yearly"):
        month_start = end_date.replace(day=1)
        monthly_snaps = _load_period_snapshots(conn, "monthly", period_start, end_date)
        completed = [s for s in monthly_snaps if _parse_date((s.get("period") or {}).get("end_date")) and _parse_date((s.get("period") or {}).get("end_date")) < month_start]
        current_mtd = build_period_snapshot(conn, "monthly", as_of=end_date.isoformat(), mode="to_date")
        base_snaps = completed + [current_mtd]
        if base_snaps:
            intervals = _intervals_from_periods(base_snaps, snapshot_type, end_date, as_of_daily)

    if intervals is None:
        intervals = _intervals(dailies, snapshot_type, end_date, as_of_daily)

    summary = _coverage_summary(dailies)

    snapshot = {
        "summary_id": summary_id,
        "snapshot_type": snapshot_type,
        "snapshot_mode": mode,
        "as_of": end_date.isoformat(),
        "period": {
            "label": _period_label(snapshot_type, end_date),
            "start_date": period_start.isoformat(),
            "end_date": end_date.isoformat(),
            "expected_days": expected_days,
            "observed_days": observed_days,
            "coverage_pct": coverage_pct,
            "is_complete": is_complete,
            "missing_dates": missing_dates,
        },
        "benchmark": _benchmark_block(benchmark_symbol, period_start, end_date),
        "period_summary": period_summary,
        "portfolio_changes": _portfolio_changes(start_holdings, end_holdings, dailies),
        "intervals": intervals,
        "summary": summary,
    }
    return snapshot
