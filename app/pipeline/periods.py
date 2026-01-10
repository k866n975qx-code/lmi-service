from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from collections import defaultdict

import pandas as pd

from ..config import settings
from ..utils import to_local_date
from .market import MarketData
from . import metrics


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
    val = snapshot.get("total_market_value")
    if isinstance(val, (int, float)):
        return float(val)
    totals = snapshot.get("totals") or {}
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


def _coverage_summary(dailies):
    derived = []
    pulled = []
    for snap in dailies:
        cov = snap.get("coverage") or {}
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
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT as_of_date_local, payload_json
        FROM snapshot_daily_current
        WHERE as_of_date_local BETWEEN ? AND ?
        ORDER BY as_of_date_local ASC
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    snapshots = []
    for as_of_date, payload_json in rows:
        try:
            snap = json.loads(payload_json)
        except Exception:
            continue
        snap["as_of_date_local"] = as_of_date
        snapshots.append(snap)
    return snapshots


def _load_period_snapshots(conn: sqlite3.Connection, period_type: str, start: date, end: date):
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT period_start_date, period_end_date, payload_json
        FROM snapshots
        WHERE period_type=? AND period_end_date BETWEEN ? AND ?
        ORDER BY period_end_date ASC
        """,
        (period_type, start.isoformat(), end.isoformat()),
    ).fetchall()
    out = []
    for start_date, end_date, payload_json in rows:
        try:
            snap = json.loads(payload_json)
        except Exception:
            continue
        if not isinstance(snap, dict) or "snapshot_type" not in snap:
            # Skip legacy placeholder snapshots.
            continue
        snap["_period_start"] = start_date
        snap["_period_end"] = end_date
        out.append(snap)
    return out


def _load_daily_snapshot(conn: sqlite3.Connection, as_of: date):
    cur = conn.cursor()
    row = cur.execute(
        "SELECT payload_json FROM snapshot_daily_current WHERE as_of_date_local=?",
        (as_of.isoformat(),),
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def _daily_series(dailies, key_path):
    data = {}
    for snap in dailies:
        as_of = snap.get("as_of") or snap.get("as_of_date_local")
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


def _portfolio_changes(start_holdings, end_holdings):
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
        if s is None and e is not None:
            added.append({"symbol": sym, "weight_end_pct": e.get("weight_pct")})
        elif e is None and s is not None:
            removed.append({"symbol": sym, "weight_start_pct": s.get("weight_pct")})
        else:
            w_start = s.get("weight_pct")
            w_end = e.get("weight_pct")
            if isinstance(w_start, (int, float)) and isinstance(w_end, (int, float)):
                delta = w_end - w_start
                if delta > 0.01:
                    increases.append({"symbol": sym, "weight_start_pct": w_start, "weight_end_pct": w_end, "weight_delta_pct": round(delta, 2)})
                elif delta < -0.01:
                    decreases.append({"symbol": sym, "weight_start_pct": w_start, "weight_end_pct": w_end, "weight_delta_pct": round(delta, 2)})

        if s and e:
            mv_start = s.get("market_value")
            mv_end = e.get("market_value")
            if isinstance(mv_start, (int, float)) and isinstance(mv_end, (int, float)) and mv_start:
                pnl = mv_end - mv_start
                pnl_pct = (mv_end / mv_start - 1.0) * 100
                item = {"symbol": sym, "pnl_pct_period": round(pnl_pct, 2), "pnl_dollar_period": round(pnl, 2)}
                if pnl >= 0:
                    gainers.append(item)
                else:
                    losers.append(item)

    gainers = sorted(gainers, key=lambda x: x["pnl_dollar_period"], reverse=True)[:5]
    losers = sorted(losers, key=lambda x: x["pnl_dollar_period"])[:5]

    return {
        "holdings_added": added,
        "holdings_removed": removed,
        "weight_increases": increases,
        "weight_decreases": decreases,
        "top_gainers": gainers,
        "top_losers": losers,
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
        dt = _parse_date(snap.get("as_of")) or _parse_date(snap.get("as_of_date_local"))
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
        start_totals = start_snap.get("totals", {}) if start_snap else {}
        end_totals = end_snap.get("totals", {}) if end_snap else {}
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
            "income": end_snap.get("income"),
            "performance": {
                "pnl_dollar_period": _round(pnl),
                "pnl_pct_period": _round(pnl_pct),
                "twr_1m_pct": (end_snap.get("portfolio_rollups") or {}).get("performance", {}).get("twr_1m_pct"),
                "twr_3m_pct": (end_snap.get("portfolio_rollups") or {}).get("performance", {}).get("twr_3m_pct"),
                "twr_12m_pct": (end_snap.get("portfolio_rollups") or {}).get("performance", {}).get("twr_12m_pct"),
            },
            "risk": {
                "sharpe_1y": (end_snap.get("portfolio_rollups") or {}).get("risk", {}).get("sharpe_1y"),
                "sortino_1y": (end_snap.get("portfolio_rollups") or {}).get("risk", {}).get("sortino_1y"),
                "sortino_6m": (end_snap.get("portfolio_rollups") or {}).get("risk", {}).get("sortino_6m"),
                "sortino_3m": (end_snap.get("portfolio_rollups") or {}).get("risk", {}).get("sortino_3m"),
                "sortino_1m": (end_snap.get("portfolio_rollups") or {}).get("risk", {}).get("sortino_1m"),
                "sortino_sharpe_ratio": (end_snap.get("portfolio_rollups") or {}).get("risk", {}).get("sortino_sharpe_ratio"),
                "sortino_sharpe_divergence": (end_snap.get("portfolio_rollups") or {}).get("risk", {}).get("sortino_sharpe_divergence"),
                "cvar_95_1d_pct": (end_snap.get("portfolio_rollups") or {}).get("risk", {}).get("cvar_95_1d_pct"),
                "max_drawdown_1y_pct": (end_snap.get("portfolio_rollups") or {}).get("risk", {}).get("max_drawdown_1y_pct"),
                "drawdown_duration_1y_days": (end_snap.get("portfolio_rollups") or {}).get("risk", {}).get("drawdown_duration_1y_days"),
                "calmar_1y": (end_snap.get("portfolio_rollups") or {}).get("risk", {}).get("calmar_1y"),
            },
            "margin": {
                "margin_loan_balance": end_totals.get("margin_loan_balance"),
                "margin_to_portfolio_pct": end_totals.get("margin_to_portfolio_pct"),
                "ltv_pct": end_totals.get("margin_to_portfolio_pct"),
                "available_to_withdraw": 0.0 if end_totals.get("margin_loan_balance") is not None else None,
            },
            "goal_progress": end_snap.get("goal_progress"),
            "goal_progress_net": end_snap.get("goal_progress_net"),
            "holdings": [
                {
                    "symbol": h.get("symbol"),
                    "weight_pct": h.get("weight_pct"),
                    "market_value": h.get("market_value"),
                    "pnl_pct": h.get("unrealized_pct"),
                    "pnl_dollar": h.get("unrealized_pnl"),
                    "projected_monthly_dividend": h.get("projected_monthly_dividend"),
                    "current_yield_pct": h.get("current_yield_pct"),
                    "yield_on_cost_pct": h.get("yield_on_cost_pct"),
                    "sharpe_1y": h.get("sharpe_1y"),
                    "sortino_1y": h.get("sortino_1y"),
                    "sortino_6m": h.get("sortino_6m"),
                    "sortino_3m": h.get("sortino_3m"),
                    "sortino_1m": h.get("sortino_1m"),
                    "risk_quality_score": h.get("risk_quality_score"),
                    "risk_quality_category": h.get("risk_quality_category"),
                    "volatility_profile": h.get("volatility_profile"),
                }
                for h in (end_snap.get("holdings") or [])
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

        totals = (end_daily or {}).get("totals") or {}
        total_mv = _total_market_value(end_daily)
        margin_loan = totals.get("margin_loan_balance")
        net_liquidation_value = None
        if isinstance(total_mv, (int, float)) and isinstance(margin_loan, (int, float)):
            net_liquidation_value = total_mv - margin_loan
        income = (end_daily or {}).get("income")
        perf = (end_daily or {}).get("portfolio_rollups", {}).get("performance") if end_daily else {}
        risk = (end_daily or {}).get("portfolio_rollups", {}).get("risk") if end_daily else {}
        holdings = (end_daily or {}).get("holdings") or []

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
                "twr_1m_pct": perf.get("twr_1m_pct") if perf else None,
                "twr_3m_pct": perf.get("twr_3m_pct") if perf else None,
                "twr_12m_pct": perf.get("twr_12m_pct") if perf else None,
            },
            "risk": {
                "sharpe_1y": risk.get("sharpe_1y") if risk else None,
                "sortino_1y": risk.get("sortino_1y") if risk else None,
                "sortino_6m": risk.get("sortino_6m") if risk else None,
                "sortino_3m": risk.get("sortino_3m") if risk else None,
                "sortino_1m": risk.get("sortino_1m") if risk else None,
                "sortino_sharpe_ratio": risk.get("sortino_sharpe_ratio") if risk else None,
                "sortino_sharpe_divergence": risk.get("sortino_sharpe_divergence") if risk else None,
                "cvar_95_1d_pct": risk.get("cvar_95_1d_pct") if risk else None,
                "max_drawdown_1y_pct": risk.get("max_drawdown_1y_pct") if risk else None,
                "drawdown_duration_1y_days": risk.get("drawdown_duration_1y_days") if risk else None,
                "calmar_1y": risk.get("calmar_1y") if risk else None,
            },
            "margin": {
                "margin_loan_balance": totals.get("margin_loan_balance"),
                "margin_to_portfolio_pct": totals.get("margin_to_portfolio_pct"),
                "ltv_pct": totals.get("margin_to_portfolio_pct"),
                "available_to_withdraw": 0.0 if totals.get("margin_loan_balance") is not None else None,
            },
            "goal_progress": (end_daily or {}).get("goal_progress"),
            "goal_progress_net": (end_daily or {}).get("goal_progress_net"),
            "holdings": [
                {
                    "symbol": h.get("symbol"),
                    "weight_pct": h.get("weight_pct"),
                    "market_value": h.get("market_value"),
                    "pnl_pct": h.get("unrealized_pct"),
                    "pnl_dollar": h.get("unrealized_pnl"),
                    "projected_monthly_dividend": h.get("projected_monthly_dividend"),
                    "current_yield_pct": h.get("current_yield_pct"),
                    "yield_on_cost_pct": h.get("yield_on_cost_pct"),
                    "sharpe_1y": h.get("sharpe_1y"),
                    "sortino_1y": h.get("sortino_1y"),
                    "sortino_6m": h.get("sortino_6m"),
                    "sortino_3m": h.get("sortino_3m"),
                    "sortino_1m": h.get("sortino_1m"),
                    "risk_quality_score": h.get("risk_quality_score"),
                    "risk_quality_category": h.get("risk_quality_category"),
                    "volatility_profile": h.get("volatility_profile"),
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

    daily_dates = sorted({_parse_date(s.get("as_of") or s.get("as_of_date_local")) for s in dailies if _parse_date(s.get("as_of") or s.get("as_of_date_local"))})
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

    totals_start = start_snap.get("totals", {})
    totals_end = end_snap.get("totals", {})
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
        dt = _parse_date(snap.get("as_of") or snap.get("as_of_date_local"))
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
        start_val = (start_snap.get("portfolio_rollups") or {}).get("performance", {}).get(key)
        end_val = (end_snap.get("portfolio_rollups") or {}).get("performance", {}).get(key)
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
        "cvar_95_1d_pct",
        "drawdown_duration_1y_days",
        "calmar_1y",
        "max_drawdown_1y_pct",
    ]
    risk_start_raw = (start_snap.get("portfolio_rollups") or {}).get("risk", {})
    risk_end_raw = (end_snap.get("portfolio_rollups") or {}).get("risk", {})
    risk_start = {key: risk_start_raw.get(key) for key in risk_keys}
    risk_end = {key: risk_end_raw.get(key) for key in risk_keys}
    risk_delta = {}
    for key in risk_keys:
        if isinstance(risk_start.get(key), (int, float)) and isinstance(risk_end.get(key), (int, float)):
            risk_delta[key] = round(risk_end.get(key) - risk_start.get(key), 2)

    macro_keys = ["ten_year_yield", "two_year_yield", "vix", "cpi_yoy"]
    macro_start_raw = (start_snap.get("macro") or {}).get("snapshot", {}) or {}
    macro_end_raw = (end_snap.get("macro") or {}).get("snapshot", {}) or {}
    macro_start = {key: macro_start_raw.get(key) for key in macro_keys}
    macro_end = {key: macro_end_raw.get(key) for key in macro_keys}
    macro_avg = {}
    for key in macro_keys:
        values = [s.get("macro", {}).get("snapshot", {}).get(key) for s in dailies]
        values = [v for v in values if isinstance(v, (int, float))]
        if values:
            macro_avg[key] = round(sum(values) / len(values), 2)
    risk_stats = {}
    for key in ["sortino_1y", "sortino_6m", "sortino_3m", "sortino_1m"]:
        values = [
            (s.get("portfolio_rollups") or {}).get("risk", {}).get(key)
            for s in dailies
        ]
        values = [v for v in values if isinstance(v, (int, float))]
        if values:
            risk_stats[key] = {
                "avg": round(sum(values) / len(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
            }

    period_summary = {
        "totals": {
            "start": {
                "total_market_value": mv_start,
                "net_liquidation_value": _round(net_start),
                "cost_basis": totals_start.get("cost_basis"),
                "unrealized_pct": totals_start.get("unrealized_pct"),
                "unrealized_pnl": totals_start.get("unrealized_pnl"),
            },
            "end": {
                "total_market_value": mv_end,
                "net_liquidation_value": _round(net_end),
                "cost_basis": totals_end.get("cost_basis"),
                "unrealized_pct": totals_end.get("unrealized_pct"),
                "unrealized_pnl": totals_end.get("unrealized_pnl"),
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
            },
        },
        "income": {
            "start": start_snap.get("income"),
            "end": end_snap.get("income"),
            "delta": {
                "projected_monthly_income": _round(end_snap.get("income", {}).get("projected_monthly_income") - start_snap.get("income", {}).get("projected_monthly_income"), 2)
                if isinstance(end_snap.get("income", {}).get("projected_monthly_income"), (int, float)) and isinstance(start_snap.get("income", {}).get("projected_monthly_income"), (int, float))
                else None,
                "forward_12m_total": _round(end_snap.get("income", {}).get("forward_12m_total") - start_snap.get("income", {}).get("forward_12m_total"), 2)
                if isinstance(end_snap.get("income", {}).get("forward_12m_total"), (int, float)) and isinstance(start_snap.get("income", {}).get("forward_12m_total"), (int, float))
                else None,
                "portfolio_current_yield_pct": _round(end_snap.get("income", {}).get("portfolio_current_yield_pct") - start_snap.get("income", {}).get("portfolio_current_yield_pct"), 2)
                if isinstance(end_snap.get("income", {}).get("portfolio_current_yield_pct"), (int, float)) and isinstance(start_snap.get("income", {}).get("portfolio_current_yield_pct"), (int, float))
                else None,
                "portfolio_yield_on_cost_pct": _round(end_snap.get("income", {}).get("portfolio_yield_on_cost_pct") - start_snap.get("income", {}).get("portfolio_yield_on_cost_pct"), 2)
                if isinstance(end_snap.get("income", {}).get("portfolio_yield_on_cost_pct"), (int, float)) and isinstance(start_snap.get("income", {}).get("portfolio_yield_on_cost_pct"), (int, float))
                else None,
            },
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
            "start": start_snap.get("goal_progress"),
            "end": end_snap.get("goal_progress"),
            "delta": {
                "progress_pct": _round((end_snap.get("goal_progress") or {}).get("progress_pct") - (start_snap.get("goal_progress") or {}).get("progress_pct"), 2)
                if isinstance((end_snap.get("goal_progress") or {}).get("progress_pct"), (int, float)) and isinstance((start_snap.get("goal_progress") or {}).get("progress_pct"), (int, float))
                else None,
                "current_projected_monthly": _round((end_snap.get("goal_progress") or {}).get("current_projected_monthly") - (start_snap.get("goal_progress") or {}).get("current_projected_monthly"), 2)
                if isinstance((end_snap.get("goal_progress") or {}).get("current_projected_monthly"), (int, float)) and isinstance((start_snap.get("goal_progress") or {}).get("current_projected_monthly"), (int, float))
                else None,
                "months_to_goal": _round((end_snap.get("goal_progress") or {}).get("months_to_goal") - (start_snap.get("goal_progress") or {}).get("months_to_goal"), 0)
                if isinstance((end_snap.get("goal_progress") or {}).get("months_to_goal"), (int, float)) and isinstance((start_snap.get("goal_progress") or {}).get("months_to_goal"), (int, float))
                else None,
            },
        },
        "goal_progress_net": {
            "start": start_snap.get("goal_progress_net"),
            "end": end_snap.get("goal_progress_net"),
            "delta": {
                "progress_pct": _round((end_snap.get("goal_progress_net") or {}).get("progress_pct") - (start_snap.get("goal_progress_net") or {}).get("progress_pct"), 2)
                if isinstance((end_snap.get("goal_progress_net") or {}).get("progress_pct"), (int, float)) and isinstance((start_snap.get("goal_progress_net") or {}).get("progress_pct"), (int, float))
                else None,
                "current_projected_monthly_net": _round((end_snap.get("goal_progress_net") or {}).get("current_projected_monthly_net") - (start_snap.get("goal_progress_net") or {}).get("current_projected_monthly_net"), 2)
                if isinstance((end_snap.get("goal_progress_net") or {}).get("current_projected_monthly_net"), (int, float)) and isinstance((start_snap.get("goal_progress_net") or {}).get("current_projected_monthly_net"), (int, float))
                else None,
            },
        },
        "composition": {
            "start": {
                "holding_count": len(start_snap.get("holdings") or []),
                "concentration_top5_pct": _top5_concentration(start_snap.get("holdings") or []),
            },
            "end": {
                "holding_count": len(end_snap.get("holdings") or []),
                "concentration_top5_pct": _top5_concentration(end_snap.get("holdings") or []),
            },
            "delta": {
                "holding_count": len(end_snap.get("holdings") or []) - len(start_snap.get("holdings") or []),
                "concentration_top5_pct": _round(_top5_concentration(end_snap.get("holdings") or []) - _top5_concentration(start_snap.get("holdings") or []), 2)
                if _top5_concentration(start_snap.get("holdings") or []) is not None and _top5_concentration(end_snap.get("holdings") or []) is not None
                else None,
            },
        },
        "macro": {
            "start": {
                "ten_year_yield": macro_start.get("ten_year_yield"),
                "two_year_yield": macro_start.get("two_year_yield"),
            },
            "end": {
                "ten_year_yield": macro_end.get("ten_year_yield"),
                "two_year_yield": macro_end.get("two_year_yield"),
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

    summary_id = f"{snapshot_type}_{end_date.isoformat()}"
    benchmark_symbol = settings.benchmark_primary

    intervals = None
    if snapshot_type == "monthly":
        week_start = end_date - timedelta(days=end_date.weekday())
        weekly_snaps = _load_period_snapshots(conn, "WEEK", period_start, end_date)
        completed = [s for s in weekly_snaps if _parse_date((s.get("period") or {}).get("end_date")) and _parse_date((s.get("period") or {}).get("end_date")) < week_start]
        current_wtd = build_period_snapshot(conn, "weekly", as_of=end_date.isoformat(), mode="to_date")
        base_snaps = completed + [current_wtd]
        if base_snaps:
            intervals = _intervals_from_periods(base_snaps, snapshot_type, end_date, as_of_daily)
    elif snapshot_type in ("quarterly", "yearly"):
        month_start = end_date.replace(day=1)
        monthly_snaps = _load_period_snapshots(conn, "MONTH", period_start, end_date)
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
        "portfolio_changes": _portfolio_changes(start_snap.get("holdings") or [], end_snap.get("holdings") or []),
        "intervals": intervals,
        "summary": summary,
    }
    return snapshot
